"""
Contains utilities for orchestration tasks, such as downloading files, calling 
Bash utilities to extract archives and move files around and so on.
"""
import asyncio
import itertools
import re
import subprocess
import uuid
import datetime
import logging

import aiofiles as aiof
import aiohttp
import numpy as np
import pandas as pd
import timbermafia as tm

import gim_cv.config as cfg

from pathlib import Path
from functools import partial
from tqdm import tqdm
from zipfile import ZipFile


log = logging.getLogger(__name__)


async def download_images_to_directory(file_urls, target_directory, overwrite=False):
    """
    Simple example implementation where we define a coroutine 
    which downloads all the files in a list of URLs file_urls and saves 
    these to target_directory
    """
    async with aiohttp.ClientSession() as sesh:
        return await download_files(
            urls=file_urls,
            save_paths=[Path(f'{target_directory}/img_{i}.tif') for i in range(len(file_urls))],
            session=sesh,
            overwrite=overwrite,
            concurrent_download_tasks=10,
            timeout=10,
            chunk_size=512_000
        )



async def download_worker(queue,
                          dlq,
                          session,
                          succeeded,
                          chunk_size=512_000,
                          timeout=7200,
                          on_fail_slowdown=5,
                          use_temp_filename=True):
    """ read url and path from queue, perform download request and insert
        path to downloaded file to the succeeded list

        in case of an issue, add url land save path to DLQ, sleep then
        go to next url in queue
    """
    while True:
        try:
            # get a download url and desired file save path
            url, save_path = await queue.get()
            # fire off request
            path = await download_file(url,
                                       save_path,
                                       session,
                                       chunk_size,
                                       timeout,
                                       use_temp_filename)
        except (aiohttp.ClientResponseError, asyncio.TimeoutError) as e:
            log.error(f"problem with request to url {url}, moving to DLQ: {e}")
            # put args in dlq for other consumers to handle
            await dlq.put((url, save_path))
            await asyncio.sleep(on_fail_slowdown)
            log.debug(f"notifying task {len(succeeded)} done for "
                      f"{save_path.parts[-1]}")
            queue.task_done()
        else:
            # put successful path in succeeded
            succeeded.append(path)
            # notify queue
            log.debug(f"notifying task {len(succeeded)} succeeded for "
                      f"{save_path.parts[-1]}")
            queue.task_done()

            
async def extract_translate_worker(queue, succeeded, overwrite=False, delete_originals=True):
    """ execute unzip + translate jobs on queue
    """
    worker_id = uuid.uuid4()
    while True:
        try:
            # get a download url and desired file save path
            archive_path, target_scale, tol = await queue.get()
            # queue up extraction and gdal_translat jobs
            raster_paths = await extract_rasters_and_translate(
                archive_path,
                target_scale=target_scale,
                delete_originals=delete_originals,
                tol=tol
            )
            # put successful path in succeeded
            succeeded.extend(raster_paths)
        except asyncio.CancelledError:
            log.debug(f"worker {worker_id} exiting...")
        except Exception as e:
            log.error(e)
            log.error(f"problem with extract+translate on url {archive_path}")
            log.error(f"worker {worker_id} notifying queue of failed task")
            queue.task_done()
        else:
            # notify queue
            log.debug(f"worker {worker_id} notifying queue successful task")
            queue.task_done()


async def produce(queue, items):
    for item in items:
        # if the queue is full (reached maxsize) this line will be blocked
        # until a consumer will finish processing a url
        await queue.put(item)
    log.debug("all jobs in queue")


async def download_file(file_url,
                        save_path,
                        session,
                        chunk_size=512_000,
                        timeout=7200,
                        use_temp_filename=True):
    """ coroutine to download a file in chunks of chunk_size at a given url,
        saving to save_path

        returns the file path upon completion for convenience
    """
    log.debug(f"download file at: {file_url} to local file: {save_path}")
    async with session.get(file_url, timeout=timeout) as resp:
        total_size = int(resp.headers.get('content-length', 0))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if use_temp_filename:
            save_path_tmp = Path(str(save_path) + '.tmp')
        else:
            save_path_tmp = save_path
        async with aiof.open(save_path_tmp, "wb") as fd:
            # raise aiohttp.ClientResponseError immediately if invalid response
            resp.raise_for_status()
            log.debug(f"start writing file {save_path_tmp}")
            while True:
                chunk = await resp.content.read(chunk_size)
                if not chunk:
                    break
                await fd.write(chunk)
        if use_temp_filename:
            mv_cmd = f'mv {save_path_tmp} {save_path}'
            log.debug(f"moving temporary file...: {mv_cmd}")
            await open_subprocess(mv_cmd)
        log.debug(f"download complete: {file_url} -> {save_path}")
    return save_path


async def handle_failed_worker(dlq):
    while True:
        # get a download url and desired file save path
        url, save_path = await dlq.get()
        log.error(f"Issue with download url {url}. Nothing saved to {save_path}.")
        dlq.task_done()

        
async def download_files(urls,
                         save_paths,
                         session,
                         overwrite=False,
                         concurrent_download_tasks=10,
                         timeout=7200,
                         chunk_size=512_000):
    # skip files which are already downloaded
    if not overwrite:
        not_existing_ixs = [save_paths.index(p) for p in save_paths if not p.exists()]
        save_paths = [save_paths[ix] for ix in not_existing_ixs]
        urls = [urls[ix] for ix in not_existing_ixs]
    # if nothing to download, stop here
    if not urls:
        log.debug("nothing to download: download_files exiting")
        return
    # cap out max tasks at number of urls
    concurrent_download_tasks = min(concurrent_download_tasks, len(urls))
    # set up queues to hold download task args
    download_queue, download_dlq, succeeded_paths = asyncio.Queue(maxsize=2000), asyncio.Queue(), []
    log.debug(f"create {concurrent_download_tasks} concurrent download tasks "
              f"for {len(urls)} items")
    # initialise consumers which block at queue.get() until producer fills queue
    dl_queue_consumers = [
        asyncio.create_task(
            download_worker(download_queue,
                            download_dlq,
                            session,
                            succeeded_paths,
                            timeout=timeout)
        )
        for _ in range(concurrent_download_tasks)
    ]
    log.debug(f"create {concurrent_download_tasks} dlq consumers")
    # initialise consumers for dlq
    dl_dlq_consumers = [
        asyncio.create_task(
            handle_failed_worker(download_dlq)
        )
        for _ in range(concurrent_download_tasks)
    ]
    # produce inputs for download tasks: urls and file paths
    producer = await produce(queue=download_queue, items=zip(urls, save_paths))
    # wait for all downloads in queue to get task_done
    log.debug("waiting for main download queue to complete...")
    await download_queue.join()
    # and for dlq
    log.debug("waiting for DLQ")
    await download_dlq.join()
    # cancel all coroutines once queue is empty
    log.debug("tasks done - terminating download worker coroutines")
    for consumer_future in dl_queue_consumers + dl_dlq_consumers:
        consumer_future.cancel()
    log.info(f"download_files completed successfully for "
             f"{len(succeeded_paths)}/{len(urls)} items")
    # return the paths to the downloaded files
    return succeeded_paths

async def extract_and_translate(archive_paths,
                                target_scales,
                                overwrite=False,
                                delete_originals=True,
                                tols=itertools.repeat(0.01),
                                concurrent_extract_jobs=8):
    all_ap = archive_paths
    # cap out max tasks at number of files
    archive_paths = [p for p in archive_paths if p.exists()]
    log.info(f"Extracting {len(archive_paths)}/{len(all_ap)} requested archives")
    concurrent_extract_jobs = min(concurrent_extract_jobs, len(archive_paths))
    # set up queues to hold extract task args
    extract_queue = asyncio.Queue()
    succeeded = []
    log.debug(f"create {concurrent_extract_jobs} concurrent extract tasks for "
              f"{len(archive_paths)} items")
    # initialise consumers which block at queue.get() until producer fills queue
    queue_consumers = [
        asyncio.create_task(
            extract_translate_worker(extract_queue,
                                     succeeded,
                                     overwrite=overwrite,
                                     delete_originals=delete_originals)
        )
        for _ in range(concurrent_extract_jobs)
    ]
    # produce inputs for download tasks: urls and file paths
    producer = await produce(queue=extract_queue,
                             items=zip(archive_paths, target_scales, tols))
    # wait for all files to be extracted
    await extract_queue.join()
    # cancel all coroutines once queue is empty
    for consumer_future in queue_consumers:
        consumer_future.cancel()
    log.info(f"Unzip and extract completed successfully for "
             f"{len(succeeded)/len(archive_paths)} items")
    # return the paths to tthe extracted rasters
    return succeeded


async def download_extract_translate(urls,
                                     filenames,
                                     save_dir,
                                     target_scales,
                                     overwrite=False,
                                     concurrent_download_tasks=10,
                                     concurrent_extract_jobs=8,
                                     delete_originals=True,
                                     timeout=7200,
                                     tols=itertools.repeat(0.01),
                                     chunk_size=512_000):
    loop = asyncio.get_event_loop()#create_loop()#asyncio.get_event_loop()
    save_paths = [save_dir / Path(f) for f in filenames]
    # download then extract those which aren't already set
    async with aiohttp.ClientSession(loop=loop) as sesh:
        # download all the archive files
        await download_files(urls=urls,
                             save_paths=save_paths,
                             session=sesh,
                             overwrite=overwrite,
                             concurrent_download_tasks=concurrent_download_tasks,
                             timeout=timeout,
                             chunk_size=chunk_size)
    # extract the archives and convert the rasters
    extracted_raster_paths = await extract_and_translate(
        archive_paths=save_paths,
        target_scales=target_scales,
        tols=tols,
        overwrite=overwrite,
        delete_originals=delete_originals,
        concurrent_extract_jobs=concurrent_extract_jobs
    )
    return extracted_raster_paths

def download_extract_translate_sync(notebook=True, *args, **kwargs):
    if notebook:
        loop = asyncio.get_event_loop()
        loop.run_until_complete(download_extract_translate(*args, **kwargs))
    else:
        return asyncio.run(download_extract_translate(*args, **kwargs))

def create_loop():
    asyncio.set_event_loop(None)
    loop = asyncio.new_event_loop()
    asyncio.get_child_watcher().attach_loop(loop)
    return loop


async def open_subprocess(cmd_text,
                          poll_period=1.,
                          log_errors=True,
                          error_log='/home/root/cmd_errs.log',
                          cmd_log='/home/root/cmd.log'):
    """ workaround async subprocess (asyncio subbprocess_shell hangs on communicate)"""
    # output = await asyncio.create_subprocess_shell(
    #    cmd,
    #    stdout=asyncio.subprocess.PIPE,
    #    stderr=asyncio.subprocess.PIPE)
    log.debug("open subprocess:")
    log.debug(cmd_text)
    output = subprocess.Popen(cmd_text,
                              shell=True,
                              stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT)
    while output is not None:
        retcode = output.poll()
        if retcode is not None:
            stdout, stderr = output.communicate()
            if retcode != 0:
                if error_log:
                    with open(error_log, 'a') as fp:
                        lines = [
                            'ERROR:\n',
                            f'cmd: {cmd_text}\n',
                            f'retcode: {retcode}\n',
                            f'datetime: {str(datetime.datetime.now())}\n'
                        ]
                        if stdout:
                            lines.append(f'stdout: {stdout.decode()}\n')
                        if stderr:
                            lines.append(f'stderr: {stderr.decode()}\n')
                        fp.writelines(lines)
                raise OSError(f"return code {retcode} : {stderr}")
            else:
                if cmd_log:
                    with open(cmd_log, 'a') as fp:
                        lines = [
                            'CMD:',
                            f'cmd: {cmd_text}\n',
                            f'retcode: {retcode}\n',
                            f'datetime: {str(datetime.datetime.now())}\n'
                        ]
                        if stdout:
                            lines.append(f'stdout: {stdout.decode()}\n')
                        if stderr:
                            lines.append(f'stderr: {stderr.decode()}\n')
                        fp.writelines(lines)
            if stdout:
                log.debug(f'[stdout]\n{stdout.decode()}')
            if stderr:
                log.debug(f'[stderr]\n{stderr.decode()}')
            # done
            return stdout.decode()
            break
        else:
            # still running? wait then check again
            await asyncio.sleep(poll_period)


def select_rasters(paths, allow_multiple_rasters=True, raise_if_none_found=False):
    """ pick out a unique tiff (high prio) or jp2 (low prio) file from
        a list of files. throw an exception if no rasters.
        used for unzipping archives.
    """
    tiffs = [f for f in paths if re.match('.*\.tiff?$', f.parts[-1])]
    jp2s = [f for f in paths if re.match('.*\.jp2$', f.parts[-1])]
    if (len(tiffs) > 1 or len(jp2s) > 1) and not allow_multiple_rasters:
        raise Exception("Ambiguous zip archive: more than one raster detected?"
                        f"tifs: {tiffs} jp2s: {jp2s}")
    elif tiffs:
        return tiffs
    elif jp2s:
        return jp2s
    else:
        if not raise_if_none_found:
            return tiffs
        raise Exception(f"No tif/jp2 rasters found in file list!"
                        f"Archive files are: {paths}")


async def extract_zipfile(filepath,
                          overwrite=False,
                          delete_archive=False,
                          return_raster_list_only=False,
                          raise_if_no_rasters_found=False):
    """ await extract_zipfile('./data/OKZRGB79_90VL_K01.zip', overwrite=True)
    """
    assert str(filepath).endswith('.zip'), f"{filepath} isn't a zip archive?"
    if not Path(filepath).exists():
        raise FileNotFoundError(f"attempt to unzip non-existent file: {filepath}")
    parent = Path(filepath).parent
    with ZipFile(filepath, 'r') as zip_obj:
        archive_files = zip_obj.namelist()
    paths = [parent / Path(f) for f in archive_files]
    # look for a raster file in zip archive
    rasters = select_rasters(paths, raise_if_none_found=raise_if_no_rasters_found)
    if return_raster_list_only:
        return rasters
    if all(r.exists() for r in rasters) and not overwrite:
        log.warning(f"raster files {rasters} already present on disk, "
                    "skipping unzip...")
    else:
        ov = '-o' if overwrite else '-n'
        unzip_cmd = f'unzip {ov} {filepath} -d {parent}'
        # open an OS subprocess to unzip
        await open_subprocess(unzip_cmd)
        assert all(r.exists() for r in rasters), (
            f"'{unzip_cmd}' succeeded but expected files {rasters} not found!"
        )
        log.debug(f"Extracted files: {archive_files} from archive: {filepath}")
    if delete_archive:
        log.debug(f"Deleting zip archive: {filepath}")
        await open_subprocess(f'rm {filepath}')
        # delete non-raster files?
        # clean_paths = [p for p in paths if p != raster]
    # return a list of the paths to the extracted files
    return rasters


async def gdal_translate(input_raster_path,
                         target_extension='.tif',
                         target_scale=1.0,
                         overwrite=False,
                         delete_originals=True,
                         tol=0.01):
    """ run gdal_translate in a shell to convert a raster to a selected format,
        optionally rescaling it in the process if scale_factor within tol of 1

        target_scale is a scale factor for the spatial resolution

    """
    input_raster_name = Path(input_raster_path).parts[-1]
    # remove ext
    base_name = '.'.join(input_raster_name.split('.')[:-1])
    if (1-tol) < target_scale < (1+tol):
        target_scale = 1.0
    size_pct = int(round(target_scale*100))
    rs_string = f'_resampled_{size_pct}pct' if target_scale != 1. else ''
    output_raster_name = base_name + rs_string + target_extension
    if output_raster_name == input_raster_name:
        warnings.warn(f"redundant gdal_translate? in={input_raster_name} "
                      f"out={output_raster_name} skipping...")
        return output_raster_name
    out_file = Path(input_raster_path).parent / output_raster_name
    if out_file.exists() and not overwrite:
        log.debug(f"gdal_translate output file {out_file} already exists, "
                  " skipping...")
        return out_file
    log.debug(f"translating {input_raster_name} -> {out_file.parts[-1]}")
    translate_cmd = ('gdal_translate -of GTiff -co COMPRESS=LZW -co TILED=YES '
                     f'-outsize {size_pct}% {size_pct}% -r bilinear '
                     f'{input_raster_path} {out_file}')
    await open_subprocess(translate_cmd)
    # if an exception is raised by the above command, this line doesnt run
    # and the original isnt deleted
    if delete_originals:
        await open_subprocess(f'rm {input_raster_path}')
    return out_file


async def extract_rasters_and_translate(archive_path,
                                        target_scale=1.,
                                        overwrite=False,
                                        delete_originals=True,
                                        tol=0.01):
    """ 
    Extracts a zip archive, identifies raster files therein and applies
    GDAL translate to resample these to target_scale m spatial resolution

    Parameters
    ----------
    archive_path :
        Path to zip file
    target_scale :
        Target spatial resolution to resample to
    overwrite : 
        Boolean flag to toggle overwriting existing rasters with same name
    delete_originals : 
        Boolean flag to get rid of un-resampled rasters from zip to save space
    tol :
        Fractional margin of error on spatial resolution with respect to the 
        target spatial resolution beyond which the raster will be resampled

    Examples
    --------

    >>> resampled_rasters = await extract_raster_and_translate(
        '/home/root/data/raw/vlaanderen/OKZRGB79_90VL_K01.zip',
            overwrite=True
        )
    """
    raster_paths = await extract_zipfile(archive_path, overwrite=overwrite)
    output_rasters = []
    for raster_path in raster_paths:
        output_rasters.append(
            await gdal_translate(raster_path,
                                 target_scale=target_scale,
                                 overwrite=overwrite,
                                 delete_originals=delete_originals,
                                 tol=tol)
        )
    return output_rasters
