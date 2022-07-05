import os
from datetime import datetime
from glob import glob

import click
import satproc.chips
import unetseg.predict
from satproc.filter import filter_by_max_prob
from satproc.postprocess.polygonize import polygonize

from maiia.postprocess import filter_by_min_area


def extract_chips(
    image_paths,
    output_path,
    aoi_path,
    size=100,
    num_channels=3,
    rescale_mode="percentiles",
    lower_cut=2,
    upper_cut=98
):
    satproc.chips.extract_chips(
        rasters=image_paths,
        output_dir=output_path,
        size=size,
        step_size=size,
        aoi=aoi_path,
        bands=range(1, num_channels + 1),
        rescale_mode=rescale_mode,
        rescale_range=(lower_cut, upper_cut),
    )


def predict(images_path, results_path, model_path, num_channels=3, unet_size=160, batch_size=16):
    config = unetseg.predict.PredictConfig(
        images_path=images_path,
        results_path=results_path,
        batch_size=batch_size,
        model_path=model_path,
        height=unet_size,
        width=unet_size,
        n_channels=num_channels,
        n_classes=1,
        class_weights=[1]
    )

    return unetseg.predict.predict(config)


def post_process(input_dir, output_path, temp_dir, threshold=0.5, min_area=500):
    if threshold < 0 or threshold > 1:
        raise ValueError("threshold must be between 0 and 1")

    filt_dir = os.path.join(temp_dir, "filt")
    filter_by_max_prob(
        os.path.realpath(input_dir),
        output_dir=os.path.realpath(filt_dir),
        threshold=threshold
    )

    poly_path = os.path.join(temp_dir, "poly.gpkg")
    polygonize(
        input_dir=filt_dir,
        output=poly_path,
        threshold=threshold,
    )

    filter_by_min_area(poly_path, output_path, min_area=min_area)


@click.command(context_settings={'show_default': True})
@click.option('--images-dir', help='Path to directory containing images')
@click.option('--model-path', help='Path to trained model (HDF5 format, .h5)')
@click.option("--output-path", help="Path to output vector file (GPKG format, .gpkg)")
@click.option('--aoi-path', help='Path to AOI vector file')
@click.option("--size", default=600, help="Size of the extracted chips")
@click.option("--threshold", "-t", default=0.5, help="Threshold for filtering (between 0 and 1)")
@click.option("--min-area", "-min", default=500, help="Minimum area of detected polygons for filtering (in meters)")
@click.option("--num-channels", default=3, help="Number of channels of input images")
@click.option("--batch-size", "-B", default=16, help="Batch size")
@click.option("--temp-dir", help="Path to temporary directory, which will contain extracted chips")
def main(images_dir, model_path, output_path, aoi_path, size, threshold, min_area, num_channels, batch_size, temp_dir):
    dt_s = datetime.today().strftime("%Y%m%d_%H%M%S")

    if not images_dir:
        images_dir = os.path.join("data", "predict", "images")

    image_paths = glob(os.path.join(images_dir, "**", '*.tif'), recursive=True)
    if not image_paths:
        raise ValueError(f"No .tif images found in {images_dir}")
    click.echo(f"Found {len(image_paths)} images on {images_dir}")

    if not model_path:
        # Try to look for model files on data/models directory
        models_dir = os.path.join("data", "models")
        model_files = glob(os.path.join(models_dir, "*.h5"))
        model_files = sorted(model_files, key=lambda t: -os.stat(t).st_mtime)
        if not model_files:
            raise click.UsageError(f"No .h5 files found in {models_dir}")
        model_path = model_files[0]
    click.echo(f"Using {model_path} model file")

    if not aoi_path:
        # Try to look for AOI files on data/train/areas directory
        areas_dir = os.path.join("data", "predict", "areas")
        areas_files = glob(os.path.join(areas_dir, "*.gpkg"))
        areas_files = sorted(areas_files, key=lambda t: -os.stat(t).st_mtime)
        if not areas_files:
            aoi_path = None
            click.echo(f"No custom AOI path was specified and no files were found in {areas_dir}. Will fully predict on all images.")
        else:
            aoi_path = areas_files[0]
    click.echo(f"Using {aoi_path} as AOI file")

    if not output_path:
        output_path = os.path.join("data", "results", f"results_{dt_s}.gpkg")

    if not temp_dir:
        temp_dir = os.path.join("data", "tmp")

    chips_path = os.path.join(temp_dir, "chips", str(size))
    click.echo(f"Extracting chips of size {size} into {chips_path}")

    extract_chips(
        image_paths,
        chips_path,
        aoi_path,
        size=size,
        num_channels=num_channels,
        rescale_mode="percentiles",
        lower_cut=2,
        upper_cut=98
    )

    result_chips_path = os.path.join(temp_dir, "result_chips", str(size))
    click.echo(f"Going to predict and generate result chips on {result_chips_path}")

    predict(
        images_path=chips_path,
        results_path=result_chips_path,
        model_path=model_path,
        num_channels=num_channels,
        batch_size=batch_size,
    )

    click.echo(f"Post-process results into {output_path} using threshold {threshold} (temp files on {temp_dir})")
    post_process(result_chips_path, output_path, threshold=threshold, min_area=min_area, temp_dir=temp_dir)

    click.echo(f"Prediction finished! Result saved to {output_path}")



if __name__ == "__main__":
    main()
