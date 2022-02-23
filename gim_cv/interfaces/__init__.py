import importlib

from pathlib import Path

import logging

log = logging.getLogger(__name__)

ALIASES = {'tiff' : 'tif'}

def get_interface(file, interface_type, aliases=ALIASES):
    """ Dispatch to the correct file <-> array interface based on the file extension

        Returns:
            one of the reader classes implemented for the file format of file
    """
    path = Path(file)
    #assert path.exists(), f"File {file} not found."
    ext = str(path).split('.')[-1]
    if ext in aliases:
        ext = aliases[ext]
    module = importlib.import_module(f'gim_cv.interfaces.{ext}')
    return getattr(module, interface_type)


#def get_writer(file, writer_type):
#    """ Dispatch to the correct file <-> array interface based on the file extension
#
#        Returns:
#            one of the writer classes implemented for the file format of file
#    """
#    path = Path(file)
#    #assert path.exists(), f"File {file} not found."
#    ext = str(path).split('.')[-1]
#    module = importlib.import_module(f'gim_cv.interfaces.{ext}')
#    return getattr(module, writer_type)
