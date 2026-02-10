import numpy as np
import xarray as xr
from igor2 import binarywave
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display
from pathlib import Path
from .metadata import build_metadata_from_ibw
# from pynxtools.dataconverter.convert import convert
        
def ibw_to_xarray(path):
    
    bw = binarywave.load(path)

    wave = bw["wave"]
    header = wave["wave_header"]
    data = wave["wData"]

    ndim = data.ndim

    sfA = header["sfA"]
    sfB = header["sfB"]
    nDim = header["nDim"]

    coords = {}
    dims = []
    metadata = build_metadata_from_ibw(bw)

    for i in range(ndim):
        n = nDim[i]
        start = sfB[i]
        delta = sfA[i]

        coord = start + delta * np.arange(n)

        dim_name = f"dim_{i}"
        dims.append(dim_name)
        coords[dim_name] = coord

    coords['X'] = metadata['sample']['position']['X']
    coords['Y'] = metadata['sample']['position']['Y']
    coords['Z'] = metadata['sample']['position']['Z']
    coords['Phi'] = metadata['sample']['position']['Phi']
    coords['Theta'] = metadata['sample']['position']['Theta']
    coords['Omega'] = metadata['sample']['position']['Omega']
    
    
    data = xr.DataArray(
        data,
        dims=dims,
        coords=coords,
        name=header["bname"].decode("ascii"),
        attrs={
            "note": wave["note"].decode("latin1", errors="ignore")
        }
    )

    return data

# def to_nexus(
#     data: xr.DataArray,
#     faddr: str,
#     reader: str,
#     definition: str,
#     input_files: str | Sequence[str],
#     **kwds,
# ):
#     """Saves the x-array provided to a NeXus file at faddr, using the provided reader,
#     NeXus definition and configuration file.

#     Args:
#         data (xr.DataArray): The data to save, containing metadata definitions in
#             data._attrs["metadata"].
#         faddr (str): The file path to save to.
#         reader (str): The name of the NeXus reader to use.
#         definition (str): The NeXus definition to use.
#         input_files (str | Sequence[str]): The file path to the configuration file to use.
#         **kwds: Keyword arguments for ``pynxtools.dataconverter.convert``.
#     """

#     if isinstance(input_files, str):
#         input_files = tuple([input_files])
#     else:
#         input_files = tuple(input_files)

#     convert(
#         input_file=input_files,
#         objects=(data),
#         reader=reader,
#         nxdl=definition,
#         output=faddr,
#         **kwds,
#     )