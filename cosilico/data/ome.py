from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Union, Iterable, Callable
import os

from einops import rearrange
from ome_types import OME, model, to_xml
from typing_extensions import Annotated, Doc
import dask.array as da
import numpy as np
import tifffile
import zarr

from cosilico.wrappers.bioformats import to_ngff
from cosilico.data.transforms import Scale
from cosilico.typing import ArrayLike


def create_ome_model(data, channels, resolution, resolution_unit) -> OME:
    # create ome tif
    dtype_name = str(np.dtype(data.dtype))
    if 'int' in dtype_name:
        assert dtype_name in ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32']
    elif 'float' in dtype_name:
        dtype_name = 'float'
    elif dtype_name == 'bool':
        dtype_name = 'bit'
    else:
        raise ValueError(f'dtype {data.dtype} was unable to be saved as ome')
    
    o = model.OME()
    o.images.append(
        model.Image(
            id='Image:0',
            pixels=model.Pixels(
                dimension_order='XYCZT',
                size_c=len(channels),
                size_t=1,
                size_z=1,
                size_x=data.shape[2],
                size_y=data.shape[1],
                type=dtype_name,
                big_endian=False,
                channels=[model.Channel(id=f'Channel:{i}', name=c) for i, c in enumerate(channels)],
                physical_size_x=resolution,
                physical_size_y=resolution,
                physical_size_x_unit=resolution_unit,
                physical_size_y_unit=resolution_unit
            )
        )
    )

    im = o.images[0]
    for i in range(len(im.pixels.channels)):
        im.pixels.planes.append(model.Plane(the_c=i, the_t=0, the_z=0))
    im.pixels.tiff_data_blocks.append(model.TiffData(plane_count=len(im.pixels.planes)))

    return o


def ome_from_data(
    data: Annotated[ArrayLike, Doc(
        'Source array to generate NGFF from. Can be dask, numpy, or zarr array. Must have shape (num_channels, height, width).'
    )],
    output_path: Annotated[os.PathLike, Doc(
        'Filepath to write OME-TIF.'
    )],
    channels: Annotated[Union[Iterable | None], Doc(
        'Channel names. Must be unique and correspond to first axis of `data`.'
    )] = None,
    resolution: Annotated[Union[float | int | None], Doc(
        'Resolution of pixels given in units per pixel.'
    )] = None,
    resolution_unit: Annotated[Union[str | None], Doc(
        'Unit used for resolution.'
    )] = None,
    ome_model: Annotated[Union[OME | None], Doc(
        'OME model to use when saving OME-TIF. By default a simple OME model will be created based on other function arguments. If more specific or complex OME metadata is desired, an `ome_types.OME` model can be provided and will be saved as the metadata for the OME-TIF.'
    )] = None
    # will add pyramids back in later, slightly complicated to do right now.
    # pyramids: Annotated[Union[int | None], Doc(
    #     'Number of pyramids to write in pyramidal OME-TIF. By default none are written.'
    # )] = None,
    ):
    assert len(data.shape) == 3, f'data must have 3 axes of shape (num_channels, height, width). Got {len(data.shape)} axes.'
    assert channels is None or len(channels) == data.shape[0], f'Number of channels (got {len(channels)}) must be equal to length of first axis in data (got {data.shape[0]}). '

    if ome_model is None:
        assert all(
            [x is not None for x in [channels, resolution, resolution_unit]]
        ), 'If not OME model is provided, then channels, resolution, and resolution_unit must be provided.'
        ome_model = create_ome_model(data, channels, resolution, resolution_unit)

    with tifffile.TiffWriter(output_path, ome=True, bigtiff=True) as out_tif:
        data = da.expand_dims(data, (0, 1))
        opts = {
            'compression': 'LZW',
        }
        out_tif.write(
            data,
            **opts
        )
        xml_str = to_xml(ome_model)
        out_tif.overwrite_description(xml_str.encode())



def ngff_from_data(
    data: Annotated[ArrayLike, Doc(
        'Source array to generate NGFF from. Can be dask, numpy, or zarr array. Must have shape (num_channels, height, width).'
    )],
    output_path: Annotated[os.PathLike, Doc(
        'Directory to write OME-NGFF zarr.'
    )],
    channels: Annotated[Union[Iterable | None], Doc(
        'Channel names. Must be unique and correspond to first axis of `data`.'
    )] = None,
    resolution: Annotated[Union[float | int | None], Doc(
        'Resolution of pixels given in units per pixel.'
    )] = None,
    resolution_unit: Annotated[Union[str | None], Doc(
        'Unit used for resolution.'
    )] = None,
    ome_model: Annotated[Union[OME | None], Doc(
        'OME model to use when saving OME-TIF. By default a simple OME model will be created based on other function arguments. If more specific or complex OME metadata is desired, an `ome_types.OME` model can be provided and will be saved as the metadata for the OME-TIF.'
    )] = None
    ):
    with TemporaryDirectory() as tempdir:
        ome_path = os.path.join(tempdir, 'temp.ome.tif')
        ome_from_data(
            data, ome_path,
            channels=channels,
            resolution=resolution,
            resolution_unit=resolution_unit,
            ome_model=ome_model
        )

        to_ngff(ome_path, output_path)

