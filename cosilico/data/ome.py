from tempfile import TemporaryDirectory
from typing import Union, Iterable, Tuple
import os

from numpy.typing import DTypeLike
from ome_types import OME, model, to_xml
from typing_extensions import Annotated, Doc
import dask.array as da
import numpy as np
import tifffile
import zarr

from cosilico.data.transforms import Scale, ScalingMethod
from cosilico.wrappers.bioformats import to_ngff
from cosilico.typing import ArrayLike


def create_ome_model(data, channels, resolution, resolution_unit) -> OME:
    if len(data.shape) == 3:
        c, h, w = data.shape
        t, z = 1, 1
    else:
        t, z, c, h, w = data.shape

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
                size_c=c,
                size_t=t,
                size_z=z,
                size_x=w,
                size_y=h,
                type=dtype_name,
                big_endian=False,
                channels=[model.Channel(id=f'Channel:{i}', name=x, samples_per_pixel=1)
                          for i, x in enumerate(channels)],
                physical_size_x=resolution,
                physical_size_y=resolution,
                physical_size_x_unit=resolution_unit,
                physical_size_y_unit=resolution_unit,
            )
        )
    )

    im = o.images[0]
    for c_idx in range(c):
        for t_idx in range(t):
            for z_idx in range(z):
                im.pixels.planes.append(model.Plane(the_c=c_idx, the_t=t_idx, the_z=z_idx))
    im.pixels.tiff_data_blocks.append(model.TiffData(plane_count=len(im.pixels.planes)))

    return o


def ome_from_data(
    data: Annotated[ArrayLike, Doc(
        'Source array to generate NGFF from. Can be dask, numpy, or zarr array. Must have shape (num_channels, height, width) or (num_timepoints, depth, num_channels, height, width).'
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
    )] = None,
    target_dtype: Annotated[Union[DTypeLike | None], Doc(
        'Target data type. `scaling_method`, `scaling_axis`, and `scaling_range` will determine how the data is scaled to be converted to the target data type. If target dtype is None or matches the input data type then no scaling will occur. By default target data type is uint8. Note that images saved for use with Cosilico viewer must be uint8.'
    )] = np.uint8,
    scaling_method: Annotated[Union[ScalingMethod | None], Doc(
        'Method used to scale data to fit the range of the target data type. Default is `min_max`.'
    )] = ScalingMethod.min_max,
    scaling_range: Annotated[Union[Tuple[int] | None], Doc(
        'If present, will be used in place of `scaling method`. Specifies a target range data will be scaled to.'
    )] = None,
    scaling_axis: Annotated[Union[int | Tuple[int], None], Doc(
        'Axes along which to scale data. If None, scaling will be done individually for each channel.'
    )] = None
    # will add pyramids back in later, slightly complicated to do right now.
    # pyramids: Annotated[Union[int | None], Doc(
    #     'Number of pyramids to write in pyramidal OME-TIF. By default none are written.'
    # )] = None,
    ):
    assert len(data.shape) == 3 or len(data.shape) == 5, f'data must have 3 axes of shape (num_channels, height, width) or 5 axes of shape (num_timepoints, depth, num_channels, height, width). Got {len(data.shape)} axes.'
    assert channels is None or len(channels) == data.shape[-3], f'Number of channels (got {len(channels)}) must be equal to length of channel axis in data (got {data.shape[-3]}). '
    if isinstance(data, zarr.Array):
        assert len(data.shape) == 5, f'If data is a zarr.Array, it must have 5 axes of shape (num_timepoints, depth, num_channels, height, width). Got {len(data.shape)} axes.'

    # scale data if we need to
    if target_dtype is not None and np.dtype(target_dtype) != np.dtype(data.dtype):
        if scaling_axis is None:
            scaling_axis = (1, 2) if len(data.shape) == 3 else (0, 1, 3, 4)
        scaler = Scale(
            method=scaling_method,
            dtype=target_dtype,
            target_range=scaling_range,
            axis=scaling_axis
        )
        data = scaler(data)

    # generate ome model
    if ome_model is None:
        assert all(
            [x is not None for x in [channels, resolution, resolution_unit]]
        ), 'If not OME model is provided, then channels, resolution, and resolution_unit must be provided.'
        ome_model = create_ome_model(data, channels, resolution, resolution_unit)

    with tifffile.TiffWriter(output_path, ome=True, bigtiff=True) as out_tif:
        if len(data.shape) == 3:
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
    )] = None,
    target_dtype: Annotated[Union[DTypeLike | None], Doc(
        'Target data type. `scaling_method`, `scaling_axis`, and `scaling_range` will determine how the data is scaled to be converted to the target data type. If target dtype is None or matches the input data type then no scaling will occur. By default target data type is uint8. Note that images saved for use with Cosilico viewer must be uint8.'
    )] = np.uint8,
    scaling_method: Annotated[Union[ScalingMethod | None], Doc(
        'Method used to scale data to fit the range of the target data type. Default is `min_max`.'
    )] = ScalingMethod.min_max,
    scaling_range: Annotated[Union[Tuple[int] | None], Doc(
        'If present, will be used in place of `scaling method`. Specifies a target range data will be scaled to.'
    )] = None,
    scaling_axis: Annotated[Union[int | Tuple[int], None], Doc(
        'Axes along which to scale data. If None, scaling will be done individually for each channel.'
    )] = None,
    tile_width: Annotated[Union[int | None], Doc(
        'Width of tiles/chunks (in pixels) that will be written in Zarr arrays. Bioformats2raw defaults are used if no size is provided.'
    )] = None,
    tile_height: Annotated[Union[int | None], Doc(
        'Height of tiles/chunks (in pixels) that will be written in Zarr arrays. Bioformats2raw defaults are used if no size is provided.'
    )] = None
    ):
    with TemporaryDirectory() as tempdir:
        ome_path = os.path.join(tempdir, 'temp.ome.tif')
        ome_from_data(
            data, ome_path,
            channels=channels,
            resolution=resolution,
            resolution_unit=resolution_unit,
            ome_model=ome_model,
            target_dtype=target_dtype,
            scaling_method=scaling_method,
            scaling_range=scaling_range,
            scaling_axis=scaling_axis
        )

        to_ngff(ome_path, output_path, tile_height=tile_height, tile_width=tile_width)