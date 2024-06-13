from typing import Union

from pint import UnitRegistry
from numpy.typing import NDArray
from scipy.sparse import spmatrix
import numpy as np
import zarr

from cosilico.data.types import (
    PixelDataType,
    ScalingMethod,
    MultiplexImage,
    PropertyGroup
)

ureg = UnitRegistry()



# Data type conversion

def convert_dtype(
    obj: Union[MultiplexImage | PropertyGroup]
    ):
    """
    Convienice method for converting data type of objects with .data and .data_type attribs
    """
    if obj.data_type is None:
        if isinstance(obj.data, NDArray):
            dtype = obj.data.dtype
        elif isinstance(obj.data, zarr.Array):
            dtype = obj.dtype.type
        
        obj.data_type = PixelDataType(dtype)
    elif all(
        obj.data_type is not None,
        obj.data_type.value != obj.data.dtype,
        ):
        obj.data = scale_data(obj.data, obj.data_type, obj.scaling_method)

    return obj

def scale_data(
        data: Union[NDArray | spmatrix | zarr.Array],
        data_type: PixelDataType,
        scaling_method: ScalingMethod
    ):
    if scaling_method.value == 'min_max':
        min_value, max_value = data.min(), data.max()
    elif scaling_method.value == 'mindtype_maxdtype':
        min_value, max_value = np.iinfo(data_type.value).min, np.iinfo(data_type.value).max
    elif scaling_method.value == 'zero_max':
        min_value, max_value = 0, data.max()
    elif scaling_method.value == 'zero_maxdtype':
        min_value, max_value = 0, np.iinfo(data_type.value).max
    elif scaling_method.value == 'min_maxdtype':
        min_value, max_value = data.min(), np.iinfo(data_type.value).max

    if scaling_method.value != 'no_scale':
        data = data - min_value
        data = data / max_value

    data = data.astype(data_type.value)

    return data


# Unit conversion

def to_microns_per_pixel(resolution, resolution_unit):
    converted = (resolution * ureg(resolution_unit)).to('micron')
    return converted.magnitude