from __future__ import annotations
from typing import Union, TYPE_CHECKING, Tuple, Iterable

from numpy.typing import NDArray
from scipy.sparse import spmatrix
import dask.array as da
import numpy as np
import zarr

from cosilico.data.types import PixelDataType
if TYPE_CHECKING:
    from cosilico.data.types import (
        ScalingMethod,
        MultiplexImage,
        PropertyGroup,
        Property
    )

# Data type conversion

def convert_dtype(
    obj: Union['MultiplexImage' | 'PropertyGroup' | 'Property'],
    scale: bool = False,
    axis: Union[int | None] = None
    ):
    """
    Convienice method for converting data type of objects with .data and .data_type attribs
    """
    if obj.data_type is None:
        if isinstance(obj.data, NDArray):
            dtype = obj.data.dtype
        else:
            dtype = obj.data.dtype.type
        
        obj.data_type = PixelDataType(dtype)
    elif all(
        obj.data_type is not None,
        obj.data_type.value != obj.data.dtype,
        ) or scale:
        obj.data = scale_data(obj.data, obj.data_type, obj.scaling_method, axis=axis)

    return obj

def scale_data(
        data: Union[NDArray | spmatrix | zarr.Array | da.core.Array],
        data_type: 'PixelDataType',
        scaling_method: 'ScalingMethod',
        axis: Union[int | Tuple[int], None] = None
    ):
    if scaling_method.value == 'min_max':
        min_value, max_value = data[:].min(axis, keepdims=True), data[:].max(axis, keepdims=True)
    elif scaling_method.value == 'mindtype_maxdtype':
        min_value, max_value = np.iinfo(data_type.value).min, np.iinfo(data_type.value).max
    elif scaling_method.value == 'zero_max':
        min_value, max_value = 0, data[:].max(axis, keepdims=True)
    elif scaling_method.value == 'zero_maxdtype':
        min_value, max_value = 0, np.iinfo(data_type.value).max
    elif scaling_method.value == 'min_maxdtype':
        min_value, max_value = data[:].min(axis, keepdims=True), np.iinfo(data_type.value).max

    if scaling_method.value != 'no_scale':
        data = data.astype(np.dtype('float32'))
        data -= min_value
        data /= max_value
        data *= np.iinfo(data_type.value).max

    data = data.astype(data_type.value)

    return data