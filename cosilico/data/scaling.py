from enum import Enum
from typing import Union, Tuple, Iterable

from numpy.typing import NDArray, DTypeLike
from scipy.sparse import spmatrix
import dask.array as da
import numpy as np
import zarr

from cosilico.typing import ArrayLike

class ScalingMethod(str, Enum):
    min_max = 'min_max'
    zero_max = 'zero_max'
    min_maxdtype = 'min_maxdtype'
    zero_maxdtype = 'zero_maxdtype'
    mindtype_max_dtype = 'mindtype_maxdtype'
    no_scale = 'no_scale'

def scale_data(
        data: ArrayLike,
        dtype: Union[DTypeLike | None] = None,
        scaling_method: Union[ScalingMethod | None] = ScalingMethod.min_max,
        axis: Union[int | Tuple[int], None] = None,
        target_range: Union[Tuple[int] | None] = None
    ) -> ArrayLike:
    if target_range is None and scaling_method is None:
        scaling_method = ScalingMethod.min_max
    
    if dtype is None:
        dtype = data.dtype

    dtype = np.dtype(dtype)

    if target_range is None:
        if scaling_method.value == 'min_max':
            min_value, max_value = data[:].min(axis, keepdims=True), data[:].max(axis, keepdims=True)
        elif scaling_method.value == 'mindtype_maxdtype':
            min_value, max_value = np.iinfo(dtype).min, np.iinfo(dtype).max
        elif scaling_method.value == 'zero_max':
            min_value, max_value = 0, data[:].max(axis, keepdims=True)
        elif scaling_method.value == 'zero_maxdtype':
            min_value, max_value = 0, np.iinfo(dtype).max
        elif scaling_method.value == 'min_maxdtype':
            min_value, max_value = data[:].min(axis, keepdims=True), np.iinfo(dtype).max
    else:
        min_value, max_value = target_range

    if scaling_method.value != 'no_scale':
        dtype_min, dtype_max = np.iinfo(dtype).min, np.iinfo(dtype).max
        data = data.astype(np.float32)
        data = (data - min_value) / (max_value - min_value)
        data = data * (dtype_max - dtype_min) + dtype_min

    data = data.astype(dtype)

    return data