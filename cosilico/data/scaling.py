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
    mindtype_maxdtype = 'mindtype_maxdtype'
    no_scale = 'no_scale'

def scale_data(
        data: ArrayLike,
        target_dtype: Union[DTypeLike | None] = None,
        scaling_method: Union[ScalingMethod | None] = 'min_max',
        axis: Union[int | Tuple[int], None] = None,
        target_range: Union[Tuple[int] | None] = None
    ) -> ArrayLike:
    if target_range is None and scaling_method is None:
        scaling_method = 'min_max'

    if scaling_method is not None:
        values = {x.value for x in ScalingMethod}
        assert scaling_method in values, f'scaling_method must be one of {values}'
    
    if target_dtype is None:
        target_dtype = data.dtype
    target_dtype = np.dtype(target_dtype)

    current_dtype = np.dtype(data.dtype)


    if target_range is None:
        if scaling_method.value == 'min_max':
            min_value, max_value = data[:].min(axis, keepdims=True), data[:].max(axis, keepdims=True)
        elif scaling_method.value == 'mindtype_maxdtype':
            min_value, max_value = np.iinfo(current_dtype).min, np.iinfo(current_dtype).max
        elif scaling_method.value == 'zero_max':
            min_value, max_value = 0, data[:].max(axis, keepdims=True)
        elif scaling_method.value == 'zero_maxdtype':
            min_value, max_value = 0, np.iinfo(current_dtype).max
        elif scaling_method.value == 'min_maxdtype':
            min_value, max_value = data[:].min(axis, keepdims=True), np.iinfo(current_dtype).max
    else:
        min_value, max_value = target_range

    if scaling_method.value != 'no_scale':
        dtype_min, dtype_max = np.iinfo(target_dtype).min, np.iinfo(target_dtype).max

        data = data.astype(np.float32)
        data = (data - min_value) / (max_value - min_value)
        data = data * (dtype_max - dtype_min) + dtype_min

    data = data.astype(target_dtype)

    return data