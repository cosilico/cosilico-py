from pathlib import Path
from typing import Union, Tuple
import os

from numpy.typing import NDArray, DTypeLike
from scipy.sparse import spmatrix
from typing_extensions import Annotated, Self, Doc
import dask.array as da
import numpy as np
import zarr
import zarr.convenience

from cosilico.data.scaling import ScalingMethod, scale_data
from cosilico.typing import ArrayLike
from cosilico.data.zarr import to_zarr


class Scale(object):
    """
    Scale data
    """
    def __init__(
        self,
        method: Union[str | ScalingMethod] = ScalingMethod.min_max,
        dtype: Union[DTypeLike | None] = None,
        target_range: Union[Tuple[int] | None] = None,
        axis: Union[int | Tuple[int], None] = None,
        ):
        self.method = ScalingMethod(method)
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.target_range = target_range
        self.axis = axis

    def __call__(
        self,
        data: ArrayLike,
        ) -> ArrayLike:

        return scale_data(
            data,
            dtype=self.dtype,
            scaling_method=self.method,
            target_range=self.target_range,
            axis=self.axis
        )