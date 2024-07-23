from typing import Union, Tuple, Iterable

from numpy.typing import NDArray, DTypeLike
from scipy.sparse import spmatrix
import dask.array as da
import numpy as np
import zarr

ArrayLike = Union[NDArray | spmatrix | da.core.Array | zarr.Array]

# currently only uint8 is supported by frontend
NGFF_DTYPES = ['uint8']