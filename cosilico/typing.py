from typing import Union, Tuple, Iterable

from numpy.typing import NDArray, DTypeLike
from scipy.sparse import spmatrix
import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr

ArrayLike = Union[np.ndarray | spmatrix | da.core.Array | zarr.Array]
DataFrameLike = Union[pd.DataFrame, dask.dataframe.DataFrame]

# currently only uint8 is supported by frontend
NGFF_DTYPES = ['uint8']
