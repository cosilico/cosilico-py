from typing import Union

from einops import rearrange
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from zarr.storage import StoreLike
import numpy as np
import zarr

def initialize_array(
    data: spmatrix,
    array: Union[zarr.Array | None] = None,
    store: Union[StoreLike | None] = None, 
    ):
    if array is None:
        if store is None:
            array = zarr.empty_like(data)
        else:
            array = zarr.open(store, mode='a', shape=data.shape, dtype=data.dtype)
    return array

def sparse_to_zarr(
    data: spmatrix,
    array: Union[zarr.Array | None] = None,
    store: Union[StoreLike | None] = None,
    ):
    array = initialize_array(data, array, store)

    dim_sizes = [dim1 // c1 + 1 for dim1, c1 in zip(array.shape, array.chunks)]
    sizes = np.stack(np.meshgrid(*[np.arange(i) for i in dim_sizes]))
    idxs = rearrange(sizes, 'n ... -> (...) n')
    for indices in idxs:
        slices = [slice(i * cs, (i + 1) * cs) for i, cs in zip(indices, array.chunks)]
        array[*slices] = data[*slices].toarray()
    
    return array

def numpy_to_zarr(
    data: NDArray,
    array: Union[zarr.Array | None] = None,
    store: Union[StoreLike | None] = None,
    ):
    array = initialize_array(data, array, store)
    array[:] = data
    return array


def to_zarr(
    data: Union[spmatrix | NDArray],
    array: Union[zarr.Array | None] = None,
    store: Union[StoreLike | None] = None,
    ):
    if isinstance(data, spmatrix):
        return sparse_to_zarr(data, array, store)
    
    return numpy_to_zarr(data, array, store)