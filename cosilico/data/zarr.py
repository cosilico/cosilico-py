from pathlib import Path
from typing import Union
from tempfile import TemporaryDirectory, gettempdir
from uuid import uuid4
import os
import shutil


from einops import rearrange
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from zarr.storage import StoreLike
import numpy as np
import zarr

TMP_PREFIX = 'csctmpzarr_'

def clear_tmp():
    tmpdir = gettempdir()
    to_delete = [os.path.join(tmpdir, name) for name in os.listdir(tmpdir)
                 if name[:len(TMP_PREFIX)] == TMP_PREFIX]
    for path in to_delete:
        shutil.rmtree(path)

def delete_if_tmp(
    x: zarr.Array
    ):
    if any([
        isinstance(x.store, zarr.DirectoryStore),
        isinstance(x.store, zarr.ZipStore)
        ]):
        path = Path(x.store.path)
        if path.parent.name[:len(TMP_PREFIX)] == TMP_PREFIX:
            shutil.rmtree(path.parent)

def initialize_array(
    data: Union[spmatrix, NDArray],
    store: Union[StoreLike | None] = None, 
    ) -> zarr.Array:
    if store is None:
        directory = TemporaryDirectory(prefix=TMP_PREFIX, delete=False)
        store = Path(directory.name) / (str(uuid4()) + '.zarr')

    array = zarr.open(store, mode='a', shape=data.shape, dtype=data.dtype)
    return array

def sparse_to_zarr(
    data: spmatrix,
    array: Union[zarr.Array | None] = None,
    store: Union[StoreLike | None] = None,
    ):
    if array is None:
        array = initialize_array(data, store)

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
    if array is None:
        array = initialize_array(data, store)
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