from pathlib import Path
from typing import Union, Iterable, Callable
import os

from typing_extensions import Annotated, Doc


from cosilico.wrappers.bioformats import to_ngff, to_ome

def ngff_from_path(
        source_path: Annotated[os.PathLike, Doc(
            'Path to source file.'
        )],
        output_path: Annotated[os.PathLike, Doc(
            'Path to directory to use as zarr store.'
        )],
        data_transforms: Annotated[Iterable[Callable], Doc(
            'Transforms to use on pixel data.'
        )],
        metadata_transforms: Annotated[Iterable[Callable], Doc(
            'Transforms to use on ngff metadata.'
        )],
    ):
    # create ngff

    # modify ngff

    # return zarr


def ngff_from_data(
        data,
        channels,
        resolution,
        resolution_unit,
        output_path,
        data_transforms,
        metadata_transforms,
    ):
    # create ome tif

    # create ngff

    # modify ngff

    # return zarr