from pathlib import Path
from tempfile import mkdtemp
from typing import Union, Iterable
import os

import dask.dataframe
from numpy.typing import NDArray
import numpy as np
import tifffile
from ome_types import from_xml
from rich import print
from spatialdata_io import xenium
from spatialdata.transformations import get_transformation
from typing_extensions import Annotated, Doc
import anndata
import dask
import dask.array as da
import numpy as np
import zarr

from cosilico.data.ome import ngff_from_data
import cosilico.data.types as cdt

TMP_PREFIX = 'xenium_'

def _process_transcripts(
        tdf: dask.dataframe.DataFrame,
        adata: anndata.AnnData,
        res: float,
        tile_size: int
    ) -> dask.dataframe.DataFrame:
    genes = sorted(adata.var.index.to_list())
    pool = set(genes)
    prop_to_idx = {p:i for i, p in enumerate(genes)}
    tdf['keep'] = da.asarray([True if p in pool else False for p in tdf['feature_name']])
    tdf = tdf.query(f'keep')

    tdf['x_target_res'] = tdf['x'] / res
    tdf['y_target_res'] = tdf['y'] / res
    tdf['chunk_x'] = (tdf['x_target_res'] // tile_size).astype(int)
    tdf['chunk_y'] = (tdf['y_target_res'] // tile_size).astype(int)
    tdf['cp_id'] = da.asarray([f'{x}_{y}_{z}' for x, y, z in zip(tdf['chunk_x'], tdf['chunk_y'], tdf['codeword_index'])])
    tdf['chunk'] = da.asarray([f'{x}_{y}' for x, y in zip(tdf['chunk_x'], tdf['chunk_y'])])

    # do prop idxs
    tdf['prop_idx'] = tdf['feature_name'].map(prop_to_idx.get).astype(int)

    # do entity idx
    chunks = da.unique(tdf['chunk'].values).compute()
    prop_to_chunk_to_count = {g:{chunk:0 for chunk in chunks} for g in genes}
    entity_idxs = []
    for g, chunk in zip(tdf['feature_name'], tdf['chunk']):
        entity_idxs.append(prop_to_chunk_to_count[g][chunk])
        prop_to_chunk_to_count[g][chunk] += 1
    tdf['entity_idx'] = da.asarray(entity_idxs)

    return tdf


def load_xenium(
    directory: Annotated[os.PathLike, Doc(
        """
        Directory path to outputs of Xenium Onboard Analysis pipeline.

        The [SpatialData io](https://spatialdata.scverse.org/projects/io/en/latest/) library is used to load the xenium run. 
        
        Typically, the directory must have the following files/directories:
        
        `morphology_focus/morphology_focus_*.ome.tif`
        `cell_boundaries.parquet`
        + Used to read in cell segmentation polygons
        `nucleus_boundaries.parquet`
        + Used to read in nuclei segmentation polygons
        `transcripts.zarr.zip`
        + Used to read in gene transcript locations
        """
    )],
    output_directory: Annotated[Union[os.PathLike | None], Doc(
        'Where to write output files used by cosilico. If not provided a temporary directory will be created.'
    )] = None,
    tile_size: Annotated[int, Doc(
        'Tile size used during image and feature writing.'
    )] = 512,
    max_features_per_tile: Annotated[int, Doc(
        'Maximum number of features displayed per zoomed tile.'
    )] = 1000,
    ) -> cdt.Experiment:
    """
    Load xenium experiment from directory.
    """
    print(f'Loading [bold blue]xenium[/bold blue] dataset from [bold green]{directory}[/bold green].')
    directory = Path(directory)

    if output_directory is None:
        output_directory_tmp = mkdtemp(prefix=TMP_PREFIX)
        output_directory = Path(output_directory_tmp)
    print(f'Will write output files to [bold red]{output_directory}[/bold red].')

    sd = xenium(directory)

    # load morphology image
    print('Loading morphology focus image.')
    assert 'morphology_focus' in sd.images, 'Morphology focus image not found.'
    img = sd.images['morphology_focus']['scale0'].image
    data = img.data
    channels = img.c.data
    res = 1 / get_transformation(sd.points['transcripts']).scale[0]

    print(f'Morphology focus image with (num_channels, height, width) of [bold green]{data.shape}[/bold green] and resolution of [bold green]{res}[/bold green] microns per pixel loaded.')

    ngff_filepath = output_directory / 'morphology_focus.ome.zarr.zip'
    print(f'Generating OME-NGFF image at [bold red]{ngff_filepath}[/bold red].')
    ngff_from_data(
        data=data,
        output_path=ngff_filepath,
        channels=channels,
        resolution=res,
        resolution_unit='µm',
        tile_height=tile_size,
        tile_width=tile_size,
    )
    ngff_group = zarr.open(ngff_filepath)
    n_zooms = len(ngff_group[0])

    print('Loading transcripts.')
    tdf = sd.points['transcripts']
    n_transcripts = tdf.shape[0].compute()
    print(f'[bold green]{n_transcripts}[/bold green] transcripts loaded.')

    adata = sd.tables['table']
    print(f'[bold green]{adata.shape[0]}[/bold green] cells loaded.')

    tdf = _process_transcripts(tdf, adata, res, tile_size)

    max_feats = tdf[['x', 'cp_id']].groupby('cp_id').count()[['x']].max().compute().item()
    max_feats = max(max_feats, max_features_per_tile)
    chunk_x_max, chunk_y_max = tdf[['chunk_x', 'chunk_y']].max().compute()



    




