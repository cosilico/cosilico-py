from pathlib import Path
from typing import Union
import os

from numpy.typing import NDArray
import numpy as np
import tifffile
from ome_types import from_xml
from typing_extensions import Annotated, Doc

from cosilico.data.types import Experiment

def load_xenium(
    directory: Annotated[Union[os.PathLike | str], Doc(
        """
        Directory path to outputs of Xenium Onboard Analysis pipeline.
        
        Directory must have the following files/directories:
        
        `morphology_focus/morphology_focus_*.ome.tif`
        + In V2, by default morphology_focus_0000.ome.tif is the only multiplex image used. Set `first_image_only` to False to include all morphology focus images. Note: including all images will take significantly more storage space.
        `cell_boundaries.parquet`
        + Used to read in cell segmentation polygons
        `nucleus_boundaries.parquet`
        + Used to read in nuclei segmentation polygons
        `transcripts.zarr.zip`
        + Used to read in gene transcript locations
        """
    )],
    first_image_only: Annotated[bool, Doc(
        'Whether first morphology focus image only. Note that including all images will take significantly more storage space'
    )] = True,
    ) -> Experiment:
    """
    Load xenium experiment from directory.
    """
    directory = Path(directory)
    # multiplex image
    focus_fps = os.listdir(directory / 'morphology_focus')





