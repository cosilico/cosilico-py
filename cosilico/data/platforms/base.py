from enum import Enum
from pathlib import Path
from typing import Union, List

from pydantic import BaseModel, Field, FilePath, model_validator
from typing_extensions import Annotated


from cosilico.data.types import (
    MultiplexImage, Layer
)


class SpatialExperiment(BaseModel):
    """
    Base class for all spatial experiments/platforms.
    """
    images: Annotated[List[MultiplexImage], Field(
        description="Images included in the experiment.",
    )]
    layers: Annotated[List[Layer], Field(
        description="Layers included in the experiment.",
    )]