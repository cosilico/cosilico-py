from enum import Enum
from pathlib import Path
from typing import Union, List

from pydantic import BaseModel, Field, FilePath, model_validator
from typing_extensions import Annotated


from cosilico.data.types import (
    MultiplexImage
)


class SpatialExperiment(BaseModel):
    """
    Base experiment class, probably abstract?
    """

class Xenium(SpatialExperiment):
    """
    Xenium experiment
    """