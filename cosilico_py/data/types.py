from enum import Enum
from pathlib import Path
from typing import Union, List

from pydantic import BaseModel, Field, FilePath, model_validator
from pydantic_extra_types.color import Color
from typing_extensions import Annotated, Self
import numpy as np

from cosilico_py.data.conversion import to_microns_per_pixel

class DataType(str, Enum):
    multiplex = "multiplex"
    he = "he"
    xenium = "xenium"
    visium = "visium"

class PixelDataType(type, Enum):
    uint8: np.uint8
    uint16: np.uint16
    uint32: np.uint32
    uint64: np.uint64
    int8: np.int8
    int16: np.int16
    int32: np.int32
    int64: np.int64
    float16: np.float16
    float32: np.float32
    float64: np.float64

class ChannelViewSettings(BaseModel):
    """
    View settings for a channel.
    """
    name: Annotated[str, Field(
        description="Name of channel."
    )]
    min_value: Annotated[float, Field(
        description="Minimum value of channel."
    )] = 0.
    max_value: Annotated[float, Field(
        description="Maximum value of channel."
    )] = 255.
    color: Annotated[Color, Field(
        description="Channel color. Can be HEX, RGB, RGBA, or a [CSS Color Module Level 3](http://www.w3.org/TR/css3-color/#svg-color) string."
    )] = "white"

class MultiplexViewSettings(BaseModel):
    """
    View settings for a multiplex image.
    """
    channel_views: Annotated[List[ChannelViewSettings], Field(
        description="Channel views. Must be in the same order as the channels in the image."
    )]


class MultiplexImage(BaseModel):
    """
    A multiplex image.
    """
    channels: Annotated[List[str], Field(
        description="Names of channels in image. Must be ordered.",
    )]
    data_type: Annotated[PixelDataType, Field(
        description="Pixel data type of the image."
    )]
    resolution: Annotated[float, Field(
        description="Resolution of image given in `resolution_unit`s per pixel",
        gt=0.
    )]
    resolution_unit: Annotated[str, Field(
        description="Resolution unit. Can be any string that is recognized by the [Pint](https://pint.readthedocs.io/en/stable/) Python library. In practice, this is a lot of unit string representations (covering many different unit systems) as long as they are reasonably named. For example, micron, micrometer, and μm could all be used for micrometers."
    )]
    filepath: Annotated[Union[FilePath | None], Field(
        description="Filepath image was loaded from."
    )] = None
    name: Annotated[str, Field(
        description="Name of image."
    )] = ''
    microns_per_pixel: Annotated[Union[float | None], Field(
        description="Resolution of image in microns per pixel. If not defined, will be automatically calculated from `resolution` and `resolution_unit`."
    )] = None
    view_settings: Annotated[Union[MultiplexViewSettings | None], Field(
        description="View settings for image. If not defined, will be automatically generated based on `channels`"
    )] = None

    @model_validator(mode='after')
    def calculate_microns_per_pixel(self) -> Self:
        if self.microns_per_pixel is None:
            self.microns_per_pixel = to_microns_per_pixel(self.resolution, self.resolution_unit)
        return self
    
    @model_validator(mode='after')
    def generate_view_settings(self) -> Self:
        if self.view_settings is None:
            self.view_settings = MultiplexViewSettings(
                channel_views=[ChannelViewSettings(
                    name=name,
                    min_value=np.iinfo(self.data_type.value).min,
                    max_value=np.iinfo(self.data_type.value).max
                ) for name in self.channels]
            )
        return self

class Layer(BaseModel):
    """
    A image layer
    """
    name: Annotated[str, Field(
        description="Name of layer"
    )]
    