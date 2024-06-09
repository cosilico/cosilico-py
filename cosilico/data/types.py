from enum import Enum
from pathlib import Path
from typing import Union, List

from pydantic import BaseModel, Field, FilePath, model_validator
from pydantic_extra_types.color import Color
from numpy.typing import NDArray
from typing_extensions import Annotated, Self
import numpy as np

from cosilico.data.conversion import to_microns_per_pixel

class DataType(str, Enum):
    multiplex = "multiplex"
    he = "he"
    xenium = "xenium"
    visium = "visium"

class PixelDataType(Enum):
    uint8 = np.uint8
    uint16 = np.uint16
    uint32 = np.uint32
    uint64 = np.uint64
    int8 = np.int8
    int16 = np.int16
    int32 = np.int32
    int64 = np.int64
    float16 = np.float16
    float32 = np.float32
    float64 = np.float64

class ScalingMethod(str, Enum):
    min_max = 'min_max'
    zero_max = 'zero_max'
    min_maxdtype = 'min_maxdtype'
    zero_maxdtype = 'zero_maxdtype'
    mindtype_max_dtype = 'mindtype_maxdtype'
    no_scale = 'no_scale'

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
        description="Names of channels in image. Must be ordered."
    )]
    data: Annotated[NDArray, Field(
        description="Pixel data for image. Image shape is (n_channels, height, width)."
    )]
    resolution: Annotated[float, Field(
        description="Resolution of image given in `resolution_unit`s per pixel",
        gt=0.
    )]
    resolution_unit: Annotated[str, Field(
        description="Resolution unit. Can be any string that is recognized by the [Pint](https://pint.readthedocs.io/en/stable/) Python library. In practice, this is a lot of unit string representations (covering many different unit systems) as long as they are reasonably named. For example, micron, micrometer, and μm could all be used for micrometers."
    )]
    source_filepath: Annotated[Union[FilePath | None], Field(
        description="Filepath source image was read from."
    )] = None
    name: Annotated[str, Field(
        description="Name of image. Defaults to `source_filepath` filename if not defined."
    )] = ''
    data_type: Annotated[Union[PixelDataType, None], Field(
        description="Pixel data type of the image. If not specified will be set to data type of `data`. If specified and `data_type` does not match data type of `data`, then data will be converted to the specified `data_type`."
    )] = None
    scaling_method: Annotated[ScalingMethod, Field(
        description="How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type."
    )] = ScalingMethod.min_max
    microns_per_pixel: Annotated[Union[float | None], Field(
        description="Resolution of image in microns per pixel. If not defined, will be automatically calculated from `resolution` and `resolution_unit`."
    )] = None
    view_settings: Annotated[Union[MultiplexViewSettings | None], Field(
        description="View settings for image. If not defined, will be automatically generated based on `channels`"
    )] = None

    @model_validator(mode='after')
    def set_name(self) -> Self:
        if not self.name:
            if self.source_filepath is not None:
                self.name = self.source_filepath.name
        return self

    @model_validator(mode='after')
    def calculate_microns_per_pixel(self) -> Self:
        if self.microns_per_pixel is None:
            self.microns_per_pixel = to_microns_per_pixel(self.resolution, self.resolution_unit)
        return self
    
    @model_validator(mode='after')
    def convert_data_type(self) -> Self:
        if self.data_type is None:
            self.data_type = PixelDataType(self.data.dtype)
        elif all(
            self.data_type is not None,
            self.data_type.value != self.data.dtype,
            ):
            if self.scaling_method.value == 'min_max':
                min_value, max_value = self.data.min(), self.data.max()
            elif self.scaling_method.value == 'mindtype_maxdtype':
                min_value, max_value = np.iinfo(self.data_type.value).min, np.iinfo(self.data_type.value).max
            elif self.scaling_method.value == 'zero_max':
                min_value, max_value = 0, self.data.max()
            elif self.scaling_method.value == 'zero_maxdtype':
                min_value, max_value = 0, np.iinfo(self.data_type.value).max
            elif self.scaling_method.value == 'min_maxdtype':
                min_value, max_value = self.data.min(), np.iinfo(self.data_type.value).max
  
            if self.scaling_method.value != 'no_scale':
                self.data = self.data - min_value
                self.data = self.data / max_value

            self.data = self.data.astype(self.data_type.value)

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






class LayerObjectType(str, Enum):
    point = "point"
    polygon = "polygon"

class LayerValueType(str, Enum):
    text = "text"
    categorical = "categorical"
    continuous = "continuous"

class Layer(BaseModel):
    """
    A image layer
    """
    name: Annotated[str, Field(
        description="Name of layer"
    )]
    object_type: Annotated[LayerObjectType, Field(
        description="Type of objects in layer."
    )]
    value_type: Annotated[LayerValueType, Field(
        description="Value type of objects in layer."
    )]


