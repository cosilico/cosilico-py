from enum import Enum
from pathlib import Path
from typing import Union, List, Dict

from geojson_pydantic import Feature, FeatureCollection
from pydantic import BaseModel, Field, FilePath, model_validator
from pydantic_extra_types.color import Color
from numpy.typing import NDArray
from scipy.sparse import spmatrix
from typing_extensions import Annotated, Self
import numpy as np
import zarr

from cosilico.data.colors import Colormap
from cosilico.data.conversion import to_microns_per_pixel, scale_data, convert_dtype
from cosilico.data.zarr import to_zarr

class DataType(str, Enum):
    multiplex = "multiplex"
    he = "he"
    xenium = "xenium"
    visium = "visium"

class PixelDataType(Enum):
    uint8 = np.uint8
    uint16 = np.uint16
    uint32 = np.uint32
    int8 = np.int8
    int16 = np.int16
    int32 = np.int32
    float16 = np.float16
    float32 = np.float32

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
    data: Annotated[Union[zarr.Array | NDArray], Field(
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
        return convert_dtype(self)
    
    @model_validator(mode='after')
    def generate_view_settings(self) -> Self:
        if self.view_settings is None:
            self.view_settings = MultiplexViewSettings(
                channel_views=[ChannelViewSettings(
                    min_value=np.iinfo(self.data_type.value).min,
                    max_value=np.iinfo(self.data_type.value).max
                ) for _ in self.channels]
            )
        return self






## Layer stuff

class RangeScalingMethod(str, Enum):
    zero_max = 'zero_max'
    min_max = 'min_max'
    symmetrical = 'symmetrical'

class PointShape(str, Enum):
    circle = "circle"

class GeometryViewSettings(BaseModel):
    """
    Default view settings applied to Point, MultiPoint, LineString, MultiLineString, Polygon and MultiPolygon geometries.
    """
    fill_color: Annotated[Color, Field(
        description="Default fill color. Polygons will be this color if no properties are provided. Can be HEX, RGB, RGBA, or a [CSS Color Module Level 3](http://www.w3.org/TR/css3-color/#svg-color) string."
    )] = 'lightgray'
    fill_alpha: Annotated[float, Field(
        description="Fill alpha.",
        ge=0.,
        le=1.
    )] = 1.
    border_color: Annotated[Color, Field(
        description="Default border color. Can be HEX, RGB, RGBA, or a [CSS Color Module Level 3](http://www.w3.org/TR/css3-color/#svg-color) string."
    )] = 'darkgray'
    border_alpha: Annotated[float, Field(
        description="Border alpha.",
        ge=0.,
        le=1.
    )] = 1.
    border_thickness: Annotated[float, Field(
        description="Border thickness in pixels",
        ge=0.
    )] = 1.
    point_shape: Annotated[PointShape, Field(
        description="Default shape used to render Point and MultiPoint geometries."
    )] = PointShape.circle


class LayerViewSettings(BaseModel):
    """
    View settings for Layer.
    """
    colormap_continuous: Annotated[Colormap, Field(
        description='Default colormap used for continuous properties.'
    )] = Colormap.viridis
    colormap_categorical: Annotated[Colormap, Field(
        description='Default colormap used for categorical properties.'
    )] = Colormap.tab10
    range_scaling_method: Annotated[RangeScalingMethod, Field(
        description='What range scaling method to use for continuous colormap.'
    )] = RangeScalingMethod.min_max
    geometry_view_settings: Annotated[GeometryViewSettings, Field(
        description='Default view settings for layer geometries.'
    )] = GeometryViewSettings()


class PropertyGroup(BaseModel):
    """
    Group of related properties of the same data type. Must have same data type and be stored in matrix form.
    """
    properties: Annotated[List[str], Field(
        description='Names of properties in property group. Are column names in `data`'
    )]
    data: Annotated[Union[zarr.Array | NDArray | spmatrix], Field(
        description='Matrix of shape (num_features, num_properties).'
    )]
    data_type: Annotated[Union[PixelDataType | None], Field(
        description='Data type to store `data` as. If not specified, will match data type of `data`. If specified and different from `data` data type then `data` will be transformed to the given data type with the specified `scaling_method`.'
    )] = None
    scaling_method: Annotated[ScalingMethod, Field(
        description="How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type."
    )] = ScalingMethod.min_max

    @model_validator(mode='after')
    def data_to_zarr(self) -> Self:
        if not isinstance(self.data, zarr.Array):
            self.data = to_zarr(self.data)

        if len(set(self.properties)) != len(self.properties):
            raise ValueError('All property names must be unique.')
        
        if len(self.properties) != self.data.shape[1]:
            raise ValueError(f'Length of `properties` (got {len(self.property_names)}) must be equal to column dimension of `data` (got {self.data.shape[1]})')

        self.data.attrs['properties'] = self.properties

        return self

    @model_validator(mode='after')
    def convert_data_type(self) -> Self:
        return convert_dtype(self)


class Property(BaseModel):
    """
    A property of layer features.
    """
    data: Annotated[Union[zarr.Array | NDArray], Field(
        description='1D array of length num_features. Order must match order of features in Layer.'
    )]
    data_type: Annotated[Union[PixelDataType | None], Field(
        description='Data type to store `data` as. If not specified, will match data type of `data`. If specified and different from `data` data type then `data` will be transformed to the given data type with the specified `scaling_method`.'
    )] = None
    scaling_method: Annotated[ScalingMethod, Field(
        description="How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type."
    )] = ScalingMethod.min_max

    @model_validator(mode='after')
    def data_to_zarr(self) -> Self:
        if not isinstance(self.data, zarr.Array):
            self.data = to_zarr(self.data)

        return self

    @model_validator(mode='after')
    def convert_data_type(self) -> Self:
        return convert_dtype(self)


class Layer(BaseModel):
    """
    A image layer.
    """
    name: Annotated[str, Field(
        description="Name of layer"
    )]
    features: Annotated[FeatureCollection, Field(
        description="Layer features. Must be GeoJSON format."
    )]
    feature_ids: Annotated[List[Union[str | int]], Field(
        description="Feature IDs. Must be unique across all features."
    )]
    view_settings: Annotated[Union[LayerViewSettings | None], Field(
        description="View settings for Layer. If not defined, will be automatically generated."
    )] = LayerViewSettings()

    @model_validator(mode='after')
    def validate_features(self) -> Self:
        if self.feature_properties is not None:
            if len(self.feature_ids) != len(self.features):
                raise ValueError(f'Length of `feature_properties.feature_ids` was {len(self.feature_ids)} and length of `features` was {self.features}. Length of `feature_ids` and `features` must be equal.')
            if len(set(self.feature_ids)) != len(self.feature_ids):
                raise ValueError('Values in `feature_ids` must be unique.')

        return self
    
    def validate_feature_properties(self) -> Self:
        if self.name_to_feature_properties is None:
            return self

        for name, prop in self.name_to_feature_properties.items():
            missing = [x for x in ['data', 'names'] if x not in prop.group_keys()]
            if missing:
                raise ValueError(f'{missing} were missing from {name}. Make sure feature property Zarr is properly formatted according to XXX.xx')

            

        for feature_property in self.group_to_feature_properties.values():
            if len(self.feature_ids) != feature_property.data.shape[0]:
                raise ValueError(f'Length of `feature_ids` was {len(self.feature_ids)} and number of rows in `feature_properties.data` was {feature_property.data.shape[0]}. Length of `feature_ids` and num rows in `feature_properties.data` must be equal.')
        return self