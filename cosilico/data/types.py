from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Union, List, Dict, Iterable
import os

from geopandas.geodataframe import GeoDataFrame, GeoSeries
from geojson_pydantic import Feature, FeatureCollection
from pydantic import BaseModel, Field, FilePath, model_validator, DirectoryPath
from pydantic_extra_types.color import Color
from numpy.typing import NDArray, DTypeLike
from rich import print
from scipy.sparse import spmatrix
from typing_extensions import Annotated, Self, TypedDict
import dask.array as da
import numpy as np
import pandas as pd
import zarr
import zarr.convenience

from cosilico.data.colors import Colormap
from cosilico.data.units import to_microns_per_pixel
from cosilico.data.ome import ngff_from_data
from cosilico.data.scaling import scale_data, SCALING_METHODS
from cosilico.data.platforms import Platform, PlatformName
from cosilico.data.zarr import to_zarr, delete_if_tmp
from cosilico.typing import ArrayLike, NGFF_DTYPES

class ChannelViewSettings(BaseModel, validate_assignment=True):
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

class MultiplexViewSettings(BaseModel, validate_assignment=True):
    """
    View settings for a multiplex image.
    """
    channel_views: Annotated[List[ChannelViewSettings], Field(
        description="Channel views. Must be in the same order as the channels in the image."
    )]


class MultiplexImage(BaseModel, validate_assignment=True, arbitrary_types_allowed=True):
    """
    A multiplex image.
    """
    name: Annotated[str, Field(
        description="Name of image."
    )]
    channels: Annotated[List[str], Field(
        description="Names of channels in image. Must be ordered."
    )]
    data: Annotated[ArrayLike, Field(
        description="Pixel data for image. Image shape is (n_channels, height, width).",
    )]
    resolution: Annotated[Union[float | None], Field(
        description="Resolution of image given in `resolution_unit`s per pixel",
        gt=0.
    )]
    resolution_unit: Annotated[Union[str | None], Field(
        description='Resolution unit. Must be valid UnitsLength as specified in [OME-TIF schema](https://www.openmicroscopy.org/Schemas/Documentation/Generated/OME-2016-06/ome.html)'
        # description="Resolution unit. Can be any string that is recognized by the [Pint](https://pint.readthedocs.io/en/stable/) Python library. In practice, this is a lot of unit string representations (covering many different unit systems) as long as they are reasonably named. For example, micron, micrometer, and μm could all be used for micrometers."
    )] = 'µm'
    ngff_filepath: Annotated[Union[os.PathLike | None], Field(
        description="Filepath where OME-NGFF zarr will be written. By default will be written to system TMP directory."
    )] = None
    ngff_tile_size: Annotated[int, Field(
        description='Tile size to use when writing OME-NGFF zarr.',
        gt=0,
    )] = 512
    ngff_zarr: Annotated[Union[zarr.Group | None], Field(
        description='Zarr group representing OME-NGFF. By default will be automatically generated.'
    )] = None
    data_type: Annotated[Union[str | np.dtype | None], Field(
        description=f"Pixel data type of the OME-NGFF image. If not specified will be set to data type of `data`. If specified and `data_type` does not match data type of `data`, then data will be converted to the specified `data_type`. Currently the following data types are supported: {NGFF_DTYPES}."
    )] = np.uint8
    scaling_method: Annotated[str, Field(
        description=f"How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type. Must be one of {SCALING_METHODS}"
    )] = 'min_max'
    force_scale: Annotated[bool, Field(
        description='Force data to scale based on scaling_method, even if data_type and data.dtype match.'
    )] = False
    view_settings: Annotated[Union[MultiplexViewSettings | None], Field(
        description="View settings for image. If not defined, will be automatically generated based on `channels`"
    )] = None
    id: Annotated[Union[str | int | None], Field(
        description='Image ID. Not required for initialization. Populated by supabase.'
    )] = None
    experiment_id: Annotated[Union[str | int | None], Field(
        description='Experiment ID image belongs to. Not required for initialization. Populated by supabase.'
    )] = None

    # @model_validator(mode='after')
    # def validate_source_path(self) -> Self:
    #     if self.source_path is not None:
    #         ext = self.source_path.suffix
    #         assert ext == '.zarr', f'Directory must be OME-NGFF formatted zarr. Got extension {ext}'
    #     return self

    # @model_validator(mode='after')
    # def set_name(self) -> Self:
    #     if not self.name:
    #         if self.source_filepath is not None:
    #             self.name = self.source_filepath.name
    #     return self

    # @model_validator(mode='after')
    # def calculate_microns_per_pixel(self) -> Self:
    #     self.microns_per_pixel = to_microns_per_pixel(self.resolution, self.resolution_unit)
    #     return self
    
    @model_validator(mode='after')
    def check_data_type(self) -> Self:
        if self.data_type is not None:
            dtype_str = np.dtype(self.data_type).name
            assert dtype_str in NGFF_DTYPES, f'data_type must be one of {NGFF_DTYPES}, got {dtype_str}.'
        else:
            self.data_type = np.uint8
        return self
    
    @model_validator(mode='after')
    def generate_ome_ngff(self) -> Self:
        if self.ngff_filepath is None:
            tmpfile = NamedTemporaryFile(delete=False)
            self.ngff_filepath = tmpfile.name + '.ome.zarr.zip'
            path = Path(self.ngff_filepath)
            tmpfile.close()
        else:
            path = Path(self.ngff_filepath)
        assert path.suffix == '.zip', f'ngff_filepath must have .zip extension, got {path}.'

        if self.ngff_zarr is None:
            print(f'Generating OME-NGFF image at [bold red]{path}[/bold red].')
            ngff_from_data(
                data=self.data,
                output_path=path,
                channels=self.channels,
                resolution=self.resolution,
                resolution_unit=self.resolution_unit,
                tile_height=self.ngff_tile_size,
                tile_width=self.ngff_tile_size,
            )
            self.ngff_zarr = zarr.open(path)
        return self
    

    
    @model_validator(mode='after')
    def generate_view_settings(self) -> Self:
        if self.view_settings is None:
            self.view_settings = MultiplexViewSettings(
                channel_views=[ChannelViewSettings(
                    min_value=np.iinfo(np.dtype(self.data_type)).min,
                    max_value=np.iinfo(np.dtype(self.data_type)).max
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

class LayerType(str, Enum):
    platform = 'platform'
    user = 'user'

class FeatureGeometry(str, Enum):
    point = 'point'
    line = 'line'
    polygon = 'polygon'
    multipoint = 'multipoint'
    multiline = 'multiline'
    multipolygon = 'multipolygon'

class GeometryViewSettings(BaseModel, validate_assignment=True):
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


class PropertyViewSettings(BaseModel, validate_assignment=True):
    """
    View settings for a Layer property.
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

class Property(BaseModel, validate_assignment=True, arbitrary_types_allowed=True):
    """
    A property of layer features.
    """
    name: Annotated[str, Field(
        description='Name of property.'
    )]
    # data: Annotated[Union[zarr.Array | NDArray | da.core.Array | pd.Series], Field(
    #     description='1D array of length num_features. Order must match order of features in Layer.'
    # )]
    data: Annotated[zarr.Array, Field(
        description='1D array of length num_features. Order must match order of features in Layer.'
    )]
    feature_ids: Annotated[Union[Iterable[Union[str | int]] | None], Field(
        description='Layer feature IDs properties correspond to. Must match order of rows in `data`.'
    )] = None
    # data_type: Annotated[Union[PixelDataType | None], Field(
    #     description='Data type to store `data` as. If not specified, will match data type of `data`. If specified and different from `data` data type then `data` will be transformed to the given data type with the specified `scaling_method`.'
    # )] = None
    # scaling_method: Annotated[ScalingMethod, Field(
    #     description="How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type."
    # )] = ScalingMethod.min_max
    # force_scale: Annotated[bool, Field(
    #     description='Force data to scale based on scaling_method, even if data_type and data.dtype match.'
    # )] = False
    view_settings: Annotated[Union[PropertyViewSettings | None], Field(
        description="View settings for a Layer property. If not provided, will be displayed with default view settings for Layer."
    )] = None
    id: Annotated[Union[str | int | None], Field(
        description='Property ID. Not required for initialization. Populated by supabase.'
    )] = None
    layer_id: Annotated[Union[str | int | None], Field(
        description='Layer ID property is linked to. Not required for initialization. Populated by supabase.'
    )] = None

    @model_validator(mode='after')
    def process_feature_ids(self) -> Self:
        if self.feature_ids is None:
            if any([
                isinstance(self.data, pd.Series),
                isinstance(self.data, pd.DataFrame)
                ]):
                self.feature_ids = self.data.index.to_list()
            else:
                raise ValueError(f'feature_ids is not provided and data is of type {type(self.data)}. Unless data is a pd.Series, feature_ids must be provided.')

        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError('All feature ids must be unique.')

        if len(self.feature_ids) != self.data.shape[0]:
            raise ValueError(f'Length of `feature_ids` (got {len(self.feature_ids)}) must be equal to row dimension of `data` (got {self.data.shape[0]})')

        if not isinstance(self.feature_ids, list):
            self.feature_ids = list(self.feature_ids)

        return self

    @model_validator(mode='after')
    def data_to_zarr(self) -> Self:
        if len(self.data.shape) > 1:
            raise ValueError(f'data must be 1D array. Got array with shape {self.data.shape}.')
        # if not isinstance(self.data, zarr.Array):
        #     self.data = to_zarr(self.data)

        return self

    # @model_validator(mode='after')
    # def convert_data_type(self) -> Self:
    #     return convert_dtype(self, scale=self.force_scale)
    
    # def __del__(self):
    #     if isinstance(self.data, zarr.Array):
    #         delete_if_tmp(self.data)


class PropertyGroup(Property, validate_assignment=True, arbitrary_types_allowed=True):
    """
    Group of related properties of the same data type. Must have same data type and be stored in matrix form.
    """
    # data: Annotated[Union[zarr.Array | NDArray | spmatrix | da.core.Array], Field(
    #     description='Matrix of shape (num_features, num_properties).'
    # )]
    data: Annotated[zarr.Array, Field(
        description='Matrix of shape (num_features, num_properties).'
    )]
    property_names: Annotated[Union[Iterable[str] | None], Field(
        description='Names of properties in property group. Are the column names in `data`'
    )] = None
    view_settings: Annotated[Dict[str, Union[PropertyViewSettings | None]], Field(
        description="View settings for a property group. A dictionary mapping property names (keys) to property view settings (values). If not provided, all properties will be displayed with default view settings for Layer. If a dictionary is provided, all properties specified in the dictionary will be displayed with the given view settings, the remaining properties will be visualized with default layer view settings."
    )] = None

    @model_validator(mode='after')
    def process_property_names(self) -> Self:
        if self.property_names is None:
            if isinstance(self.data, pd.DataFrame):
                self.property_names = list(self.data.columns)
            else:
                raise ValueError(f'property_names is not provided and data is of type {type(self.data)}. Unless data is a pd.DataFrame, property_names must be provided.')

        if len(set(self.property_names)) != len(self.property_names):
            raise ValueError('All property names must be unique.')
        
        if len(self.property_names) != self.data.shape[1]:
            raise ValueError(f'Length of `property_names` (got {len(self.property_names)}) must be equal to column dimension of `data` (got {self.data.shape[1]})')

        return self

    # @model_validator(mode='after')
    # def data_to_zarr(self) -> Self:
    #     if not isinstance(self.data, zarr.Array):
    #         self.data = to_zarr(self.data)

    #     self.data.attrs['property_names'] = self.property_names

    #     return self
    
    # @model_validator(mode='after')
    # def convert_data_type(self) -> Self:
    #     return convert_dtype(self, scale=self.force_scale, axis=0)

    # def __del__(self):
    #     if isinstance(self.data, zarr.Array):
    #         delete_if_tmp(self.data)
    
class LayerFeatures(BaseModel, validate_assignment=True, arbitrary_types_allowed=True):
    """
    Feature of a layer.
    """
    data: Annotated[zarr.Array, Field(
        description='Underlying data describing features.'
    )]
    # data: Annotated[Union[zarr.Array | da.core.Array | NDArray | GeoSeries | GeoDataFrame | FeatureCollection], Field(
    #     description="""
    #                 Underlying data describing features.

    #                 Can be either a numpy/Zarr/dask Array or GeoJSON FeatureCollection.

    #                 A GeoDataFrame is assumed to have a column "geometry" that holds shaply geometry objects
    #                 A GeoSeries is assumed to be a geoseries holding shaply geometry objects

    #                 A array can be the following shape based on feature geometry type:
    #                 + point - (n_features, 2)
    #                 + line - (n_features, 2, 2)
    #                 + polygon - (n_features, n_max_poly_coords, 2)
    #                 + multipoint - (n_features, n_max_objects, 2)
    #                 + multiline - (n_features, n_max_objects, 2, 2)
    #                 + multipolygon - (n_features, n_max_objects, n_max_poly_coords, 2)

    #                 n_features - number of features in layer
    #                 n_max_objects - maximum number of objects in a multi-geometry
    #                 n_max_poly_coords - maximum number of points describing polygon coordinates

    #                 Coordinates are stored as float32

    #                 ```python
    #                 array.attrs["feature_ids"] = feature_ids
    #                 ```
                    
    #                 GeoJSON FeatureCollection should only be used for Layers with a small number of features (i.e. manually drawn annotations, etc.) or it is necessary to have multiple geometry types within the same layer (i.e. mixing points and polygons, etc.). GeoJSON files get large very quickly and should be avoided if a Zarr can be used.

    #                 """
    # )]
    feature_ids: Annotated[Union[List[Union[str | int]] | None], Field(
        description="Feature IDs. Must be unique across all features."
    )] = None
    geometry: Annotated[Union[FeatureGeometry | None], Field(
        description='Geometry type of feature contained in layer.'
    )] = None

    @model_validator(mode='after')
    def validate_features(self) -> Self:
        # if isinstance(self.data, FeatureCollection):
        #     n_features = len(self.data.features)
        #     assert self.feature_ids is not None, 'Must provide feature_ids if data is a FeatureCollection.'
        # if isinstance(self.data, GeoDataFrame) or isinstance(self.data, GeoSeries):
        #     n_features = self.data.shape[0]
        #     self.feature_ids = self.data.index.to_list()
        # else:
        n_features = self.data.shape[0]
        assert self.feature_ids is not None, 'Must provide feature_ids if data is an array.'


        if len(self.feature_ids) != n_features:
            raise ValueError(f'Length of `feature_ids` was {len(self.feature_ids)} and `data` was length {n_features}. Length of `feature_ids` and length of `data` must be equal.')
        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError('Values in `feature_ids` or GeoSeries index must be unique.')

        return self

    @model_validator(mode='after')
    def validate_geometry(self) -> Self:
        
        # TODO

        return self


class Layer(BaseModel, validate_assignment=True):
    """
    A image layer.
    """
    name: Annotated[str, Field(
        description="Name of layer"
    )]
    features: Annotated[LayerFeatures, Field(
        description="Layer features."
    )]
    properties: Annotated[Union[List[Union[Property | PropertyGroup] | None]], Field(
        description='Properties associated with layer.'
    )] = None
    layer_type: Annotated[LayerType, Field(
        description='Type of layer. `user` layers are defined by users. `platform` layers are defined by the experimental platform.'
    )] = LayerType.user
    view_settings: Annotated[Union[PropertyViewSettings | None], Field(
        description="Default view settings for properties of the Layer. If not defined, will be automatically generated."
    )] = PropertyViewSettings()
    id: Annotated[Union[str | int | None], Field(
        description='Layer ID. Not required for initialization. Populated by supabase.'
    )] = None
    experiment_id: Annotated[Union[str | int | None], Field(
        description='Experiment ID layer is assigned to. Not required for initialization. Populated by supabase.'
    )] = None
    
    @model_validator(mode='after')
    def validate_properties(self) -> Self:
        if self.properties is not None:
            for prop in self.properties:
                if prop.feature_ids != self.features.feature_ids:
                    raise ValueError(f'Feature ids of property must be equal to feature ids of layer. Feature ids of {prop.name} do not match.')
        return self


## experiments

class Experiment(BaseModel, validate_assignment=True):
    name: Annotated[str, Field(
        description="Name of Experiment."
    )]
    platform: Annotated[Platform, Field(
        description='Platform used to generate experiment.'
    )]
    images: Annotated[Union[List[MultiplexImage] | None], Field(
        description='Images for experiment.'
    )]
    layers: Annotated[Union[List[Layer] | None], Field(
        description='Layers for experiment.'
    )]
    id: Annotated[Union[str | int | None], Field(
        description='Experiment ID. Not required for initialization. Populated by supabase.'
    )] = None
    collection_id: Annotated[Union[str | int | None], Field(
        description='Collection ID experiment is a member of. Not required for initialization. Populated by supabase.'
    )] = None