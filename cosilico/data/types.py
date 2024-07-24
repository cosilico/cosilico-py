from enum import Enum
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Union, List, Dict, Iterable
import os

import dask.dataframe
from geopandas.geodataframe import GeoDataFrame, GeoSeries
from geojson_pydantic import Feature, FeatureCollection
from pydantic import BaseModel, Field, FilePath, model_validator, DirectoryPath
from pydantic_extra_types.color import Color
from numpy.typing import NDArray, DTypeLike
from rich import print
from scipy.sparse import spmatrix
from typing_extensions import Annotated, Self, TypedDict
import dask
import dask.array as da
import numpy as np
import pandas as pd
import zarr
import zarr.convenience

from cosilico.data.colors import Colormap
from cosilico.data.units import to_microns_per_pixel
from cosilico.data.ome import ngff_from_data
from cosilico.data.scaling import scale_data, ScalingMethod
from cosilico.data.platforms import Platform, PlatformName
from cosilico.data.geopandas import geoseries_to_coords
from cosilico.data.zarr import to_zarr, delete_if_tmp
from cosilico.typing import ArrayLike, DataFrameLike, NGFF_DTYPES

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
    scaling_method: Annotated[ScalingMethod, Field(
        description=f"How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type. Must be one of {[x.value for x in ScalingMethod]}"
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
    
    @model_validator(mode='after')
    def check_data_type(self) -> Self:
        if self.data_type is not None:
            dtype_str = np.dtype(self.data_type).name
            assert dtype_str in NGFF_DTYPES, f'data_type must be one of {NGFF_DTYPES}, got {dtype_str}.'
        else:
            self.data_type = np.uint8
        return self

    @model_validator(mode='after')
    def set_ngff_filepath(self) -> Self:
        if self.ngff_filepath is None:
            tmpfile = NamedTemporaryFile(delete=False)
            self.ngff_filepath = tmpfile.name + '.ome.zarr.zip'
            tmpfile.close()
        path = Path(self.ngff_filepath)

        assert path.suffix == '.zip', f'ngff_filepath must have .zip extension, got {path}.'

        return self
    
    @model_validator(mode='after')
    def generate_ome_ngff(self) -> Self:
        if self.ngff_zarr is None:
            print(f'Generating OME-NGFF image at [bold red]{self.ngff_filepath}[/bold red].')
            ngff_from_data(
                data=self.data,
                output_path=self.ngff_filepath,
                channels=self.channels,
                resolution=self.resolution,
                resolution_unit=self.resolution_unit,
                tile_height=self.ngff_tile_size,
                tile_width=self.ngff_tile_size,
            )
            self.ngff_zarr = zarr.open(self.ngff_filepath)
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

class GeometryType(str, Enum):
    point = 'point'
    polygon = 'polygon'

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



    
    # @model_validator(mode='after')
    # def generate_zarr(self) -> Self:
    #     df = dask.dataframe.from_pandas(
    #         pd.DataFrame(index=np.arange(len(self.feature_ids))),
    #         npartitions=1
    #     )

    #     tdf['x_target_res'] = tdf['x'] / res
    #     tdf['y_target_res'] = tdf['y'] / res
    #     tdf['chunk_x'] = (tdf['x_target_res'] // tile_size).astype(int)
    #     tdf['chunk_y'] = (tdf['y_target_res'] // tile_size).astype(int)
    #     tdf['cp_id'] = da.asarray([f'{x}_{y}_{z}' for x, y, z in zip(tdf['chunk_x'], tdf['chunk_y'], tdf['codeword_index'])])
    #     tdf['chunk'] = da.asarray([f'{x}_{y}' for x, y in zip(tdf['chunk_x'], tdf['chunk_y'])])

    #     # do prop idxs
    #     tdf['prop_idx'] = tdf['feature_name'].map(prop_to_idx.get).astype(int)

    #     # do entity idx
    #     chunks = da.unique(tdf['chunk'].values).compute()
    #     prop_to_chunk_to_count = {g:{chunk:0 for chunk in chunks} for g in genes}
    #     entity_idxs = []
    #     for g, chunk in zip(tdf['feature_name'], tdf['chunk']):
    #         entity_idxs.append(prop_to_chunk_to_count[g][chunk])
    #         prop_to_chunk_to_count[g][chunk] += 1
    #     tdf['entity_idx'] = da.asarray(entity_idxs)

    #     return self

    # @model_validator(mode='after')
    # def data_to_zarr(self) -> Self:
    #     if len(self.data.shape) > 1:
    #         raise ValueError(f'data must be 1D array. Got array with shape {self.data.shape}.')
    #     # if not isinstance(self.data, zarr.Array):
    #     #     self.data = to_zarr(self.data)

    #     return self

    # @model_validator(mode='after')
    # def convert_data_type(self) -> Self:
    #     return convert_dtype(self, scale=self.force_scale)
    
    # def __del__(self):
    #     if isinstance(self.data, zarr.Array):
    #         delete_if_tmp(self.data)




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
    
# class LayerFeatures(BaseModel, validate_assignment=True, arbitrary_types_allowed=True):
#     """
#     Feature of a layer.
#     """
#     data: Annotated[zarr.Array, Field(
#         description='Underlying data describing features.'
#     )]
#     # data: Annotated[Union[zarr.Array | da.core.Array | NDArray | GeoSeries | GeoDataFrame | FeatureCollection], Field(
#     #     description="""
#     #                 Underlying data describing features.

#     #                 Can be either a numpy/Zarr/dask Array or GeoJSON FeatureCollection.

#     #                 A GeoDataFrame is assumed to have a column "geometry" that holds shaply geometry objects
#     #                 A GeoSeries is assumed to be a geoseries holding shaply geometry objects

#     #                 A array can be the following shape based on feature geometry type:
#     #                 + point - (n_features, 2)
#     #                 + line - (n_features, 2, 2)
#     #                 + polygon - (n_features, n_max_poly_coords, 2)
#     #                 + multipoint - (n_features, n_max_objects, 2)
#     #                 + multiline - (n_features, n_max_objects, 2, 2)
#     #                 + multipolygon - (n_features, n_max_objects, n_max_poly_coords, 2)

#     #                 n_features - number of features in layer
#     #                 n_max_objects - maximum number of objects in a multi-geometry
#     #                 n_max_poly_coords - maximum number of points describing polygon coordinates

#     #                 Coordinates are stored as float32

#     #                 ```python
#     #                 array.attrs["feature_ids"] = feature_ids
#     #                 ```
                    
#     #                 GeoJSON FeatureCollection should only be used for Layers with a small number of features (i.e. manually drawn annotations, etc.) or it is necessary to have multiple geometry types within the same layer (i.e. mixing points and polygons, etc.). GeoJSON files get large very quickly and should be avoided if a Zarr can be used.

#     #                 """
#     # )]
#     feature_ids: Annotated[Union[List[Union[str | int]] | None], Field(
#         description="Feature IDs. Must be unique across all features."
#     )] = None
#     geometry: Annotated[Union[FeatureGeometry | None], Field(
#         description='Geometry type of feature contained in layer.'
#     )] = None

#     @model_validator(mode='after')
#     def validate_features(self) -> Self:
#         # if isinstance(self.data, FeatureCollection):
#         #     n_features = len(self.data.features)
#         #     assert self.feature_ids is not None, 'Must provide feature_ids if data is a FeatureCollection.'
#         # if isinstance(self.data, GeoDataFrame) or isinstance(self.data, GeoSeries):
#         #     n_features = self.data.shape[0]
#         #     self.feature_ids = self.data.index.to_list()
#         # else:
#         n_features = self.data.shape[0]
#         assert self.feature_ids is not None, 'Must provide feature_ids if data is an array.'


#         if len(self.feature_ids) != n_features:
#             raise ValueError(f'Length of `feature_ids` was {len(self.feature_ids)} and `data` was length {n_features}. Length of `feature_ids` and length of `data` must be equal.')
#         if len(set(self.feature_ids)) != len(self.feature_ids):
#             raise ValueError('Values in `feature_ids` or GeoSeries index must be unique.')

#         return self
    
#     # @model_validator(mode='after')
#     # def generate_zarr(self) -> Self:
#     #     df = dask.dataframe.from_pandas(
#     #         pd.DataFrame(index=np.arange(len(self.feature_ids))),
#     #         npartitions=1
#     #     )

#     #     tdf['x_target_res'] = tdf['x'] / res
#     #     tdf['y_target_res'] = tdf['y'] / res
#     #     tdf['chunk_x'] = (tdf['x_target_res'] // tile_size).astype(int)
#     #     tdf['chunk_y'] = (tdf['y_target_res'] // tile_size).astype(int)
#     #     tdf['cp_id'] = da.asarray([f'{x}_{y}_{z}' for x, y, z in zip(tdf['chunk_x'], tdf['chunk_y'], tdf['codeword_index'])])
#     #     tdf['chunk'] = da.asarray([f'{x}_{y}' for x, y in zip(tdf['chunk_x'], tdf['chunk_y'])])

#     #     # do prop idxs
#     #     tdf['prop_idx'] = tdf['feature_name'].map(prop_to_idx.get).astype(int)

#     #     # do entity idx
#     #     chunks = da.unique(tdf['chunk'].values).compute()
#     #     prop_to_chunk_to_count = {g:{chunk:0 for chunk in chunks} for g in genes}
#     #     entity_idxs = []
#     #     for g, chunk in zip(tdf['feature_name'], tdf['chunk']):
#     #         entity_idxs.append(prop_to_chunk_to_count[g][chunk])
#     #         prop_to_chunk_to_count[g][chunk] += 1
#     #     tdf['entity_idx'] = da.asarray(entity_idxs)

#     #     return self

#     @model_validator(mode='after')
#     def validate_geometry(self) -> Self:
        
#         # TODO

#         return self


class Layer(BaseModel, validate_assignment=True, arbitrary_types_allowed=True):
    """
    A image layer.
    """
    name: Annotated[str, Field(
        description="Name of layer"
    )]
    geometry_type: Annotated[GeometryType, Field(
        description='Type of Geometry. Can be `point` or `polygon`.'
    )]
    coordinates: Annotated[ArrayLike, Field(
        description="""
        Coordinates for layer geometries.
        
        Can be the following inputs based on geometry type.
- 
        Points
        + ArrayLike(n_features, 2) where X coordinate is the first column, and Y coordinate is the second column
        + List[List] Where each first entry of each nested list is X coordinate and second entry is Y coordinate
        Polygons
        + GeoSeries - GeoPandas GeoSeries of shapely polygon objects where each entity represents a geometry object.
        + List[ArrayLike(2, n_verts)] where each list entry is a polygon where the first axis is <X, Y> and n_verts is the number of vertices for the given polygon.
        + ArrayLike(n_features, 2, max_verts) where second axis is <X, Y>. n_verts is the maximum number of vertices a polygon can have in the entire layer. Unused verticies should be -1.
        """
    )]
    image: Annotated[MultiplexImage, Field(
        description='Image the layer is annotating.'
    )]
    resolution: Annotated[Union[float | None], Field(
        description='Resolution of `coordinates`. Must be in same resolution unit as `image.resolution_unit`. If not provided, coordinates are assumed to be the same resolution as `image`.',
        gt=0.
    )] = None
    feature_ids: Annotated[Union[List[str] | None], Field(
        description='Feature IDs. Must be list of strings.'
    )] = None
    properties: Annotated[List, Field(
        description='Properties associated with layer.'
    )] = None
    layer_type: Annotated[LayerType, Field(
        description='Type of layer. `user` layers are defined by users. `platform` layers are defined by the experimental platform.'
    )] = 'user'
    max_visible_features: Annotated[int, Field(
        description='Maximum number of features visible for a given tile.'
    )] = 10000
    max_vertices: Annotated[Union[int | None], Field(
        description='Roughly maximum number of vertices polygons will be simplified to. If None then polygons are not simplified.'
    )] = None
    zarr_filepath: Annotated[Union[os.PathLike | None], Field(
        description="Filepath used to store zarr group representing the layer. If not provided a location will be created in systems temporary directory."
    )] = None
    layer_zarr: Annotated[Union[zarr.Group | None], Field(
        description="""
        Zarr group representing the layer. Is automatically generated if not provided.

        - attrs[type] (point or polygon)
        - ids (int)
        - geometries - the coordinates of features
        - - 0 (n_feats, 2) or (n_feats, 2, n_verts)
        - - 1 ... (only have zoom levels you need)
        - indices - general feature display for each zooms
        - - 0 (y_chunks, x_chunks, max_visible_features)
        - - 1 (y_chunks, x_chunks, max_visible_features)
        - - .... for every zoom
        """
    )] = None
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
    def set_resolution(self) -> Self:
        if self.resolution is None:
            self.resolution = self.image.resolution
        return self
    
    @model_validator(mode='after')
    def set_feature_ids(self) -> Self:
        if self.feature_ids is None:
            self.feature_ids = [f'feature_{i}' for i in range(len(self.coordinates))]
        
        assert len(self.feature_ids) == len(self.coordinates), f'Length of feature_ids and coordinates must be equal, got {len(self.feature_ids)} and {len(self.coordinates)}.'
        return self
    
    @model_validator(mode='after')
    def set_zarr_filepath(self) -> Self:
        if self.zarr_filepath is None:
            tmpfile = NamedTemporaryFile(delete=False)
            self.zarr_filepath = tmpfile.name + '.zarr.zip'
            tmpfile.close()
        path = Path(self.zarr_filepath)

        assert path.suffix == '.zip', f'zarr_filepath must have .zip extension, got {path}.'

        return self

    @model_validator(mode='after')
    def generate_zarr(self) -> Self:
        n_zooms = len(self.image.ngff_zarr[0])
        tile_size = self.image.ngff_tile_size
        scaler = self.image.resolution / self.resolution

        g = zarr.Group(self.zarr_filepath)
        g.attrs['type'] = self.geometry_type if isinstance(
            self.geometry_type, Enum) else self.geometry_type
        g['ids'] = np.arange(len(self.feature_ids))

        # create geometries
        print(f'Processing coordinates for {len(self.coordinates)} features.')
        if isinstance(self.coordinates, GeoSeries):
            coords = geoseries_to_coords(
                self.coordinates,
                n_target_verts=self.max_vertices
            )
            coords = da.asarray(coords, dtype=np.float32)
        elif isinstance(self.coordinates, ArrayLike):
            assert self.coordinates.shape[1] == 2, f'Second axis should be of size 2 (i.e. X, Y), got {self.coordinates.shape}'
            coords = da.asarray(self.coordinates)
        elif isinstance(self.coordinates, Iterable):
            assert len(self.coordinates[0]) == 2, f'Length of entries should be of size 2 (i.e. X, Y), got {len(self.coordinates[0])}'
            coords = da.asarray(self.coordinates)

        coords *= scaler
        assert coords.shape[1] == 2, f'Second axis should be of size 2 (i.e. X, Y), got {self.coordinates.shape}'
        g['geometries'][0] = coords.compute() # just pull all into memory for now, may need to make mem efficient later

        # do indices for each zoom level
        print('Calculating visible features for each zoom level.')
        df = dask.dataframe.from_pandas(
            pd.DataFrame(index=np.arange(len(coords))),
            npartitions=1
        )
        if len(coords.shape) == 3:
            df['x'], df['y'] = coords[:, 0, 0], coords[:, 1, 0]
        else:
            df['x'], df['y'] = coords[:, 0], coords[:, 1]
        for zoom in range(n_zooms):
            print(f'Starting zoom level {zoom}.')
            df[f'chunk_x_{zoom}'] = (df['x'] // tile_size).astype(int)
            df[f'chunk_y_{zoom}'] = (df['y'] // tile_size).astype(int)
            df[f'chunk_id_{zoom}'] = da.asarray(
                [f'{x}_{y}' for x, y in zip(df[f'chunk_x_{zoom}'], df[f'chunk_y_{zoom}'])]
            )

            chunks = da.unique(df[f'chunk_id_{zoom}'].values).compute()
            chunk_to_count = {chunk:0 for chunk in chunks}
            entity_idxs = []
            for chunk in df[f'chunk_id_{zoom}']:
                entity_idxs.append(chunk_to_count[chunk])
                chunk_to_count[chunk] += 1
            df[f'entity_idx_{zoom}'] = da.asarray(entity_idxs)

            # only to max count
            n_feats = self.max_visible_features
            df['keep'] = da.asarray(df[f'entity_idx_{zoom}'] < n_feats)
            df = df.query('keep')

            chunk_x_max, chunk_y_max = df[[f'chunk_x_{zoom}', f'chunk_x_{zoom}']].max().compute()

            idxs = df[[f'chunk_y_{zoom}', f'chunk_x_{zoom}', f'entity_idx_{zoom}']].values.compute()

            g['indices'][zoom] = zarr.full(
                (chunk_y_max + 1, chunk_x_max + 1, n_feats),
                -1,
                chunks=(1, 1, n_feats),
                dtype=np.int32
            )
            g['indices'][zoom][
                idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(df.shape[0].compute())
   
        self.layer_zarr = g

        return self


class Property(BaseModel, validate_assignment=True, arbitrary_types_allowed=True):
    """
    A property of layer features.
    """
    name: Annotated[str, Field(
        description='Name of property.'
    )]
    data: Annotated[Union[zarr.Array | NDArray | da.core.Array | pd.Series], Field(
        description='1D array of length num_features. Order must match order of features in Layer.'
    )]
    layer: Annotated[Layer, Field(
        description='Layer the properties apply to.'
    )]
    feature_ids: Annotated[Union[Iterable[Union[str | int]] | None], Field(
        description='Layer feature IDs properties correspond to. Must match order of rows in `data`.'
    )] = None
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
                raise ValueError(f'feature_ids was not provided and data is of type {type(self.data)}. Unless data is a pd.Series, feature_ids must be provided.')

        if len(set(self.feature_ids)) != len(self.feature_ids):
            raise ValueError('All feature ids must be unique.')

        if len(self.feature_ids) != self.data.shape[0]:
            raise ValueError(f'Length of `feature_ids` (got {len(self.feature_ids)}) must be equal to row dimension of `data` (got {self.data.shape[0]})')

        if not isinstance(self.feature_ids, list):
            self.feature_ids = list(self.feature_ids)

        return self
    
class PropertyGroup(Property, validate_assignment=True, arbitrary_types_allowed=True):
    """
    Group of related properties of the same data type. Must have same data type and be stored in matrix form.
    """
    data: Annotated[Union[ArrayLike | DataFrameLike], Field(
        description='Matrix or dataframe of shape (num_features, num_properties).'
    )]
    layer: Annotated[Layer, Field(
        description='Layer property group is annotating.'
    )]
    property_names: Annotated[Union[Iterable[str] | None], Field(
        description='Names of properties in property group. Are the column names in `data`. Must be provided if data is not a dataframe.'
    )] = None
    feature_ids: Annotated[Union[Iterable[str] | None], Field(
        description='Feature IDs (i.e. rows in data). If not provided, data will be assumed to be ordered exactly as in `Layer.feature_ids`'
    )]
    view_settings: Annotated[Dict[str, Union[PropertyViewSettings | None]], Field(
        description="View settings for a property group. A dictionary mapping property names (keys) to property view settings (values). If not provided, all properties will be displayed with default view settings for Layer. If a dictionary is provided, all properties specified in the dictionary will be displayed with the given view settings, the remaining properties will be visualized with default layer view settings."
    )] = None

    @model_validator(mode='after')
    def process_property_names(self) -> Self:
        if self.property_names is None:
            if isinstance(self.data, pd.DataFrame) or isinstance(self.data, dask.dataframe.DataFrame):
                self.property_names = list(self.data.columns)
            else:
                raise ValueError(f'property_names is not provided and data is of type {type(self.data)}. Unless data is a DataFrame, property_names must be provided.')

        if len(set(self.property_names)) != len(self.property_names):
            raise ValueError('All property names must be unique.')
        
        if len(self.property_names) != self.data.shape[1]:
            raise ValueError(f'Length of `property_names` (got {len(self.property_names)}) must be equal to column dimension of `data` (got {self.data.shape[1]})')

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