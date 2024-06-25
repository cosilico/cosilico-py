from pathlib import Path
from numbers import Number
from tempfile import TemporaryDirectory
from typing import Union
import os

from numpy.typing import NDArray, DTypeLike
from osgeo import gdal, osr
from rich.progress import track
from typing_extensions import Annotated, Doc
import dask.array as da
import numpy as np
import zarr

X1, X2 = -180., 180.
Y1, Y2 = 90., -90.

DTYPE_MAPPING = {
    'float32': gdal.GDT_Float32,
    'float64': gdal.GDT_Float64,
    'int8': gdal.GDT_Int8,
    'int16': gdal.GDT_Int16,
    'int32': gdal.GDT_Int32,
    'int64': gdal.GDT_Int64,
    'uint8': gdal.GDT_Byte,
    'uint16': gdal.GDT_UInt16,
    'uint32': gdal.GDT_UInt32,
    'uint64': gdal.GDT_UInt64,
}

class GeoTransform(object):
    def __init__(
            self, w: Number, h: Number,
            x1: Number = X1, x2: Number = X2,
            y1: Number = Y1, y2: Number = Y2    
        ):
        self.h, self.w = h, w
        self.x_res = abs(x1 - x2) / w
        self.y_res = abs(y1 - y2) / h

    def latlon_to_pixel(self, lat: Number, lon: Number):
        x = (lat - X1) / self.x_res
        y = (lon - Y1) / self.y_res
        return x, y

    def pixel_to_latlon(self, x: Number, y: Number):
        lat = X1 + x * self.x_res
        lon = Y1 + y * self.y_res
        return lat, lon

def to_gdal_dtype(
    dtype: DTypeLike,
    ) -> int:
    name = np.dtype(dtype).name
    if name not in DTYPE_MAPPING:
        raise ValueError(f'data type {name} cannot be converted to GDAL data type.')
    
    return DTYPE_MAPPING[name]

def write_geotiff(
    data: Annotated[Union[NDArray, zarr.Array, da.core.Array], Doc(
        'Pixel data. Must be of shape (n_channels, height, width)'
    )],
    output_filepath: Annotated[Union[os.PathLike | str], Doc(
        'Where to write GeoTIFF'
    )]
    ):
    assert len(data.shape) == 3, f'data has {len(data.shape)} axes, expected shape of (n_channels, height, width)'

    output_filepath = Path(output_filepath)
    
    h, w = data.shape[-2:]

    x_res = abs(X1 - X2) / w
    y_res = abs(Y1 - Y2) / h

    driver = gdal.GetDriverByName('GTiff')

    opts = {
        'COMPRESS': 'LZW',
        'BIGTIFF': 'YES'
    }
    dtype = to_gdal_dtype(data.dtype)
    ds = driver.Create(
        str(output_filepath),
        xsize=w,
        ysize=h,
        bands=data.shape[0],
        eType=dtype,
        options=opts
    )

    ds.SetGeoTransform((X1, x_res, 0, Y1, 0, y_res))

    for i in track(range(data.shape[0]), description='Writting channels...'):
        band = ds.GetRasterBand(i + 1)
        band.WriteArray(data[i])
        band.FlushCache()
    
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    ds.SetProjection(srs.ExportToWkt())

    return ds


def write_cog(
    data: Annotated[Union[NDArray, zarr.Array, da.core.Array], Doc(
        'Pixel data. Must be of shape (n_channels, height, width)'
    )],
    output_filepath: Annotated[Union[os.PathLike | str], Doc(
        'Where to write COG GeoTIFF'
    )] = None
    ):
    assert len(data.shape) == 3, f'data has {len(data.shape)} axes, expected shape of (n_channels, height, width)'

    with TemporaryDirectory() as directory:
        geotiff_fp = Path(directory) / 'geo.tif'

        geotiff_ds = write_geotiff(data, geotiff_fp)

        driver = gdal.GetDriverByName('COG')
        opts = {
            'COMPRESS': 'LZW',
            'BIGTIFF': 'YES'
        }
        ds = driver.CreateCopy(output_filepath, geotiff_ds, options=opts)

        # make sure we close datasets
        geotiff_ds = None
        ds = None
    