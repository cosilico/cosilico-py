"""
Convienence methods related to GeoPandas
"""
from typing import List, Union

from geopandas.geodataframe import GeoSeries
from numpy.typing import NDArray
import numpy as np

def geoseries_to_coords(
        geoms: GeoSeries,
        n_target_verts: Union[int | None]=None,
        scaler: Union[float | None]=None
    ) -> List[NDArray]:
    # simplify method too slow for large number of geometries
    # simplified = [g.boundary.simplify(g.length * .05) for g in test]

    coords = []
    for g in geoms:
        stacked = np.stack(g.boundary.coords.xy)
        if n_target_verts is None:
            stacked = stacked[[1, 0]]      
        else:
            n_verts = stacked.shape[1]
            step_size = n_verts // n_target_verts
            stacked = stacked[[1, 0], slice(0, n_verts, step_size)]

        if scaler is not None:
            stacked *= scaler

        coords.append(stacked)

    max_verts = np.max([coord.shape[1] for coord in coords])

    for i, coord in enumerate(coords):
        coords[i] = np.pad(coord, [(0, 0), (0, max_verts - coord.shape[1])], constant_values=(-1,))

    return coords