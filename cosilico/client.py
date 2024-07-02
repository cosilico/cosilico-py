from pathlib import Path
from typing import Union, Dict, Any
import os

from rich import print
from supabase import create_client, Client
import toml
import typer

from cosilico.main import APP_NAME, DEFAULT_KEY
from cosilico.data.types import Experiment, MultiplexImage, Layer, Property, PropertyGroup
    
def load_config() -> Union[Dict | None]:
    app_dir = typer.get_app_dir(APP_NAME)
    # check if config exists
    config_path = Path(app_dir) / "cosilico.config"
    config = {}
    if not config_path.is_file():
        return None

    with config_path.open() as f:
        config = toml.load(f)
    
    return config
    
def read_from_config(keys, organization) -> Dict[str, Any]:
    config = load_config()

    if organization not in config:
        return {}
    return {k:config[organization][k] for k in keys}

class CosilicoClient(object):
    def __init__(
        self,
        email: str = '',
        password: str = '',
        host: str = '',
        api_key: str = '',
        organization: str = DEFAULT_KEY
        ):
        keys = ['email', 'password', 'host', 'api_key']
        cfg = read_from_config(keys, organization)
        email, password, host, api_key = [
            cfg.get(k, val) if not val else val
            for k, val in zip(keys, [email, password, host, api_key])
        ]

        for k, val in zip(keys, [email, password, host, api_key]):
            if not val:
                raise ValueError(f'{k} was not specified and was not retrieved from config due to {organization} not in config.')

        # create client
        self._supabase = create_client(host, api_key)

        # login
        data = self._supabase.auth.sign_in_with_password({
                "email": email,
                "password": password
        })

        self._session = data.sessions
        self._user = data.user

    # admin methods
    def create_user(self):
        pass
    def add_user(self):
        pass
    def get_user(self):
        pass
    def delete_user(self):
        pass
    def reassign_project(self):
        pass
    def reassign_collection(self):
        pass
    def reassign_experiment(self):
        pass



    # user methods
    def get_organization(self):
        pass
    def list_organization_members(self):
        pass
    
    def create_experiment(self):
        pass
    def create_layer(self):
        pass
    def create_image(
            # scaling_method: Annotated[ScalingMethod, Field(
            #     description="How to scale data if data type conversion is required. Only applicable if `data_type` is different from `data` data type."
            # )] = ScalingMethod.min_max
            # force_scale: Annotated[bool, Field(
            #     description='Force data to scale based on scaling_method, even if data_type and data.dtype match.'
            # )] = False
            # microns_per_pixel: Annotated[Union[float | None], Field(
            #     description="Resolution of image in microns per pixel. If not defined, will be automatically calculated from `resolution` and `resolution_unit`."
            # )] = None
        ):
        # user_defined = all(
        #     self.channels is not None,
        #     self.data is not None,
        #     self.resolution is not None,
        #     self.resolution_unit is not None
        # )
        # if self.source_filepath is None and not user_defined:
        #     raise ValueError(f'If source image filepath is not provided, then channels, data, resolution and resolution_unit must be specified.')

        # if self.source_filepath is not None:
        #     ext = self.source_filepath.suffix
        #     # check ome zarr
        #     if ext == '.zarr' and 


        # if user_defined:
        #     pass
        # else:
    #         @model_validator(mode='after')
    # def calculate_microns_per_pixel(self) -> Self:
    #     if self.microns_per_pixel is None:
    #         self.microns_per_pixel = to_microns_per_pixel(self.resolution, self.resolution_unit)
    #     return self
        
            

        # return self
    def create_property(self):
        pass
    def add_experiment(self):
        pass
    def add_layer(self):
        pass
    def add_image(self):
        pass
    def add_property(self):
        pass
    def get_experiment(self):
        pass
    def get_layer(self):
        pass
    def get_image(self):
        pass
    def get_property(self):
        pass
    def delete_experiment(self):
        pass
    def delete_layer(self):
        pass
    def delete_image(self):
        pass
    def delete_property(self):
        pass
    def list_experiments(self):
        pass
    def list_layers(self):
        pass
    def list_images(self):
        pass
    def list_properties(self):
        pass
    def move_experiment(self):
        pass
    
    def add_group(self):
        pass
    def add_group_member(self):
        pass
    def get_group(self):
        pass
    def list_group_members(self):
        pass
    def delete_group(self):
        pass
    def delete_group_member(self):
        pass
    
    def add_project(self):
        pass
    def add_collection(self):
        pass
    def add_project_member(self):
        pass
    def add_collection_member(self):
        pass
    def get_project(self):
        pass
    def get_collection(self):
        pass
    def get_project_members(self):
        pass
    def get_collection_members(self):
        pass
    def delete_project(self):
        pass
    def delete_collection(self):
        pass
    def delete_project_member(self):
        pass
    def delete_collection_member(self):
        pass
    def list_projects(self):
        pass
    def list_collections(self):
        pass
    def move_collection(self):
        pass
