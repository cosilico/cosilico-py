from pathlib import Path
from typing import Union, Dict, Any
import os

from rich import print
from supabase import create_client, Client
import toml
import typer

from cosilico.main import APP_NAME, DEFAULT_KEY
    
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

    def add_user(): pass
    def get_user(): pass
    def delete_user(): pass
    def reassign_project(): pass
    def reassign_collection(): pass
    def reassign_experiment(): pass

    def get_organization(): pass
    def list_organization_members(): pass
    
    def add_experiment(): pass
    def add_layer(): pass
    def add_image(): pass
    def add_property(): pass
    def get_experiment(): pass
    def get_layer(): pass
    def get_image(): pass
    def get_property(): pass
    def delete_experiment(): pass
    def delete_layer(): pass
    def delete_image(): pass
    def delete_property(): pass
    def list_experiments(): pass
    def list_layers(): pass
    def list_images(): pass
    def list_properties(): pass
    def move_experiment(): pass
    
    def add_group(): pass
    def add_group_member(): pass
    def get_group(): pass
    def list_group_members(): pass
    def delete_group(): pass
    def delete_group_member(): pass
    
    def add_project(): pass
    def add_collection(): pass
    def add_project_member(): pass
    def add_collection_member(): pass
    def get_project(): pass
    def get_collection(): pass
    def get_project_members(): pass
    def get_collection_members(): pass
    def delete_project(): pass
    def delete_collection(): pass
    def delete_project_member(): pass
    def delete_collection_member(): pass
    def list_projects(): pass
    def list_collections(): pass
    def move_collection(): pass

    

  


