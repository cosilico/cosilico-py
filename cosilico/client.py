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

        self._session = data.session
        self._user = data.user
    
    def create_experiment(): pass

    def 
    

  


