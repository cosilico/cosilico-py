from pathlib import Path
from typing import Optional, Dict, Union
import toml

import typer
import validators
from rich import print
from rich.console import Console
from rich.table import Table
from supabase import create_client, Client
from typing_extensions import Annotated

from cosilico_py.supabase.utils import check_logged_in
from cosilico_py.supabase.storage import upload_data
from cosilico_py.data.types import DataType, CompressionType

APP_NAME = "cosilico"
DEFAULT_KEY = 'default'

app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
app.add_typer(config_app, name="config")
data_app = typer.Typer(no_args_is_help=True)
app.add_typer(data_app, name="data")

app_dir = typer.get_app_dir(APP_NAME)
console = Console()
client: Union[Client | None] = None

@app.command()
def login(
    email: Annotated[str, typer.Option()] = '',
    password: Annotated[str, typer.Option()] = '',
    host: Annotated[str, typer.Option()] = '',
    api_key: Annotated[str, typer.Option()] = '',
    organization: Annotated[str, typer.Option()] = 'default'
    ):
    """
    Login to Cosilico
    """
    # check if config exists
    config_path: Path = Path(app_dir) / "cosilico.config"
    cfg = {}
    if not config_path.is_file():
        print("Config file doesn't exist, in the future create one with \
[bold red]skeleton-tool config[/bold red] for faster login.")
    else:
        with config_path.open() as f:
            cfg: Dict = toml.load(f)

    if email in cfg and not email:
        email = cfg[organization]['email']
    if organization in cfg and not password:
        password = cfg[organization]['password']
    if host in cfg and not host:
        host = cfg[organization]['host']
    if api_key in cfg and not api_key:
        api_key = cfg[organization]['api_key']

    if not email:
        email = typer.prompt('Enter email')
    if not password:
        password = typer.prompt('Enter password', hide_input=True)
    if not host:
        host = typer.prompt('Enter host')
        while not validators.url(host):
            typer.prompt(f'{host} is not a valid url. Please enter another')
    if not api_key:
        api_key = typer.prompt('Enter API key')

    # perform login
    client = create_client(host, api_key)

    print(f"Logged [bold green]{email}[/bold green] \
into [bold green]{host}[/bold green] :test_tube:")

@config_app.command('view')
def config_view(
    organization: Annotated[str, typer.Option()] = ''    
    ):
    """
    View cosilico.config contents
    """
    Path(app_dir).mkdir(parents=True, exist_ok=True)
    config_path: Path = Path(app_dir) / "cosilico.config"
    if config_path.is_file():
        with config_path.open() as f:
            cfg: Dict = toml.load(f)
    else:
        print('No cosilico.config file found. \
Use [bold green]cosilico config create[/bold green] to create one.')
        raise typer.Exit(1)
    
    if organization:
        if organization not in cfg:
            print(f'Organization {organization} not in cosilico.config.')
            raise typer.Exit(1)
        cfg = cfg[organization]

    print(cfg)

@config_app.command('create')
def config_create(
    organization: Annotated[str, typer.Option(
        prompt='Enter organization',
        help="this is a help message"
    )],
    host: Annotated[str, typer.Option(prompt='Enter host')],
    api_key: Annotated[str, typer.Option(prompt='Enter API key')],
    email: Annotated[str, typer.Option(prompt='Enter email')],
    password: Annotated[str, typer.Option(prompt='Enter password', hide_input=True)],
    make_default: Annotated[bool, typer.Option(prompt='Make default')]
    ):
    """
    Add entry to cosilico config.
    """
    Path(app_dir).mkdir(parents=True, exist_ok=True)
    config_path: Path = Path(app_dir) / "cosilico.config"
    cfg = {}
    if config_path.is_file():
        with config_path.open() as f:
            cfg: Dict = toml.load(f)
    
    if not validators.url(host):
        print(f'{host} is not a valid host URL')
        raise typer.Exit(code=1)
    
    if not validators.email(email):
        print(f'{email} is not a correctly formatted email')
        raise typer.Exit(code=1)
    
    entry = {
        'email': email,
        'password': password,
        'host': host,
        'api_key': api_key,
        'organization': organization,
    }
    cfg[organization] = entry

    if make_default:
        cfg[DEFAULT_KEY] = entry
    
    with config_path.open('w') as f:
        toml.dump(cfg, f)

    table = Table("organization", "host", "email")
    table.add_row(organization, host, email)

    print(f"Added [bold green]{organization}[/bold green] to config :fire:")
    console.print(table)

    print("Use [bold green]cosilico login[/bold green] to login.")

@data_app.command('upload')
def data_upload(
    project: Annotated[str, typer.Argument()],
    collection: Annotated[str, typer.Argument()],
    filepath: Annotated[Path, typer.Argument(exists=True)],
    data_type: Annotated[Union[DataType | None], typer.Option()] = None,
    name: Annotated[str, typer.Option()] = '',
    ):
    """
    Upload data to Cosilico.
    """
    check_logged_in(client)
    upload_data(
        client, project, collection, filepath,
        data_type=data_type, name=name
    )
    print(f"{filepath} uploaded to [bold green]{project}/{collection}[/bold green]")

@data_app.command('convert')
def data_convert(
    input_filepath: Annotated[Path, typer.Argument(exists=True)],
    output_filepath: Annotated[Path, typer.Argument(writable=True)],
    data_type: Annotated[Union[DataType, None], typer.Option()] = None,
    compression: Annotated[CompressionType, typer.Option()] = CompressionType.lzw,
    ):
    """
    Convert data
    """

@data_app.command('download')
def data_download(

    ):
    """
    Download data
    """

@app.command()
def load(
    name: Annotated[str, typer.Argument(help="name of blah")],
    foo: Annotated[Optional[int], typer.Argument(help="blah blah")] = None,
    bar: Annotated[int, typer.Option(help="foo foo bar")] = 1,
    formal: bool = False
):
    """
    Load the portal gun
    """
    print("Loading portal gun")