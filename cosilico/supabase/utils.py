from typing import Union

import typer
from rich import print
from supabase import Client


def check_logged_in(client: Union[Client | None]):
    if client is None:
        print("Must be logged in to use this command. \
Login with [bold green]cosilico login[/bold green]")
        raise typer.Exit(code=1)