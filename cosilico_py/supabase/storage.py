from pathlib import Path
from typing import Union

from supabase import Client

from cosilico_py.data.types import DataType

def upload_data(
    client: Client, project: str, collection: str, filepath: Path,
    data_type: Union[DataType | None] = None, name: str =''
    ):
    pass