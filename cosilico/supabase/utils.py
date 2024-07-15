import os
import re
from pathlib import Path

from rich.progress import track
from supabase import Client

def listfiles(folder: os.PathLike, regex: str = None):
    """Return all files with the given regex in the given folder structure"""
    for root, folders, files in os.walk(folder):
        for filename in folders + files:
            if regex is None:
                yield os.path.join(root, filename)
            elif re.findall(regex, os.path.join(root, filename)):
                yield os.path.join(root, filename)


def upload_dir(
        supabase: Client,
        bucket: str,
        source_dir: os.PathLike,
        target_dir: os.PathLike
    ):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)

    bucket = supabase.storage.from_(bucket)

    filepaths = listfiles(source_dir)
    targets = [fp.replace(str(source_dir.absolute), str(target_dir.absolute)) for fp in filepaths]

    for filepath, target in track(zip(filepaths, targets), description='Uploading Directory...'):
        with open(filepath, 'rb') as f:
            bucket.upload(
                file=f,
                path=target,
                # file_options={"content-type": "audio/mpeg"}
            )