"""
Wrappers for bioformats image conversion

bfconvert documentation: https://bio-formats.readthedocs.io/en/stable/users/comlinetools/conversion.html
"""
from pathlib import Path
from typing import Union
import os
import re

from typing_extensions import Annotated, Doc

from cosilico.wrappers.external import realtime_subprocess

def check_in_out_paths(
        input_filepath: os.PathLike,
        output_filepath: os.PathLike
    ):
    if not input_filepath.exists():
        raise ValueError(f'Input file {input_filepath} does not exist')
    
    if not os.access(output_filepath.parent, os.W_OK):
        raise ValueError(f'Cannot write {output_filepath}. Directory is not writeable.')


def to_ngff(
        input_filepath:  Annotated[
            Union[os.PathLike | str],
            Doc("""
                Filepath to image to be converted.
                
                Must be a file format supported by bioformats v7.3.0 or additional format listed in bioformats2raw documentation.
                
                A list of supported file formats can be found [here](https://bio-formats.readthedocs.io/en/stable/supported-formats.html) and [here](https://github.com/glencoesoftware/bioformats2raw?tab=readme-ov-file#additional-readers).
                """
            )
        ],
        output_directory: Annotated[
            Union[os.PathLike | str | None],
            Doc("""
                Filepath to write zarr directory for OME-NGFF image.

                If not specified will default to filename prefix of input_filepath. Filename prefix will be all characters prior to first '.' in filename.
                """
            )
        ] = None,
        compression: Annotated[
            str,
            Doc("Compression method to use in NGFF.")
        ] = 'blosc',
        overwrite: Annotated[
            bool,
            Doc("Whether to overwrite output file if it already exists.")
        ] = True,
        resolutions: Annotated[
            Union[int | None],
            Doc("Number of pyramid resolutions to use in output file.")
        ] = None,
        dimension_order: Annotated[
            str,
            Doc("Dimension order. Must be valid order as specified by OME schema")
        ] = 'XYCZT'
    ):
    input_filepath, output_directory = Path(input_filepath), Path(output_directory)

    if output_directory is None:
        output_directory = input_filepath.parent / input_filepath.name.split('.')[0]
    
    output_directory.mkdir(exist_ok=True, parents=True)
    check_in_out_paths(input_filepath, output_directory)

    if not os.path.isdir(output_directory):
        raise ValueError(f'Output directory {output_directory} is not a directory.')

    opts = []
    if overwrite:
        opts += ['--overwrite']
    if resolutions is not None:
        opts += ['--resolutions', str(resolutions)]

    cmds = [
        'bioformats2raw',
        '--compression', compression,
        '--dimension-order', dimension_order,
        *opts,
        str(input_filepath),
        str(output_directory)
    ]

    realtime_subprocess(cmds)


def to_ome(
        input_filepath:  Annotated[
            Union[os.PathLike | str],
            Doc("""
                Filepath to image to be converted.
                
                Must be a file format supported by bioformats v7.3.0.
                
                A list of supported file formats can be found [here](https://bio-formats.readthedocs.io/en/stable/supported-formats.html).
                """
            )
        ],
        output_filepath: Annotated[
            Union[os.PathLike | str | None],
            Doc("""
                Filepath to write converted OME-TIF image.

                If not specified will default to filename prefix of input_filepath. Filename prefix will be all characters prior to first '.' in filename.
                """
            )
        ] = None,
        compression: Annotated[
            str,
            Doc("Compression method to use in output file.")
        ] = 'LZW',
        overwrite: Annotated[
            bool,
            Doc("Whether to overwrite output file if it already exists.")
        ] = True,
        pyramid_resolutions: Annotated[
            int,
            Doc("Number of pyramid resolutions to use in output file.")
        ] = 5,
        pyramid_scale: Annotated[
            int,
            Doc("Downsample factor to write pyramid resolutions at.")
        ] = 2,
        series: Annotated[
            Union[int | None],
            Doc("Number of Series to keep. By default all series are kept.")
        ] = None,
        timepoint: Annotated[
            Union[int | None],
            Doc("Number of timepoints to keep. By default all are kept.")
        ] = None,
    ):
    """
    Convert image to OME-TIF using bfconvert from bioformats.

    A list of supported file formats can be found [here](https://bio-formats.readthedocs.io/en/stable/supported-formats.html).
    """
    input_filepath, output_filepath = Path(input_filepath), Path(output_filepath)

    if output_filepath is None:
        output_filepath = input_filepath.parent / (input_filepath.name.split('.')[0] + '.ome.tiff')
    
    check_in_out_paths(input_filepath, output_filepath)
    
    if not re.findall(r'\.ome\.tiff?$', str(output_filepath)):
        raise ValueError(f'Extension of output filename must be ".ome.tiff" or ".ome.tif", got {output_filepath.name}')

    opts = []
    if series is not None:
        opts += ['-series', str(series)]
    if timepoint is not None:
        opts += ['-timepoint', str(timepoint)]

    cmds = [
        'bfconvert',
        '-compression', compression,
        '-overwrite' if overwrite else '-nooverwrite',
        '-pyramid-resolutions', str(pyramid_resolutions),
        '-pyramid-scale', str(pyramid_scale),
        *opts,
        str(input_filepath),
        str(output_filepath)
    ]

    realtime_subprocess(cmds)