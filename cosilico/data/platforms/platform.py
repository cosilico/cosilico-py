from enum import Enum
from typing import Dict, Any

from pydantic import BaseModel, Field
from typing_extensions import Annotated

class PlatformName(str, Enum):
    """
    All supported platforms
    """
    xenium = 'xenium'


class Platform(BaseModel):
    """
    Platform metadata.
    """
    name: Annotated[PlatformName, Field(
        description='Platform name.'
    )]
    provider: Annotated[str, Field(
        description='Company/provider of platform.'
    )]
    platform_version: Annotated[str, Field(
        description='Platform version.'
    )]
    software_version: Annotated[str, Field(
        description='Processing software version.'
    )]
    metadata: Annotated[Dict[str, Any], Field(
        description='Any metadata related to the platform.'
    )]



