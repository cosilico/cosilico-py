from enum import Enum

from pydantic import BaseModel

class DataType(str, Enum):
    multiplex = "multiplex"
    he = "he"
    xenium = "xenium"
    visium = "visium"

class CompressionType(str, Enum):
    lzw = "LZW"