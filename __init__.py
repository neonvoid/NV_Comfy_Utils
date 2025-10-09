

import os

__all__ = [
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    
]

__author__ = """elkkkk"""
__email__ = "you@gmail.com"
__version__ = "0.0.1"

from .src.KNF_Utils.nodes import NODE_CLASS_MAPPINGS
from .src.KNF_Utils.nodes import NODE_DISPLAY_NAME_MAPPINGS

# WEB_DIRECTORY is the directory that ComfyUI will link and auto-load for frontend extensions
WEB_DIRECTORY = "./web"


