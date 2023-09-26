"""Provide a type annotation object for path like parameters.

Attributes:
    PathLike (str, or pathlib.Path): type annotation object for path like objects
"""
from typing import Union
import pathlib

PathLike = Union[str, pathlib.Path]
