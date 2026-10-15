"""Register cleared light-sheet brains to a template and on to CCF."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("aind-lightsheet-registration")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"
