"""dolcestat: machine learning algorithms from scratch with NumPy, for learning."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("dolcestat")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0+unknown"
