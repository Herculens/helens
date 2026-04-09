# Copyright (c) 2023, herculens developers and contributors
# Copyright (c) 2024, helens developers and contributors

from importlib.metadata import version, PackageNotFoundError

from .solver import LensEquationSolver

try:
    __version__ = version("helens")
except PackageNotFoundError:
    __version__ = "unknown"
