from . import core
from . import data

from .data import get

# from .core import year_frac, PerformanceStats, GroupStats, merge
from .core import *

core.extend_pandas()

__version__ = "1.0.1"
