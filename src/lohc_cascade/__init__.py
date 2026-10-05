"""
lohc-storage-ress
~~~~~~~~~~~~~~~~~
Cascade (pinch) analysis for self-sufficient PV–LOHC energy systems.

Based on the framework of Mah et al., Energy 218 (2021) 119475.
LOHC carrier: dibenzyltoluene (H0-DBT / H18-DBT).
"""

from .params import Params, SR, DEMAND
from .fractions import hydrogenation_fractions, dehydrogenation_fractions
from .cascade import cascade, size_system
from .demand_forecaster import make_annual_demand, train_demand_forecaster, forecast_demand

__version__ = "0.1.0"
__all__ = [
    "Params",
    "SR",
    "DEMAND",
    "hydrogenation_fractions",
    "dehydrogenation_fractions",
    "cascade",
    "size_system",
    "make_annual_demand",
    "train_demand_forecaster",
    "forecast_demand",
]