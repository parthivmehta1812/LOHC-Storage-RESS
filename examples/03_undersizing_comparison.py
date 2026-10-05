"""
Example 3 — With vs. without internal energy demand.

Quantifies how much each equipment item is underestimated when internal
process energy demand (pumps, burners) is ignored.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from lohc_cascade import (
    Params,
    hydrogenation_fractions,
    dehydrogenation_fractions,
    size_system,
)

p = Params()
fp_h, ff_h, _ = hydrogenation_fractions(p)
fp_d, ff_d, _ = dehydrogenation_fractions(p)

res_s, _, _ = size_system(p, (fp_h, ff_h, fp_d, ff_d))  # with internal demand
res_n, _, _ = size_system(p, (0, 0, 0, 0))               # demand neglected

comp = pd.DataFrame({
    "With internal demand":    res_s,
    "Internal demand neglected": res_n,
})
comp["Undersizing (%)"] = (comp.iloc[:, 0] / comp.iloc[:, 1] - 1) * 100

pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
print(comp.to_string())