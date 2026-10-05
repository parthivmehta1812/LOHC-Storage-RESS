"""
Example 2 — Heat-integration sensitivity.

Sweeps the share of dehydrogenation heat supplied by external waste heat
and shows how PV area, H₂ storage, and electrolyzer scale.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dataclasses import replace
from lohc_cascade import (
    Params,
    hydrogenation_fractions,
    dehydrogenation_fractions,
    size_system,
)

p = Params()
print(f"{'Waste-heat share':>20} {'PV area (m²)':>14} {'H₂ storage (kg)':>18} {'Electrolyzer (kW)':>20}")
print("-" * 76)

for share in [0.0, 0.25, 0.50, 0.75, 1.0]:
    ps = replace(p, ext_heat_share=share)
    fp_h, ff_h, _ = hydrogenation_fractions(ps)
    fp_d, ff_d, _ = dehydrogenation_fractions(ps)
    res, _, _     = size_system(ps, (fp_h, ff_h, fp_d, ff_d))
    print(f"  {share*100:>16.0f} %  {res['PV area (m2)']:>14,.0f} "
          f"{res['H2 storage (kg)']:>18,.1f} {res['Electrolyzer (kW)']:>20,.1f}")