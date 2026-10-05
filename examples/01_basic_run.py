"""
Example 1 — Basic cascade analysis (base case).

Runs the PV–LOHC cascade for the Mah et al. (2021) base case and
prints equipment sizes and key performance indicators.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from lohc_cascade import (
    Params,
    hydrogenation_fractions,
    dehydrogenation_fractions,
    size_system,
)

p = Params()

fp_h, ff_h, it_h = hydrogenation_fractions(p)
fp_d, ff_d, _    = dehydrogenation_fractions(p)
f_self = (fp_h, ff_h, fp_d, ff_d)

print("Self-consumption fractions")
print(f"  Hydrogenation  — pump: {fp_h:.4f}, H₂ burned: {ff_h:.4f}  ({it_h} iters)")
print(f"  Dehydrogenation — pump: {fp_d:.4f}, H₂ burned: {ff_d:.4f}")

results, df, _ = size_system(p, f_self, verbose=True)

print("\nEquipment sizing results")
for key, val in results.items():
    print(f"  {key:<45s}: {val:>12,.1f}")

E_el = df.E_EL.sum()
E_fc = -df.E_FC.sum()
print(f"\nRound-trip efficiency : {E_fc / E_el * 100:.1f} %")