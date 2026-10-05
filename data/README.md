# Data

This directory is reserved for user-supplied input data.

## Hourly load profiles

The default irradiance and electricity demand profiles used in the base
case (820 households, German summer day, 45.1 MWh/day) are bundled
directly in `src/lohc_cascade/params.py` as `SR` and `DEMAND` arrays.

To run the model with custom profiles, pass your own `np.ndarray` arrays
to `cascade()` or modify the `SR` / `DEMAND` constants in `params.py`.

## Reference

Mah, A.X.Y. et al., "Cascade (pinch) analysis for self-sufficient
PV–LOHC energy systems", *Energy* **218** (2021) 119475.