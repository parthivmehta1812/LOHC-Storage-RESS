"""
Cascade (pinch) analysis and iterative PV-area sizing.

``cascade`` computes the hourly hydrogen storage trajectory for a given
PV area and self-consumption fractions.  ``size_system`` iterates on the
PV area until the daily storage balance closes (circular day: storage at
hour 0 equals storage at hour 24) to within the convergence tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .params import Params, SR, DEMAND, HOURS, TOL


def cascade(
    A_pv: float,
    p: Params,
    f: tuple[float, float, float, float],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute hourly H₂ storage cascade for a given PV area.

    Parameters
    ----------
    A_pv:
        PV panel area [m²].
    p:
        System parameters.
    f:
        ``(fp_h, ff_h, fp_d, ff_d)`` — hydrogenation pump fraction,
        hydrogenation fuel fraction, dehydrogenation pump fraction,
        dehydrogenation fuel fraction.

    Returns
    -------
    df:
        Hour-by-hour energy and hydrogen flows (DataFrame).
    raw:
        Cumulative H₂ storage [kg] starting at 0 (length 25, hour 0 → 24).
    shifted:
        ``raw`` shifted so that the pinch (minimum) equals zero —
        the physically closed daily storage profile.
    """
    fp_h, ff_h, fp_d, ff_d = f
    df = pd.DataFrame({"hour": HOURS, "SR": SR, "E_D": DEMAND.astype(float)})
    df["E_PV"] = df.SR * A_pv * p.PV_eff
    df["E_EX"] = df.E_PV - df.E_D / p.INV_eff

    surplus = df.E_EX > 0

    # Surplus hours → electrolysis → hydrogenation
    df["E_EL"] = np.where(surplus, df.E_EX * (1 - fp_h), 0.0)
    df["H_G"] = df.E_EL * p.Y_EL
    df["H_HGN"] = df.H_G * (1 - ff_h)
    df["H_CH"] = df.H_HGN * p.eta_CH

    # Deficit hours → dehydrogenation → fuel cell (negative = withdrawal from storage)
    df["E_FC"] = np.where(~surplus, df.E_EX / (1 - fp_d), 0.0)
    df["H_FC"] = df.E_FC / (p.FC_eff * p.LHV_kWh)
    df["H_DHGN"] = df.H_FC / (1 - ff_d)
    df["H_DI"] = df.H_DHGN / p.eta_DI

    df["dH_ST"] = df.H_CH + df.H_DI

    raw = np.concatenate([[0.0], np.cumsum(df.dH_ST)])  # H_ST with H_ST(t₀) = 0
    shifted = raw - raw.min()                           # pinch: storage = 0 at most deficient hour
    return df, raw, shifted


def size_system(
    p: Params,
    f: tuple[float, float, float, float],
    A0: float = 200_000.0,
    verbose: bool = False,
) -> tuple[dict[str, float], pd.DataFrame, np.ndarray]:
    """Iteratively size the PV area until the daily storage balance closes.

    Uses a Newton-style update derived analytically from the linear
    dependence of H₂ storage balance on PV area.

    Parameters
    ----------
    p:
        System parameters.
    f:
        Self-consumption fractions ``(fp_h, ff_h, fp_d, ff_d)``.
    A0:
        Initial PV area guess [m²].
    verbose:
        Print iteration details if ``True``.

    Returns
    -------
    results:
        Equipment sizing dict with keys for PV area, electrolyzer,
        fuel cell, inverter, reactors, H₂ storage, and DBT inventory.
    df:
        Final cascade DataFrame (including ``H_ST`` column).
    shifted:
        Final shifted storage profile [kg H₂], length 25.
    """
    fp_h, ff_h, _, _ = f
    A = A0
    for it in range(1, 201):
        _, raw, _ = cascade(A, p, f)
        denom = SR.sum() * p.PV_eff * p.Y_EL * (1 - ff_h) * (1 - fp_h) * p.eta_CH
        A_new = A - (raw[-1] - raw[0]) / denom
        pct_change = abs(A_new - A) / A * 100
        if verbose:
            print(f"  it {it:2d}: A = {A:10.0f} m² → {A_new:10.0f} m²  ({pct_change:.3f} %)")
        A = A_new
        if pct_change < TOL:
            break

    df, raw, st = cascade(A, p, f)
    E_INV = df.E_PV - df.E_EL + df.E_FC.abs()
    ST_tot = st.max() / p.eta_DHGN

    results = {
        "PV area (m2)": A,
        "Electrolyzer (kW)": df.E_EL.max(),
        "Fuel cell (kW H2 input)": abs(df.E_FC.min()) / p.FC_eff,
        "Inverter (kW)": E_INV.max(),
        "Hydrogenation reactor (kg H2/h)": df.H_HGN.max() / p.eta_DHGN,
        "Dehydrogenation reactor (kg H2/h)": abs(df.H_DI.min()) / p.eta_DHGN,
        "H2 storage (kg)": ST_tot,
        "DBT inventory (kg)": ST_tot * (1 - p.L_cap) / (p.L_cap * p.eta_HGN),
    }
    df["H_ST"] = st[1:]
    return results, df, st