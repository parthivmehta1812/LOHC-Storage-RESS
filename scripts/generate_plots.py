#!/usr/bin/env python3
"""
Generate all publication-quality figures for the PV–LOHC cascade analysis.

Runs the full model (base case + sensitivities), exports results to
``docs/images/`` and saves equipment sizing to ``equipment_sizing.csv``.

Usage
-----
    python scripts/generate_plots.py
"""

from __future__ import annotations

import os
import sys

# Allow running without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import replace

from lohc_cascade import (
    Params, hydrogenation_fractions, dehydrogenation_fractions, size_system,
    make_annual_demand, train_demand_forecaster, forecast_demand,
)

# ── Output directories ────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = os.path.join(ROOT, "docs", "images")
os.makedirs(IMG, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
NAVY   = "#0B2F5B"
TEAL   = "#1A8C8C"
ORANGE = "#E07A1F"
GREY   = "#8A94A6"
LIGHT  = "#D9E4F0"

# ── Global rcParams ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "DejaVu Sans",
    "font.size":           13,
    "axes.titlesize":      15,
    "axes.titleweight":    "bold",
    "axes.labelsize":      13,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.edgecolor":      "#333",
    "legend.frameon":      False,
    "legend.fontsize":     12,
    "savefig.dpi":         300,
    "figure.dpi":          100,
})

HOURS = np.arange(1, 25)
H0    = np.arange(0, 25)


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, name), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Daily supply–demand mismatch
# ─────────────────────────────────────────────────────────────────────────────
def plot_supply_demand(df_s: pd.DataFrame) -> None:
    pv = df_s.E_PV / 1000
    d  = df_s.E_D  / 1000
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.fill_between(HOURS, pv, d, where=pv > d, color=TEAL,   alpha=0.25,
                    label="Surplus → electrolysis + hydrogenation")
    ax.fill_between(HOURS, pv, d, where=pv <= d, color=ORANGE, alpha=0.25,
                    label="Deficit → dehydrogenation + fuel cell")
    ax.plot(HOURS, pv, color=TEAL,  lw=2.5, marker="o", ms=4, label="PV generation (DC)")
    ax.plot(HOURS, d,  color=NAVY,  lw=2.5, ls="--", marker="s", ms=4, label="Electricity demand")
    ax.set(xlabel="Hour of day", ylabel="Energy (MWh/h)",
           xticks=range(1, 25, 2), xlim=(1, 24))
    ax.set_title("Daily supply–demand mismatch  (45.1 MWh/day, 820 households)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, "01_supply_vs_demand.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Self-consumption split
# ─────────────────────────────────────────────────────────────────────────────
def plot_self_consumption(f: tuple) -> None:
    cats   = ["Hydrogenation\n(per kg H₂ produced)", "Dehydrogenation\n(per kg H₂ released)"]
    burned = np.array([f[1], f[3]]) * 100
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.barh(cats, 100 - burned, color=TEAL,   label="Useful H₂ (to reactor / fuel cell)")
    ax.barh(cats, burned,       left=100 - burned, color=ORANGE, label="Burned for process heat")
    for i, b in enumerate(burned):
        x  = (100 - b / 2) if b > 10 else (100 - b - 1.5)
        ax.text(x, i, f"{b:.1f} %",
                ha="center" if b > 10 else "right", va="center",
                color="white", fontweight="bold", fontsize=14)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of hydrogen (%)")
    ax.set_title("Internal heat demand: dehydrogenation dominates")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2)
    _save(fig, "02_self_consumption_split.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Hydrogen storage cascade
# ─────────────────────────────────────────────────────────────────────────────
def plot_storage_cascade(st_s: np.ndarray, st_n: np.ndarray) -> None:
    pinch = int(np.argmin(st_s))
    peak  = int(np.argmax(st_s))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(H0, st_s, color=NAVY, lw=2.8, marker="o", ms=4,
            label="With internal energy demand")
    ax.plot(H0, st_n, color=GREY, lw=2.2, ls="--", marker="s", ms=4,
            label="Internal demand neglected")
    ax.fill_between(H0, st_n, st_s, color=NAVY, alpha=0.10)
    ax.annotate("Pinch (storage empty)", (pinch, 0),
                xytext=(pinch - 5.5, 1000),
                arrowprops=dict(arrowstyle="->", color="#333"), fontsize=12)
    ax.annotate(f"Peak {st_s.max():,.0f} kg H₂", (peak, st_s.max()),
                xytext=(peak - 7.5, st_s.max() + 80),
                arrowprops=dict(arrowstyle="->", color="#333"), fontsize=12)
    ax.set(xlabel="Hour of day", ylabel="H₂ in storage (kg)",
           xticks=range(0, 25, 2), xlim=(0, 24))
    ax.set_ylim(0, st_s.max() * 1.18)
    ax.set_title("Hydrogen storage cascade (daily balance closed in both cases)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    _save(fig, "03_storage_cascade.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — Undersizing if internal demand neglected
# ─────────────────────────────────────────────────────────────────────────────
def plot_undersizing(comp: pd.DataFrame) -> None:
    keep   = ["PV area (m2)", "Electrolyzer (kW)",
              "Hydrogenation reactor (kg H2/h)", "Dehydrogenation reactor (kg H2/h)",
              "H2 storage (kg)", "DBT inventory (kg)",
              "Fuel cell (kW H2 input)", "Inverter (kW)"]
    labels = ["PV area", "Electro-\nlyzer",
              "Hydro-\ngenation\nreactor", "Dehydro-\ngenation\nreactor",
              "H₂\nstorage", "DBT\ninventory", "Fuel cell", "Inverter"]
    vals = comp.loc[keep, "Undersizing if neglected (%)"].values
    fig, ax = plt.subplots(figsize=(10, 4.8))
    bars = ax.bar(labels, vals,
                  color=[NAVY if v > 5 else GREY for v in vals], width=0.6)
    for b, v in zip(bars, vals):
        label = f"+{v:.0f} %" if v > 0.5 else "≈ 0 %"
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, label,
                ha="center", fontweight="bold", fontsize=12)
    ax.set_ylabel("Extra capacity needed (%)")
    ax.set_ylim(0, max(vals) * 1.18)
    ax.set_title("Capacity underestimated if internal energy demand is neglected")
    plt.setp(ax.get_xticklabels(), fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "04_undersizing.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Reactor load profile
# ─────────────────────────────────────────────────────────────────────────────
def plot_reactor_load(df_s: pd.DataFrame) -> None:
    hgn = df_s.H_HGN.clip(lower=0)
    dhg = (-df_s.H_DI).clip(lower=0)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(HOURS, hgn / hgn.max() * 100, color=TEAL,   width=0.8, label="Hydrogenation reactor")
    ax.bar(HOURS, dhg / dhg.max() * 100, color=ORANGE, width=0.8, label="Dehydrogenation reactor")
    ax.axhspan(0, 30, color=GREY, alpha=0.15, zorder=0)
    ax.text(24.6, 15,
            "Illustrative\nmin-load\nregion\n(< 30 %)",
            fontsize=10, color="#444", va="center")
    ax.set(xlabel="Hour of day", ylabel="Reactor load (% of design)",
           xticks=range(1, 25, 2), ylim=(0, 115))
    ax.set_title("Required reactor load profile: model assumes ideal flexibility")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "05_reactor_load_profile.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 6 — Energy chain waterfall (round-trip efficiency)
# ─────────────────────────────────────────────────────────────────────────────
def plot_energy_waterfall(p: Params, df_s: pd.DataFrame) -> None:
    E_el  = df_s.E_EL.sum() / 1000
    HG    = df_s.H_G.sum()
    H_HGN = df_s.H_HGN.sum()
    H_CH  = df_s.H_CH.sum()
    H_DI  = -df_s.H_DI.sum()
    H_DHGN = -df_s.H_DHGN.sum()
    H_FC  = -df_s.H_FC.sum()
    L     = p.LHV_kWh / 1000
    E_out = -df_s.E_FC.sum() / 1000

    steps = [
        ("Electricity to\nelectrolyzer",     E_el),
        ("Electrolysis\nlosses",              -(E_el - HG * L)),
        ("H₂ burned\n(HGN preheat)",          -(HG - H_HGN) * L),
        ("Charging\nlosses",                  -(H_HGN - H_CH) * L),
        ("Discharging\nlosses",               -(H_DI - H_DHGN) * L),
        ("H₂ burned\n(DHGN heat)",            -(H_DHGN - H_FC) * L),
        ("Fuel cell\nlosses",                 -(H_FC * L - E_out)),
        ("Electricity\nfrom FC",              E_out),
    ]
    rt_eff = E_out / E_el * 100

    fig, ax = plt.subplots(figsize=(11, 5))
    run = 0.0
    for i, (lab, v) in enumerate(steps):
        if i == 0:
            ax.bar(i, v, color=NAVY)
            ax.text(i, v + 2, f"{v:.0f}", ha="center", fontweight="bold")
            run = v
        elif i == len(steps) - 1:
            ax.bar(i, v, color=NAVY)
            ax.text(i, v + 2, f"{v:.0f}", ha="center", fontweight="bold")
        else:
            color = ORANGE if "DHGN" in lab else GREY
            ax.bar(i, v, bottom=run, color=color)
            ax.text(i, run + 2, f"{v:.0f}", ha="center", fontsize=11)
            run += v

    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels([s[0] for s in steps], fontsize=10.5)
    ax.set_ylabel("Energy (MWh/day)")
    ax.set_ylim(0, E_el * 1.15)
    ax.set_title(f"Power-to-LOHC-to-power chain: round-trip efficiency ≈ {rt_eff:.0f} %")
    ax.grid(axis="y", alpha=0.3)
    _save(fig, "06_energy_chain_waterfall.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 7 — Heat-integration sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def plot_heat_integration(sens_Q: list) -> None:
    s = np.array([x[0] for x in sens_Q]) * 100
    A = np.array([x[2] for x in sens_Q])
    S = np.array([x[3] for x in sens_Q])
    E = np.array([x[4] for x in sens_Q])
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(s, (A / A[0] - 1) * 100, color=NAVY,   lw=2.6, marker="o", label="PV area")
    ax.plot(s, (E / E[0] - 1) * 100, color=TEAL,   lw=2.6, marker="s", label="Electrolyzer")
    ax.plot(s, (S / S[0] - 1) * 100, color=ORANGE, lw=2.6, marker="^", label="H₂ storage / DBT")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set(xlabel="Share of dehydrogenation heat supplied by waste heat (%)",
           ylabel="Change vs. base case (%)")
    ax.set_title("Heat integration of dehydrogenation is the key design lever")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    _save(fig, "07_heat_integration_lever.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 8 — 24h-ahead demand forecasting with uncertainty bands
# ─────────────────────────────────────────────────────────────────────────────
def plot_demand_forecast(
    annual_demand: np.ndarray,
    fc_point: np.ndarray,
    fc_lower: np.ndarray,
    fc_upper: np.ndarray,
    test_start: int,
    mae: float,
    rmse: float,
    cov90: float,
) -> None:
    actual_test = annual_demand[test_start:]
    n_test      = len(actual_test)
    hours_all   = np.arange(n_test)
    WIN         = 14 * 24
    hours_win   = np.arange(WIN)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    # Top: 2-week detail window
    ax = axes[0]
    ax.fill_between(hours_win,
                    fc_lower[:WIN] / 1000, fc_upper[:WIN] / 1000,
                    color=TEAL, alpha=0.20, label="80% prediction interval")
    ax.plot(hours_win, fc_point[:WIN] / 1000, color=TEAL, lw=2,
            label="Forecast (p50)")
    ax.plot(hours_win, actual_test[:WIN] / 1000, color=NAVY, lw=1.5,
            ls="--", label="Actual demand")
    ax.set_xlim(0, WIN)
    ax.set_xlabel("Hour of forecast window")
    ax.set_ylabel("Electricity demand (MWh/h)")
    ax.set_title(
        f"24h-ahead electricity demand forecast — 2-week window  "
        f"(MAE = {mae/1000:.1f} MWh,  RMSE = {rmse/1000:.1f} MWh)"
    )
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Bottom: full test-period overview
    ax2 = axes[1]
    ax2.fill_between(hours_all,
                     fc_lower / 1000, fc_upper / 1000,
                     color=TEAL, alpha=0.15, label="80% prediction interval")
    ax2.plot(hours_all, actual_test / 1000, color=NAVY, lw=0.8,
             alpha=0.7, label="Actual demand")
    ax2.plot(hours_all, fc_point / 1000, color=TEAL, lw=1.0,
             label="Forecast (p50)")
    ax2.set_xlim(0, n_test)
    ax2.set_xlabel("Hour of test period (last 3 months)")
    ax2.set_ylabel("Electricity demand (MWh/h)")
    ax2.set_title(
        f"Full test period — p90 empirical coverage {cov90*100:.1f}%  (target 90%)"
    )
    ax2.legend(loc="upper right")
    ax2.grid(alpha=0.3)

    _save(fig, "08_demand_forecast.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Running cascade analysis …")
    p = Params()

    # Self-consumption fractions
    fp_h, ff_h, it_h = hydrogenation_fractions(p)
    fp_d, ff_d, heat = dehydrogenation_fractions(p)
    f_self = (fp_h, ff_h, fp_d, ff_d)
    print(f"  HGN  : f_pump = {fp_h:.5f},  f_fuel = {ff_h:.4f}  ({it_h} iterations)")
    print(f"  DHGN : f_pump = {fp_d:.5f},  f_fuel = {ff_d:.4f}")

    # Base case sizing
    print("\nSizing with internal energy demand …")
    res_s, df_s, st_s = size_system(p, f_self, verbose=True)
    print("Sizing without internal energy demand …")
    res_n, df_n, st_n = size_system(p, (0, 0, 0, 0), verbose=True)

    # Comparison table
    comp = pd.DataFrame({"With internal demand": res_s, "Internal demand neglected": res_n})
    comp["Undersizing if neglected (%)"] = (comp.iloc[:, 0] / comp.iloc[:, 1] - 1) * 100
    pd.set_option("display.float_format", lambda x: f"{x:,.1f}")
    print("\nEquipment sizing\n", comp)

    # KPIs
    E_el = df_s.E_EL.sum()
    E_fc = -df_s.E_FC.sum()
    hgn  = df_s.H_HGN[df_s.H_HGN > 0]
    dhg  = -df_s.H_DI[df_s.H_DI < 0]
    kpi  = {
        "Round-trip efficiency (FC out / EL in)": E_fc / E_el,
        "HGN operating hours":                    len(hgn),
        "HGN min load (% of max)":                hgn.min() / hgn.max() * 100,
        "DHGN operating hours":                   len(dhg),
        "DHGN min load (% of max)":               dhg.min() / dhg.max() * 100,
    }
    print("\nKPIs")
    for k, v in kpi.items():
        print(f"  {k}: {v:.3g}")

    # Temperature sensitivity
    sens_T = []
    for T in [290, 310, 330, 350]:
        pt = replace(p, T_DHGN=T)
        fp_h_, ff_h_, _ = hydrogenation_fractions(pt)
        fp_d_, ff_d_, _ = dehydrogenation_fractions(pt)
        r, _, _ = size_system(pt, (fp_h_, ff_h_, fp_d_, ff_d_))
        sens_T.append((T, ff_d_, r["PV area (m2)"], r["H2 storage (kg)"]))

    # Heat-integration sensitivity
    sens_Q = []
    for s in [0, 0.25, 0.5, 0.75, 1.0]:
        ps = replace(p, ext_heat_share=s)
        fp_h_, ff_h_, _ = hydrogenation_fractions(ps)
        fp_d_, ff_d_, _ = dehydrogenation_fractions(ps)
        r, d, _ = size_system(ps, (fp_h_, ff_h_, fp_d_, ff_d_))
        sens_Q.append((s, ff_d_, r["PV area (m2)"], r["H2 storage (kg)"],
                       r["Electrolyzer (kW)"], -d.E_FC.sum() / d.E_EL.sum()))

    # Export CSVs
    comp.to_csv(os.path.join(ROOT, "equipment_sizing.csv"))
    df_s.to_csv(os.path.join(ROOT, "cascade_with_internal_demand.csv"), index=False)
    df_n.to_csv(os.path.join(ROOT, "cascade_internal_demand_neglected.csv"), index=False)
    print("\nExported equipment_sizing.csv and cascade CSVs.")

    # Generate figures
    print("\nGenerating figures …")
    plot_supply_demand(df_s)
    plot_self_consumption(f_self)
    plot_storage_cascade(st_s, st_n)
    plot_undersizing(comp)
    plot_reactor_load(df_s)
    plot_energy_waterfall(p, df_s)
    plot_heat_integration(sens_Q)

    # ── Figure 8: demand forecasting ─────────────────────────────────────────
    print("\nGenerating annual demand profile and training demand forecaster …")
    annual_demand = make_annual_demand(n_days=365, seed=42)

    TRAIN_END  = int(365 * 24 * 0.75)   # first 9 months (~6,570 h)
    TEST_START = TRAIN_END
    TEST_END   = len(annual_demand)

    fc_models = train_demand_forecaster(annual_demand, train_end_h=TRAIN_END)
    fc_point, fc_lower, fc_upper = forecast_demand(
        fc_models, annual_demand, TEST_START, TEST_END
    )

    actual_test = annual_demand[TEST_START:TEST_END]
    mae   = float(np.mean(np.abs(fc_point - actual_test)))
    rmse  = float(np.sqrt(np.mean((fc_point - actual_test) ** 2)))
    cov90 = float(np.mean(actual_test <= fc_upper))
    print(f"[demand_forecaster] MAE={mae:.0f} kWh  RMSE={rmse:.0f} kWh  p90-coverage={cov90*100:.1f}%")

    plot_demand_forecast(annual_demand, fc_point, fc_lower, fc_upper,
                         TEST_START, mae, rmse, cov90)

    print(f"\nAll 8 figures saved to {IMG}/")


if __name__ == "__main__":
    main()