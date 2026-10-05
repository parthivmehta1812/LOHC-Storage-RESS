"""
System parameters and hourly load profiles.

All thermodynamic constants, efficiencies, and irradiance/demand
profiles are centralised here so that sensitivity studies only need
to call ``dataclasses.replace(Params(), field=value)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Params:
    """Immutable parameter set for a PV–LOHC self-sufficient energy system.

    LOHC carrier: dibenzyltoluene (H0-DBT / H18-DBT).
    Default values correspond to the base case of Mah et al. (2021).
    """

    # ── LOHC (DBT) and reactions ──────────────────────────────────────────────
    L_cap: float = 0.062        # kg H₂ / kg loaded DBT (maximum loading capacity)
    eta_HGN: float = 0.90       # degree of loading (hydrogenation)
    eta_DHGN: float = 0.88      # degree of unloading (dehydrogenation)
    T_HGN: float = 150.0        # °C — hydrogenation temperature
    T_DHGN: float = 330.0       # °C — dehydrogenation temperature
    p_HGN: float = 30e5         # Pa — hydrogenation pressure
    p_DHGN: float = 5e5         # Pa — dehydrogenation pressure
    dH_r: float = 65.4          # kJ/mol H₂ (endothermic dehydrogenation reaction enthalpy)
    cp_UL: float = 1.56         # kJ/(kg·K) — specific heat, unloaded DBT
    cp_L: float = 1.55          # kJ/(kg·K) — specific heat, loaded DBT
    rho_UL: float = 1037.0      # kg/m³ — density, unloaded DBT
    rho_L: float = 907.0        # kg/m³ — density, loaded DBT

    # ── Hydrogen and system ───────────────────────────────────────────────────
    M_H2: float = 2.016         # g/mol — molar mass of H₂
    LHV_kJ: float = 120_000.0   # kJ/kg — lower heating value of H₂
    LHV_kWh: float = 33.3       # kWh/kg — lower heating value of H₂
    eta_CH: float = 0.97        # charging efficiency (storage filling)
    eta_DI: float = 0.99        # discharging efficiency (storage withdrawal)
    PV_eff: float = 0.15        # PV module efficiency
    INV_eff: float = 0.96       # inverter efficiency (DC → AC)
    Y_EL: float = 0.021         # kg H₂ / kWh — electrolyzer specific yield
    FC_eff: float = 0.50        # fuel cell electrical efficiency
    eta_burner: float = 0.80    # burner efficiency (H₂ → heat)
    eta_pump: float = 0.80      # pump efficiency

    # ── Ambient conditions ────────────────────────────────────────────────────
    T_amb: float = 30.0         # °C — ambient temperature
    p_amb: float = 1.013e5      # Pa — ambient pressure

    # ── Heat integration ──────────────────────────────────────────────────────
    ext_heat_share: float = 0.0  # fraction of dehydrogenation heat supplied by external waste heat


# ── Hourly load profiles (01:00 … 24:00) ─────────────────────────────────────
# German summer-day irradiance and community electricity demand
# Basis: 820 households, total demand 45.1 MWh/day (Mah et al. 2021)

SR = np.array([
    0, 0, 0, 0, 0,
    0.049, 0.126, 0.125, 0.177, 0.253, 0.309, 0.434,
    0.491, 0.610, 0.470, 0.430, 0.484, 0.231, 0.207,
    0.075, 0.002, 0, 0, 0,
])  # kW/m²

DEMAND = np.array([
    1230, 984, 656, 410, 164,
    3116, 5248, 4100, 1230, 1312, 1230, 1148,
    246, 574, 574, 574, 1640, 1558, 2132,
    5166, 6396, 2460, 1722, 1230,
])  # kWh

HOURS = np.arange(1, 25)
TOL = 0.05  # % convergence tolerance for iterative sizing