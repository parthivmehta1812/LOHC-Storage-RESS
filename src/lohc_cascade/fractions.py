"""
Self-consumption fraction calculations.

Computes the fraction of hydrogen (or electricity) consumed internally
by process utilities — pumps and burners — during hydrogenation and
dehydrogenation.  These fractions couple back into the cascade model
and must be solved iteratively (hydrogenation) or directly
(dehydrogenation).
"""

from __future__ import annotations

from .params import Params, TOL


def hydrogenation_fractions(p: Params, E_ex: float = 100.0) -> tuple[float, float, int]:
    """Iteratively solve for hydrogenation self-consumption fractions.

    Parameters
    ----------
    p:
        System parameters.
    E_ex:
        Basis excess DC electricity [kWh] (arbitrary; fractions are
        basis-independent at convergence).

    Returns
    -------
    f_pump:
        Fraction of excess DC electricity consumed by the HGN pump.
    f_fuel:
        Fraction of produced H₂ burned for reactor preheating.
    iterations:
        Number of iterations until convergence.
    """
    f_pump, f_fuel = 0.5, 0.5
    for it in range(1, 101):
        HG = E_ex * (1 - f_pump) * p.Y_EL
        H_HGN = HG * (1 - f_fuel)
        L_UL = H_HGN * (1 - p.L_cap) / (p.L_cap * p.eta_HGN * p.eta_DHGN)
        Q = L_UL * p.cp_UL * (p.T_HGN - p.T_amb)
        H_fuel = Q / (p.LHV_kJ * p.eta_burner)
        L_UL_new = (HG - H_fuel) * (1 - p.L_cap) / (p.L_cap * p.eta_HGN * p.eta_DHGN)
        E_pump_DC = (
            L_UL_new * (p.p_HGN - p.p_amb) / (p.rho_UL * p.eta_pump * 1000 * 3600) / p.INV_eff
        )
        f_fuel_new = H_fuel / HG
        f_pump_new = E_pump_DC / E_ex
        converged = (
            abs(f_fuel_new - f_fuel) / f_fuel * 100 < TOL
            and abs(f_pump_new - f_pump) / f_pump * 100 < TOL
        )
        f_pump, f_fuel = f_pump_new, f_fuel_new
        if converged:
            break
    return f_pump, f_fuel, it


def dehydrogenation_fractions(
    p: Params, L_L: float = 12.0
) -> tuple[float, float, dict[str, float]]:
    """Compute dehydrogenation self-consumption fractions directly.

    Parameters
    ----------
    p:
        System parameters.
    L_L:
        Loaded DBT flow rate [kg/h]. Representative of a demonstrated
        LOHC pilot unit (~12 kg/h); fractions are flow-rate-independent.

    Returns
    -------
    f_pump:
        Fraction of fuel-cell electricity consumed by the DHGN pump.
    f_fuel:
        Fraction of released H₂ burned for preheating and reaction heat.
    heat_breakdown:
        Dict with keys ``Q_pre`` and ``Q_rxn`` [kJ/h] for diagnostics.
    """
    Q_pre = L_L * p.cp_L * (p.T_DHGN - p.T_amb)
    H_DI = L_L * p.L_cap * p.eta_HGN * p.eta_DHGN
    Q_rxn = H_DI * p.dH_r / p.M_H2 * 1000
    H_DHGN = H_DI * p.eta_DI
    H_fuel = (Q_pre + Q_rxn) * (1 - p.ext_heat_share) / (p.LHV_kJ * p.eta_burner)
    H_FC = H_DHGN - H_fuel
    E_FC = H_FC * p.FC_eff * p.LHV_kWh
    E_pump_DC = (
        L_L * (p.p_DHGN - p.p_amb) / (p.rho_L * p.eta_pump * 1000 * 3600) / p.INV_eff
    )
    return (
        E_pump_DC / E_FC,
        H_fuel / H_DHGN,
        {"Q_pre": Q_pre, "Q_rxn": Q_rxn},
    )