"""
Unit tests for the PV–LOHC cascade analysis package.
"""

import numpy as np
import pytest
from lohc_cascade import (
    Params,
    hydrogenation_fractions,
    dehydrogenation_fractions,
    cascade,
    size_system,
)


class TestParams:
    def test_defaults(self):
        p = Params()
        assert p.L_cap == pytest.approx(0.062)
        assert p.PV_eff == pytest.approx(0.15)
        assert p.FC_eff == pytest.approx(0.50)

    def test_immutable(self):
        p = Params()
        with pytest.raises((TypeError, AttributeError)):
            p.L_cap = 0.1  # frozen dataclass


class TestHydrogenationFractions:
    def test_converges(self):
        p = Params()
        fp, ff, iters = hydrogenation_fractions(p)
        assert iters < 100, "Should converge within 100 iterations"

    def test_fractions_in_range(self):
        p = Params()
        fp, ff, _ = hydrogenation_fractions(p)
        assert 0 < fp < 1, f"Pump fraction {fp} out of (0, 1)"
        assert 0 < ff < 1, f"Fuel fraction {ff} out of (0, 1)"

    def test_basis_independence(self):
        p = Params()
        fp1, ff1, _ = hydrogenation_fractions(p, E_ex=100.0)
        fp2, ff2, _ = hydrogenation_fractions(p, E_ex=500.0)
        assert fp1 == pytest.approx(fp2, rel=1e-3)
        assert ff1 == pytest.approx(ff2, rel=1e-3)


class TestDehydrogenationFractions:
    def test_returns_tuple(self):
        p = Params()
        result = dehydrogenation_fractions(p)
        assert len(result) == 3

    def test_fractions_in_range(self):
        p = Params()
        fp, ff, _ = dehydrogenation_fractions(p)
        assert 0 < fp < 1
        assert 0 < ff < 1

    def test_heat_breakdown_keys(self):
        p = Params()
        _, _, heat = dehydrogenation_fractions(p)
        assert "Q_pre" in heat
        assert "Q_rxn" in heat

    def test_ext_heat_reduces_fuel(self):
        from dataclasses import replace
        p_base = Params()
        p_heat = replace(p_base, ext_heat_share=0.5)
        _, ff_base, _ = dehydrogenation_fractions(p_base)
        _, ff_heat, _ = dehydrogenation_fractions(p_heat)
        assert ff_heat < ff_base, "More external heat should reduce H₂ burned"


class TestCascade:
    def _fractions(self):
        p = Params()
        fp_h, ff_h, _ = hydrogenation_fractions(p)
        fp_d, ff_d, _ = dehydrogenation_fractions(p)
        return (fp_h, ff_h, fp_d, ff_d)

    def test_returns_dataframe_and_arrays(self):
        import pandas as pd
        p = Params()
        f = self._fractions()
        df, raw, shifted = cascade(200_000, p, f)
        assert isinstance(df, pd.DataFrame)
        assert len(raw) == 25
        assert len(shifted) == 25

    def test_shifted_min_is_zero(self):
        p = Params()
        f = self._fractions()
        _, _, shifted = cascade(200_000, p, f)
        assert shifted.min() == pytest.approx(0.0, abs=1e-9)

    def test_no_negative_storage(self):
        p = Params()
        f = self._fractions()
        _, _, shifted = cascade(218_000, p, f)
        assert (shifted >= -1e-6).all(), "Shifted storage should be non-negative"


class TestSizeSystem:
    def _fractions(self):
        p = Params()
        fp_h, ff_h, _ = hydrogenation_fractions(p)
        fp_d, ff_d, _ = dehydrogenation_fractions(p)
        return (fp_h, ff_h, fp_d, ff_d)

    def test_pv_area_positive(self):
        p = Params()
        f = self._fractions()
        res, _, _ = size_system(p, f)
        assert res["PV area (m2)"] > 0

    def test_pv_area_plausible(self):
        """Base case should converge near 218 000 m² as per Mah et al. (2021)."""
        p = Params()
        f = self._fractions()
        res, _, _ = size_system(p, f)
        assert 150_000 < res["PV area (m2)"] < 300_000

    def test_balance_closes(self):
        """Storage at hour 0 and hour 24 must be equal (within 0.05 %)."""
        p = Params()
        f = self._fractions()
        _, _, st = size_system(p, f)
        assert abs(st[0] - st[-1]) / max(st.max(), 1) * 100 < 0.1

    def test_all_sizing_keys_present(self):
        p = Params()
        f = self._fractions()
        res, _, _ = size_system(p, f)
        expected = {"PV area (m2)", "Electrolyzer (kW)", "Fuel cell (kW H2 input)",
                    "Inverter (kW)", "Hydrogenation reactor (kg H2/h)",
                    "Dehydrogenation reactor (kg H2/h)", "H2 storage (kg)", "DBT inventory (kg)"}
        assert expected.issubset(res.keys())

    def test_neglect_increases_area(self):
        """Neglecting internal demand should yield a SMALLER PV area (undersizing)."""
        p = Params()
        f = self._fractions()
        res_s, _, _ = size_system(p, f)
        res_n, _, _ = size_system(p, (0, 0, 0, 0))
        assert res_s["PV area (m2)"] > res_n["PV area (m2)"]
