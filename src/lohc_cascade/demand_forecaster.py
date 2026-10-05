"""
Electricity demand forecasting module.

Trains gradient-boosted regression trees (scikit-learn) with quantile
loss to produce 24-hour-ahead point forecasts and 80% prediction
intervals for community electricity demand.

This mirrors the real-world task of forecasting industrial electricity
consumption — the input to any BESS dispatch or storage sizing workflow.

Features
--------
- Cyclic calendar encoding: hour-of-day, day-of-week, month (sin/cos)
- Lagged demand: t-24, t-48, t-168 (same hour: yesterday, 2 days ago, last week)
- Rolling 24-hour mean and standard deviation

Three models are trained jointly:
    model_p50  — median forecast (squared-error loss)
    model_p10  — 10th-percentile lower bound (quantile loss)
    model_p90  — 90th-percentile upper bound (quantile loss)
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from .params import DEMAND, TOL


def make_annual_demand(
    n_days: int = 365,
    weekday_scale: float = 1.0,
    weekend_scale: float = 0.65,
    seasonal_amplitude: float = 0.12,
    noise_std: float = 0.04,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a synthetic annual electricity demand profile (8,760 hours).

    Extends the single-day base profile from ``params.DEMAND`` with:
    - Weekday / weekend variation
    - Sinusoidal seasonal variation (higher demand in winter, lower in summer)
    - Gaussian noise for realistic hour-to-hour variability

    Parameters
    ----------
    n_days:
        Number of days to generate (default 365).
    weekday_scale:
        Demand multiplier for weekdays (Mon–Fri).
    weekend_scale:
        Demand multiplier for weekends (Sat–Sun).
    seasonal_amplitude:
        Peak-to-trough seasonal swing as a fraction of mean demand.
    noise_std:
        Standard deviation of additive Gaussian noise (fraction of demand).
    seed:
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray of shape (n_days × 24,) with demand in kWh/h.
    """
    rng    = np.random.default_rng(seed)
    base   = DEMAND.astype(float)
    hours  = np.arange(n_days * 24)

    # Seasonal factor: peaks in winter (day 0/365), trough in summer (day 182)
    day_idx    = hours // 24
    seasonal   = 1.0 + seasonal_amplitude * np.cos(2 * np.pi * day_idx / 365)

    # Day-of-week factor
    dow_factor = np.where((day_idx % 7) < 5, weekday_scale, weekend_scale)

    # Tile the 24h base profile across all days
    profile = np.tile(base, n_days)

    # Apply scaling
    profile = profile * seasonal * dow_factor

    # Add noise
    noise   = rng.normal(0, noise_std * profile.mean(), size=len(profile))
    profile = np.clip(profile + noise, base.min() * 0.5, base.max() * 1.5)

    return profile


def _make_features(demand: np.ndarray, t: int) -> list[float]:
    """Build feature vector for hour *t* using only past information."""
    hod = t % 24
    dow = (t // 24) % 7
    mon = int((t / (365 * 24 / 12))) % 12      # approximate month index

    hod_sin = np.sin(2 * np.pi * hod / 24)
    hod_cos = np.cos(2 * np.pi * hod / 24)
    dow_sin = np.sin(2 * np.pi * dow / 7)
    dow_cos = np.cos(2 * np.pi * dow / 7)
    mon_sin = np.sin(2 * np.pi * mon / 12)
    mon_cos = np.cos(2 * np.pi * mon / 12)

    lag24  = float(demand[t - 24])  if t >= 24  else float(demand.mean())
    lag48  = float(demand[t - 48])  if t >= 48  else float(demand.mean())
    lag168 = float(demand[t - 168]) if t >= 168 else float(demand.mean())

    window    = demand[max(0, t - 24):t]
    roll_mean = float(window.mean()) if len(window) > 0 else float(demand.mean())
    roll_std  = float(window.std())  if len(window) > 1 else 0.0

    return [
        hod_sin, hod_cos,
        dow_sin, dow_cos,
        mon_sin, mon_cos,
        lag24, lag48, lag168,
        roll_mean, roll_std,
    ]


def train_demand_forecaster(
    demand: np.ndarray,
    train_end_h: int,
    n_estimators: int = 250,
) -> tuple:
    """
    Train point-forecast and quantile-regression GBM models on demand data.

    Parameters
    ----------
    demand:
        Full hourly demand array [kWh/h], shape (T,).
    train_end_h:
        Exclusive end hour for training (hours 168 … train_end_h are used).
    n_estimators:
        Number of boosting rounds per model.

    Returns
    -------
    (model_p50, model_p10, model_p90) — three fitted scikit-learn estimators.
    """
    X, y = [], []
    for t in range(168, train_end_h):
        X.append(_make_features(demand, t))
        y.append(float(demand[t]))

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    print(f"[demand_forecaster] Training on {len(y_arr):,} samples (hours 168–{train_end_h})")

    common = dict(
        n_estimators     = n_estimators,
        max_depth        = 5,
        learning_rate    = 0.05,
        subsample        = 0.8,
        min_samples_leaf = 10,
        random_state     = 42,
    )
    model_p50 = GradientBoostingRegressor(loss="squared_error", **common)
    model_p10 = GradientBoostingRegressor(loss="quantile", alpha=0.10, **common)
    model_p90 = GradientBoostingRegressor(loss="quantile", alpha=0.90, **common)

    model_p50.fit(X_arr, y_arr)
    model_p10.fit(X_arr, y_arr)
    model_p90.fit(X_arr, y_arr)

    print("[demand_forecaster] Training complete.")
    return model_p50, model_p10, model_p90


def forecast_demand(
    models: tuple,
    demand: np.ndarray,
    start_h: int,
    end_h: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate 24-hour-ahead demand forecasts for hours [start_h, end_h).

    Parameters
    ----------
    models:
        Output of ``train_demand_forecaster``.
    demand:
        Full demand array (used for lag features only — no look-ahead).
    start_h:
        First hour to forecast.
    end_h:
        Exclusive end hour.

    Returns
    -------
    (point, lower, upper) — p50, p10, p90 forecasts as np.ndarray [kWh/h].
    """
    model_p50, model_p10, model_p90 = models
    X = np.array(
        [_make_features(demand, t) for t in range(start_h, end_h)],
        dtype=float,
    )
    return (
        model_p50.predict(X),
        model_p10.predict(X),
        model_p90.predict(X),
    )