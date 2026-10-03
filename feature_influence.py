"""
feature_influence.py
====================
How much does each group of features contribute to the forecast?

Features are split into three groups:
  Fundamentals   what physically sets the price: load and RES forecasts, net
                 load, gas and carbon prices, hydro reservoirs, forecast flows
  Price history  statistics of past prices: lags and rolling mean/max/min
  Calendar       hour, month, day of week, weekend

Setup: the general model of candidate B (the model behind hours 04-23 of the
final forecast), retrained at the start of each quarter of the test year
(July 2025 - June 2026) on the preceding 18 months, and scored on that
quarter. This is a lighter version of the daily retraining used for the main
results, so its MAE is somewhat higher.

Three measures, because each answers a different question:
  1. Built-in importance (XGBoost total gain): how much each group was used
     to fit the training data. Cheap but rough -- correlated features share
     credit arbitrarily.
  2. Permutation: scramble one group on the test data and measure the rise
     in MAE. How much THIS model relies on the group.
  3. Ablation: retrain without the group. How much information the group
     adds that the other groups cannot replace.

Usage:  python feature_influence.py      (~2-5 min)
Writes results/feature_influence.md
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

import run_candidates as rc

warnings.simplefilter("ignore")
ROOT = Path(__file__).resolve().parent

GROUPS = {
    "Fundamentals": ["load_forecast_mw", "res_forecast_mw", "net_load_proxy_mw", "net_load_total_mw",
                     "NGAS_Price", "carbon_price_eur", "hydro_reservoir_mwh", "hydro_reservoir_change",
                     "net_out_forecast"],
    "Price history": ["rolling_mean_1day", "rolling_mean_7days", "lag_24h", "lag_25h", "lag_48h",
                      "lag_168h", "rolling_max_1week", "rolling_min_1week"],
    "Calendar": ["hour", "month", "day_of_week", "is_weekend"],
}
QUARTERS = pd.date_range("2025-07-01", "2026-06-30", freq="QS-JUL")
TEST_END = pd.Timestamp("2026-06-30 23:00")
N_PERMUTATIONS = 5


def fit(train: pd.DataFrame, cols: list) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**rc.XGB_PARAMS)
    model.fit(train[cols], train[rc.TARGET])
    return model


def main() -> None:
    df, general, _ = rc.build_features(rc.load_data(rc.DEFAULT_DATA), "v2")
    assert sorted(sum(GROUPS.values(), [])) == sorted(general), "groups must cover the model's features"
    df = df.dropna(subset=general + [rc.TARGET])
    rng = np.random.default_rng(0)

    rows, gain = [], pd.Series(0.0, index=general)
    for q_start in QUARTERS:
        q_end = min(q_start + pd.DateOffset(months=3) - pd.Timedelta(hours=1), TEST_END)
        train = df[(df.timestamp >= q_start - pd.DateOffset(months=rc.MONTHS_USED)) & (df.timestamp < q_start)]
        test = df[(df.timestamp >= q_start) & (df.timestamp <= q_end)]
        y = test[rc.TARGET].to_numpy()
        err = lambda m, X: np.abs(m.predict(X) - y)

        full = fit(train, general)
        g = pd.Series(full.get_booster().get_score(importance_type="total_gain")).reindex(general).fillna(0)
        gain += g / g.sum()
        row = {"quarter": str(q_start.date()), "n": len(test), "full": err(full, test[general]).sum()}

        for name, cols in GROUPS.items():
            rises = []
            for _ in range(N_PERMUTATIONS):
                X = test[general].copy()
                idx = rng.permutation(len(X))
                for c in cols:  # scramble the group's rows jointly
                    X[c] = X[c].to_numpy()[idx]
                rises.append(err(full, X).sum())
            row[f"perm_{name}"] = np.mean(rises)
            kept = [c for c in general if c not in cols]
            row[f"drop_{name}"] = err(fit(train, kept), test[kept]).sum()

        for c in GROUPS["Fundamentals"]:
            X = test[general].copy()
            X[c] = X[c].to_numpy()[rng.permutation(len(X))]
            row[f"permfeat_{c}"] = err(full, X).sum()
        rows.append(row)
        print(f"  quarter from {q_start.date()} done", flush=True)

    r = pd.DataFrame(rows)
    n = r["n"].sum()
    base = r["full"].sum() / n
    gain /= len(QUARTERS)

    lines = [
        f"Test year July 2025 - June 2026, general model retrained quarterly. "
        f"MAE with all features: **{base:.2f} EUR/MWh**.",
        "",
        "| Group | Built-in importance (gain) | MAE rise when scrambled | MAE without the group (retrained) "
        "| Error reduction from adding the group |",
        "|---|---|---|---|---|",
    ]
    for name, cols in GROUPS.items():
        perm = r[f"perm_{name}"].sum() / n - base
        drop = r[f"drop_{name}"].sum() / n
        lines.append(f"| {name} | {gain[cols].sum():.0%} | +{perm:.2f} | {drop:.2f} ({drop - base:+.2f}) "
                     f"| {(drop - base) / drop:.0%} |")
    lines += ["", "Single fundamental features, MAE rise when scrambled:", ""]
    per_feat = {c: r[f"permfeat_{c}"].sum() / n - base for c in GROUPS["Fundamentals"]}
    for c, v in sorted(per_feat.items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{c}`: {v:+.2f}")

    out = ROOT / "results" / "feature_influence.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))
    print(f"\nSaved {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
