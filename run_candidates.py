"""
run_candidates.py
=================
Runs one candidate model over a rolling day-ahead evaluation window and saves
its hourly predictions to results/predictions_<candidate>.csv.

Candidates (the four model families behind the original final results):
  A  Original hybrid: general model + peak-hour zone model, "original" features.
  B  Hybrid v2: same routing, "v2" features (net load, skew/kurt in zone model).
  C  Hour-specific chain, original features, one model per hour (hours 0-5).
  D  Hour-specific chain, v2 features (hours 0-5). Hours 1-2 train on all hours,
     the rest on their own hour only -- the same setup that produced the
     original final results.

Leakage fixes compared with the earlier scripts:
  * net_out (REALIZED cross-border flow, published after the fact) is never a
    feature. v2's net load uses net_out_forecast instead.
  * outage_mw gets an extra 12h delay (24h -> 36h total), so no outage that
    starts after the ~12:00 D-1 forecast time is visible to any hour of day D.
  * Noise added to lag_1h / lag_2h in chain training is the RMSE of the
    previous hour's model measured ONLY on the selection year, not on the test
    year.

Usage:
  python run_candidates.py --candidate A
  python run_candidates.py --candidate C --data path/to/final_dataset.csv
Run C and D after nothing else -- they are self-contained (each chain runs its
own hours in order).
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.simplefilter("ignore")

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "processed" / "final_dataset.csv"
RESULTS_DIR = ROOT / "results"

EVAL_START = "2024-07-01"
EVAL_END = "2026-06-30"
SELECTION_END = "2025-06-30"   # noise std is estimated on [EVAL_START, SELECTION_END] only

MONTHS_USED = 18
MIN_TRAIN_HOURS = 1000
MIN_TRAIN_DAYS = 40
PEAK_HOURS = {10, 11, 12, 18, 19, 20}
CHAIN_HOURS = [0, 1, 2, 3, 4, 5]
D_ALL_HOURS_TRAINING = {1, 2}
NOISE_SEED = 42

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=0.5,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

TARGET = "mcp_eur_per_mwh"
SKEW_KURT_COLS = ["price_daily_skew", "price_daily_kurt", "price_weekly_skew", "price_weekly_kurt"]
EXCLUDE = {
    "original": {"timestamp", TARGET, "date", "net_out"},
    "v2": {"timestamp", TARGET, "date", "net_out", "outage_mw"},
}


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    # outage_mw already carries a 24h visibility lag from the dataset builder;
    # add 12h more so the total (36h) respects a ~12:00 D-1 forecast time.
    if "outage_mw" in df.columns:
        s = df.set_index("timestamp")["outage_mw"].shift(freq="12h")
        df["outage_mw"] = s.reindex(df["timestamp"]).values
    return df


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    p = df[TARGET]
    df["rolling_mean_1day"] = p.shift(24).rolling(24).mean()
    df["rolling_mean_7days"] = p.shift(24).rolling(24 * 7).mean()
    df["lag_24h"] = p.shift(24)
    df["lag_25h"] = p.shift(25)
    df["lag_48h"] = p.shift(48)
    df["lag_168h"] = p.shift(168)
    df["rolling_max_1week"] = p.shift(24).rolling(168).max()
    df["rolling_min_1week"] = p.shift(24).rolling(168).min()
    return df


def add_skew_kurt(df: pd.DataFrame) -> pd.DataFrame:
    """Daily stats on D-1 and weekly stats on D-7..D-1, broadcast to day D."""
    date = df["timestamp"].dt.normalize()
    daily = df.groupby(date)[TARGET]
    d_skew, d_kurt = daily.skew(), daily.apply(pd.Series.kurt)
    hourly = df.set_index("timestamp")[TARGET]
    w_skew, w_kurt = {}, {}
    for d in d_skew.index:
        w = hourly.loc[d - pd.Timedelta(days=7): d - pd.Timedelta(seconds=1)]
        w_skew[d] = w.skew() if len(w) > 1 else np.nan
        w_kurt[d] = w.kurt() if len(w) > 1 else np.nan
    stats = pd.DataFrame({
        "price_daily_skew": d_skew.shift(1),
        "price_daily_kurt": d_kurt.shift(1),
        "price_weekly_skew": pd.Series(w_skew),
        "price_weekly_kurt": pd.Series(w_kurt),
    })
    df["date"] = date
    df = df.join(stats, on="date")
    return df.drop(columns="date")


def build_features(raw: pd.DataFrame, feature_set: str) -> tuple[pd.DataFrame, list, list]:
    df = add_price_features(raw.copy())
    if feature_set == "v2":
        df["net_load_total_mw"] = df["load_forecast_mw"] + df["net_out_forecast"] - df["res_forecast_mw"]
        df = add_skew_kurt(df)
    general = [c for c in df.columns if c not in EXCLUDE[feature_set] and c not in SKEW_KURT_COLS]
    zone = general + (SKEW_KURT_COLS if feature_set == "v2" else [])
    return df, general, zone


def fit_predict(X_train, y_train, X_test) -> np.ndarray:
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    return model.predict(X_test)


# ---------------------------------------------------------------------------
# Candidates A / B: hybrid general + peak-zone model
# ---------------------------------------------------------------------------
def run_hybrid(raw: pd.DataFrame, feature_set: str, start: str, end: str) -> pd.DataFrame:
    df, general, zone = build_features(raw, feature_set)
    df = df.dropna(subset=zone + [TARGET]).reset_index(drop=True)
    train_hours = MONTHS_USED * 30 * 24
    days = pd.date_range(start, min(pd.Timestamp(end), df["timestamp"].max().normalize()), freq="D")
    records = []
    for i, day in enumerate(days, 1):
        test = df[(df["timestamp"] >= day) & (df["timestamp"] < day + pd.Timedelta(days=1))]
        train_end = day - pd.Timedelta(hours=1)
        train = df[(df["timestamp"] > train_end - pd.Timedelta(hours=train_hours)) & (df["timestamp"] <= train_end)]
        if len(test) < 24 or len(train) < MIN_TRAIN_HOURS:
            print(f"[{i}/{len(days)}] {day.date()} skipped (incomplete data)")
            continue
        pred = fit_predict(train[general], train[TARGET], test[general])
        in_zone = test["hour"].isin(PEAK_HOURS).values
        zone_train = train[train["hour"].isin(PEAK_HOURS)]
        if len(zone_train) >= 100 and in_zone.any():
            pred[in_zone] = fit_predict(zone_train[zone], zone_train[TARGET], test.loc[in_zone, zone])
        for ts, hr, y, p in zip(test["timestamp"], test["hour"], test[TARGET], pred):
            records.append((ts, hr, y, p))
        print(f"[{i}/{len(days)}] {day.date()} MAE={np.mean(np.abs(pred - test[TARGET].values)):.2f}")
    return pd.DataFrame(records, columns=["timestamp", "hour", "actual", "predicted"])


# ---------------------------------------------------------------------------
# Candidates C / D: hour-specific chain for the early hours
# ---------------------------------------------------------------------------
def run_chain(raw: pd.DataFrame, feature_set: str, all_hours_training: set, start: str, end: str) -> pd.DataFrame:
    base, general, _ = build_features(raw, feature_set)
    base["lag_1h"] = base[TARGET].shift(1)
    base["lag_2h"] = base[TARGET].shift(2)
    base = base.dropna(subset=general + ["lag_1h", "lag_2h", TARGET]).reset_index(drop=True)
    features = general + ["lag_1h", "lag_2h"]
    base["date"] = base["timestamp"].dt.normalize()

    days = pd.date_range(start, min(pd.Timestamp(end), base["timestamp"].max().normalize()), freq="D")
    preds: dict[int, pd.Series] = {}     # hour -> predictions indexed by date
    rmse_sel: dict[int, float] = {}      # hour -> RMSE on the selection year only
    out = []

    for h in CHAIN_HOURS:
        rng = np.random.default_rng(NOISE_SEED + h)
        df = base.copy()
        # lag_k refers to hour h-k of the same day when h-k >= 0: at forecast
        # time that value is itself a prediction, so training sees actual+noise.
        for k, col in ((1, "lag_1h"), (2, "lag_2h")):
            if h - k >= 0:
                df[col] = df[col] + rng.normal(0.0, rmse_sel[h - k], size=len(df))
        scope_all = h in all_hours_training
        pool = df if scope_all else df[df["hour"] == h]
        pool = pool.set_index("timestamp").sort_index()
        test_rows = df[df["hour"] == h].set_index("date")

        hour_preds = {}
        for i, day in enumerate(days, 1):
            if day not in test_rows.index:
                continue
            train = pool.loc[day - pd.DateOffset(months=MONTHS_USED): day - pd.Timedelta(seconds=1)]
            n_train = train.index.normalize().nunique()
            if n_train < MIN_TRAIN_DAYS or (scope_all and len(train) < MIN_TRAIN_HOURS):
                continue
            X_test = test_rows.loc[[day], features].copy()
            missing_upstream = False
            for k, col in ((1, "lag_1h"), (2, "lag_2h")):
                if h - k >= 0:
                    if day not in preds[h - k].index:
                        missing_upstream = True
                        break
                    X_test[col] = preds[h - k].loc[day]
            if missing_upstream:
                continue
            p = float(fit_predict(train[features], train[TARGET], X_test)[0])
            hour_preds[day] = p
            out.append((day + pd.Timedelta(hours=h), h, float(test_rows.loc[day, TARGET]), p))
            if i % 50 == 0:
                print(f"  hour {h:02d}: {i}/{len(days)} days")

        preds[h] = pd.Series(hour_preds)
        actual = test_rows[TARGET]
        sel = preds[h][(preds[h].index >= pd.Timestamp(start)) & (preds[h].index <= pd.Timestamp(SELECTION_END))]
        err = sel - actual.reindex(sel.index)
        rmse_sel[h] = float(np.sqrt((err ** 2).mean()))
        print(f"hour {h:02d} done: {len(preds[h])} days, selection-year RMSE={rmse_sel[h]:.2f}"
              f" (used as training noise for the next hours)")
    return pd.DataFrame(out, columns=["timestamp", "hour", "actual", "predicted"])


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, choices=["A", "B", "C", "D"])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--start", default=EVAL_START)
    ap.add_argument("--end", default=EVAL_END)
    args = ap.parse_args()

    raw = load_data(args.data)
    c = args.candidate
    if c == "A":
        result = run_hybrid(raw, "original", args.start, args.end)
    elif c == "B":
        result = run_hybrid(raw, "v2", args.start, args.end)
    elif c == "C":
        result = run_chain(raw, "original", set(), args.start, args.end)
    else:
        result = run_chain(raw, "v2", D_ALL_HOURS_TRAINING, args.start, args.end)

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"predictions_{c}.csv"
    result.to_csv(out_path, index=False)
    mae = (result["predicted"] - result["actual"]).abs().mean()
    print(f"\nCandidate {c}: {len(result)} hourly predictions, MAE over the whole run = {mae:.2f} EUR/MWh")
    print(f"Saved to {out_path}")
