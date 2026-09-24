"""
rolling_evaluation_hybrid_v2.py
=================================
Rolling day-ahead αξιολόγηση ολόκληρου έτους, HYBRID μοντέλο (ίδια βασική
αρχιτεκτονική routing με το rolling_evaluation_hybrid.py):

    - Ώρες μέσα στη ζώνη PEAK_HOURS: πρόβλεψη από zone-specific μοντέλο,
      εκπαιδευμένο ΜΟΝΟ σε δεδομένα αυτών των ωρών.
    - Όλες οι άλλες ώρες: πρόβλεψη από το γενικό μοντέλο, εκπαιδευμένο σε
      ΟΛΕΣ τις ώρες.

ΑΛΛΑΓΕΣ σε σχέση με το rolling_evaluation_hybrid.py:

  1. Αφαιρέθηκε το outage_mw από τα features (δοκιμάστηκε σε ξεχωριστό
     πείραμα -- δεν βοηθούσε ουσιαστικά).

  2. Προστέθηκαν price_daily_skew / price_daily_kurt / price_weekly_skew /
     price_weekly_kurt (ίδια σύμβαση με τα προηγούμενα πειράματα: υπολο-
     γισμός σε d-1 / d-7..d-1, broadcast στην ημέρα d, χωρίς leakage) --
     ΑΛΛΑ μόνο στο zone-specific μοντέλο (ώρες αιχμής). Το γενικό μοντέλο
     ΔΕΝ τα βλέπει καθόλου, όπως ζητήθηκε.

  3. Το net_out δεν μπαίνει πια ως ανεξάρτητο feature -- ενσωματώνεται στο
     συνολικό φορτίο: net_load_total_mw = load_forecast_mw + net_out -
     res_forecast_mw. Αυτό μπαίνει σε ΑΜΦΟΤΕΡΑ τα μοντέλα (γενικό + zone),
     αφού είναι βασικό χαρακτηριστικό φορτίου, όχι κάτι ζωνικό.

  4. Το reporting στο τέλος περιορίστηκε ΜΟΝΟ σε: MAE ανά μήνα, MAE ανά
     ώρα, και συνολικό MAE. Αφαιρέθηκαν το seasonal breakdown, το spike
     breakdown (εντός/εκτός ζώνης) και τα γραφήματα -- δεν ζητήθηκαν εδώ.

  5. ΝΕΟ: όλα τα features (raw + engineered) ΚΑΙ οι προβλέψεις του hybrid
     μοντέλου αποθηκεύονται μαζί σε ΕΝΑ dataset -- vol2_final.csv (βλ.
     VOL2_FINAL_PATH). Η στήλη predicted_hybrid_v2 είναι NaN για ώρες
     εκτός του eval window ή ημέρες που παραλείφθηκαν (ελλιπή δεδομένα).

ΣΗΜΑΝΤΙΚΟ -- vol2_final ΣΩΡΕΥΤΙΚΟ, όχι overwrite: το vol2_final.csv είναι
κοινό μεταξύ ΟΛΩΝ των rolling_evaluation_*.py scripts (αυτό, το
_hybrid.py original, το _single_no_kurt.py). Κάθε script γράφει τις
προβλέψεις του σε ΔΙΚΗ ΤΟΥ στήλη (εδώ: predicted_hybrid_v2) και
ενημερώνει/προσθέτει μόνο τις δικές του στήλες -- αν το αρχείο υπάρχει
ήδη από προηγούμενο run άλλου script, ΔΕΝ το σβήνει, μόνο το επεκτείνει.

Ό,τι ΔΕΝ άλλαξε: MONTHS_USED (18, παραμένει pending προς εξέταση),
PEAK_HOURS, XGB_PARAMS, MIN_TRAIN_HOURS, rolling_mean/lag/rolling_max-min
features του γενικού feature set.
"""

from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]  # repository root

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_PATH = Path(str(REPO / "Dataset_Creation" / "processed" / "final_dataset.csv"))

# vol2_final.csv -- ΟΛΑ τα features (raw + engineered) ΚΑΙ οι προβλέψεις
# του hybrid μοντέλου μαζεμένα σε ένα ενιαίο dataset. Ένα αρχείο-αναφορά
# για ό,τι χρησιμοποιήσαμε/φτιάξαμε, ώστε να μη χρειάζεται να ξανα-φτιάχνει
# κανείς τα features από την αρχή για να δει τι μπήκε στο μοντέλο.
VOL2_FINAL_PATH = DATASET_PATH.parent / "vol2_final.csv"

EVAL_YEAR = 2025          # ποιο ημερολογιακό έτος αξιολογούμε
EVAL_START = None         # π.χ. "2025-01-01" -- None = 1 Ιανουαρίου του EVAL_YEAR
EVAL_END = None           # π.χ. "2025-12-31" -- None = 31 Δεκεμβρίου του EVAL_YEAR
                          # (κόβεται αυτόματα στη μέγιστη διαθέσιμη ημερομηνία)

MONTHS_USED = 18           # μήνες rolling training window πριν από κάθε μέρα -- pending εξέταση
MIN_TRAIN_HOURS = 1000    # ελάχιστο πλήθος ωρών training, αλλιώς η μέρα παραλείπεται

PEAK_HOURS = {10, 11, 12, 18, 19, 20}   # ζώνη ωρών με zone-specific μοντέλο

QUANTILE_ALPHA = 0.5      # 0.5 = median (μαθηματικά MAE-optimal στόχος)

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILE_ALPHA,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

SKEW_KURT_COLS = [
    "price_daily_skew", "price_daily_kurt",
    "price_weekly_skew", "price_weekly_kurt",
]

# columns που δεν μπαίνουν ΠΟΤΕ ως feature σε κανένα από τα δύο μοντέλα
EXCLUDE_ALWAYS = {
    "timestamp", "mcp_eur_per_mwh", "date",
    "net_out",       # ενσωματωμένο στο net_load_total_mw αντί να μπαίνει raw
    "outage_mw",     # δεν βοηθούσε -- αφαιρέθηκε
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rolling/lag features (ίδιο με πριν) + net_load_total_mw."""
    df = df.copy()
    df["rolling_mean_1day"] = df["mcp_eur_per_mwh"].shift(24).rolling(24).mean()
    df["rolling_mean_7days"] = df["mcp_eur_per_mwh"].shift(24).rolling(24 * 7).mean()
    df["lag_24h"] = df["mcp_eur_per_mwh"].shift(24)
    df["lag_25h"] = df["mcp_eur_per_mwh"].shift(25)
    df["lag_48h"] = df["mcp_eur_per_mwh"].shift(48)
    df["lag_168h"] = df["mcp_eur_per_mwh"].shift(168)

    df["rolling_max_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).max()
    df["rolling_min_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).min()

    # net_out ενσωματωμένο στο συνολικό φορτίο, αντί να μπαίνει raw ξεχωριστά
    df["net_load_total_mw"] = df["load_forecast_mw"] + df["net_out"] - df["res_forecast_mw"]

    return df.dropna().reset_index(drop=True)


def add_daily_weekly_skew_kurt(frame: pd.DataFrame, col: str, prefix: str) -> pd.DataFrame:
    """Ίδια σύμβαση με τα προηγούμενα πειράματα (price_model_v26/v27/v28,
    price_forecasting_attempt1): {prefix}_daily_skew/kurt = στατιστικό
    πάνω στην ημέρα d-1, {prefix}_weekly_skew/kurt = πάνω στις ώρες
    d-7..d-1. Broadcast σε όλες τις ώρες της ημέρας d. Χωρίς leakage.
    Γίνεται merge (left) -- δεν πετάει γραμμές, μόνο η πρώτη εβδομάδα του
    dataset θα έχει NaN σε αυτές τις στήλες."""
    frame = frame.copy()
    frame["date"] = frame["timestamp"].dt.normalize()

    daily_skew = frame.groupby("date")[col].skew().sort_index()
    daily_kurt = frame.groupby("date")[col].apply(pd.Series.kurt).sort_index()

    hourly = frame.set_index("timestamp")[col]
    daily_index = daily_skew.index

    def _window_stat(stat_func):
        out = {}
        for d in daily_index:
            start = d - pd.Timedelta(days=7)
            end = d - pd.Timedelta(seconds=1)
            window = hourly.loc[start:end]
            out[d] = stat_func(window) if len(window) > 1 else np.nan
        return pd.Series(out)

    weekly_skew = _window_stat(lambda s: s.skew())
    weekly_kurt = _window_stat(lambda s: s.kurt())

    daily_skew_prev = daily_skew.shift(1).rename(f"{prefix}_daily_skew")
    daily_kurt_prev = daily_kurt.shift(1).rename(f"{prefix}_daily_kurt")
    weekly_skew_map = weekly_skew.rename(f"{prefix}_weekly_skew")
    weekly_kurt_map = weekly_kurt.rename(f"{prefix}_weekly_kurt")

    frame = frame.merge(daily_skew_prev, left_on="date", right_index=True, how="left")
    frame = frame.merge(daily_kurt_prev, left_on="date", right_index=True, how="left")
    frame = frame.merge(weekly_skew_map, left_on="date", right_index=True, how="left")
    frame = frame.merge(weekly_kurt_map, left_on="date", right_index=True, how="left")
    return frame


def merge_into_vol2_final(new_data: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Ενημερώνει/δημιουργεί το vol2_final.csv ΧΩΡΙΣ να χάνει στήλες που
    έγραψαν εκεί προηγούμενα runs ΑΛΛΩΝ scripts (π.χ. οι προβλέψεις του
    hybrid original ή του single_no_kurt) -- το vol2_final είναι κοινό/
    σωρευτικό ανάμεσα σε όλα τα rolling_evaluation_*.py.

    new_data: πρέπει να έχει στήλη "timestamp" -- όλες οι υπόλοιπες
    στήλες του μπαίνουν/ενημερώνονται στο αρχείο, ό,τι άλλο υπάρχει ήδη
    εκεί (από άλλο script) παραμένει ανέπαφο."""
    new_data = new_data.set_index("timestamp")
    if path.exists():
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        existing = existing.set_index("timestamp")
        full_index = existing.index.union(new_data.index)
        existing = existing.reindex(full_index)
        new_data = new_data.reindex(full_index)
        for col in new_data.columns:
            existing[col] = new_data[col]
        result = existing
    else:
        result = new_data
    return result.sort_index().reset_index()


def _resolve_eval_window(df: pd.DataFrame, eval_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-09-01")
    end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"2026-06-30")

    max_available = df["timestamp"].max()
    if end > max_available:
        print(f"[Προσοχή] EVAL_END ({end.date()}) > μέγιστη διαθέσιμη ημερομηνία "
              f"στο dataset ({max_available.date()}). Κόβεται αυτόματα.")
        end = max_available.normalize()
    return start, end


def get_feature_columns(df: pd.DataFrame) -> tuple[list, list]:
    """Ενιαία πηγή αλήθειας για το ποιες στήλες χρησιμοποιούνται πραγματικά
    ως features -- τη χρησιμοποιεί και το training loop, και το main για
    να αποφασίσει τι μπαίνει στο vol2_final (μόνο ό,τι όντως χρησιμοποιείται
    ή φτιάχτηκε, όχι raw στήλες του dataset που αγνοούνται, π.χ. outage_mw)."""
    feature_cols_general = [c for c in df.columns if c not in EXCLUDE_ALWAYS and c not in SKEW_KURT_COLS]
    feature_cols_zone = feature_cols_general + SKEW_KURT_COLS
    return feature_cols_general, feature_cols_zone


def run_rolling_evaluation_hybrid(
    df: pd.DataFrame, eval_year: int,
    feature_cols_general: list, feature_cols_zone: list,
) -> pd.DataFrame:
    train_hours = MONTHS_USED * 30 * 24
    eval_start, eval_end = _resolve_eval_window(df, eval_year)

    hourly_records = []
    current_day = eval_start
    n_days = (eval_end - eval_start).days + 1
    i = 0

    while current_day <= eval_end:
        i += 1
        test_start = current_day
        test_end = current_day + pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        train_start = train_end - pd.Timedelta(hours=train_hours - 1)

        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()
        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)].copy()

        if len(df_test) < 24 or len(df_train) < MIN_TRAIN_HOURS:
            print(f"[{i}/{n_days}] {current_day.date()} -> παραλείπεται (ελλιπή δεδομένα)")
            current_day += pd.Timedelta(days=1)
            continue

        is_zone_train = df_train["hour"].isin(PEAK_HOURS)
        is_zone_test = df_test["hour"].isin(PEAK_HOURS).values

        # --- Μοντέλο A: γενικό, εκπαιδευμένο σε ΟΛΕΣ τις ώρες, ΧΩΡΙΣ skew/kurt ---
        X_train_all = df_train[feature_cols_general]
        y_train_all = df_train["mcp_eur_per_mwh"]
        model_all = xgb.XGBRegressor(**XGB_PARAMS)
        model_all.fit(X_train_all, y_train_all)

        # --- Μοντέλο B: zone-specific, ΜΕ skew/kurt, μόνο στη ζώνη αιχμής ---
        df_train_zone = df_train[is_zone_train].dropna(subset=SKEW_KURT_COLS)
        model_zone = None
        if len(df_train_zone) >= 100:  # ελάχιστο αξιόπιστο δείγμα για fit
            model_zone = xgb.XGBRegressor(**XGB_PARAMS)
            model_zone.fit(df_train_zone[feature_cols_zone], df_train_zone["mcp_eur_per_mwh"])

        # --- Πρόβλεψη: hybrid routing βάσει ώρας ---
        y_pred = model_all.predict(df_test[feature_cols_general])
        if model_zone is not None and is_zone_test.any():
            zone_rows = df_test[is_zone_test]
            y_pred[is_zone_test] = model_zone.predict(zone_rows[feature_cols_zone])

        y_test = df_test["mcp_eur_per_mwh"].values
        mae_day = float(np.mean(np.abs(y_pred - y_test)))

        for ts, actual, pred, hr in zip(df_test["timestamp"].values, y_test, y_pred, df_test["hour"].values):
            hourly_records.append({"timestamp": ts, "actual": actual, "predicted": pred, "hour": hr})

        print(f"[{i}/{n_days}] {current_day.date()} -> MAE: {mae_day:.2f}")

        current_day += pd.Timedelta(days=1)

    return pd.DataFrame(hourly_records)


def summarize(hourly: pd.DataFrame) -> None:
    hourly = hourly.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])
    hourly["abs_err"] = (hourly["predicted"] - hourly["actual"]).abs()
    hourly["year_month"] = hourly["timestamp"].dt.to_period("M")

    print("\n" + "=" * 50)
    print(f"MAE ΑΝΑ ΜΗΝΑ — {len(hourly)} αξιολογημένες ώρες")
    print("=" * 50)
    mae_by_month = hourly.groupby("year_month")["abs_err"].agg(["mean", "count"])
    mae_by_month = mae_by_month.rename(columns={"mean": "mae", "count": "n_hours"})
    print(mae_by_month.to_string())

    print("\n" + "=" * 50)
    print("MAE ΑΝΑ ΩΡΑ")
    print("=" * 50)
    mae_by_hour = hourly.groupby("hour")["abs_err"].agg(["mean", "count"])
    mae_by_hour = mae_by_hour.rename(columns={"mean": "mae", "count": "n_days"})
    for hour, row in mae_by_hour.iterrows():
        print(f"  {int(hour):02d}:00  MAE={row['mae']:.2f}  (n={int(row['n_days'])})")

    print("\n" + "=" * 50)
    print("ΣΥΝΟΛΙΚΟ MAE")
    print("=" * 50)
    print(f"MAE = {hourly['abs_err'].mean():.2f} EUR/MWh  (n={len(hourly)} ώρες)")


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering (rolling/lag + net_load_total_mw)...")
    df = build_features(df)

    print("Υπολογισμός price_daily/weekly_skew/kurt (μόνο για το zone μοντέλο)...")
    df = add_daily_weekly_skew_kurt(df, "mcp_eur_per_mwh", "price")

    feature_cols_general, feature_cols_zone = get_feature_columns(df)

    # --- vol2_final base: timestamp + target + ΜΟΝΟ ό,τι όντως χρησιμοποιείται
    # ή φτιάχτηκε (feature_cols_zone είναι υπερσύνολο -- γενικό + skew/kurt).
    # Raw στήλες που αγνοούνται (π.χ. outage_mw, raw net_out) ΔΕΝ μπαίνουν.
    vol2_final_cols = ["timestamp", "mcp_eur_per_mwh"] + feature_cols_zone
    vol2_final = df[vol2_final_cols].copy()

    print(f"Rolling HYBRID evaluation για το έτος {EVAL_YEAR}...\n")
    hourly = run_rolling_evaluation_hybrid(df, EVAL_YEAR, feature_cols_general, feature_cols_zone)

    if hourly.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε το EVAL_YEAR/EVAL_START/EVAL_END/dataset.")
    else:
        summarize(hourly)

        out_path = DATASET_PATH.parent / f"rolling_evaluation_hybrid_v2_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα ωριαία αποτελέσματα: {out_path}")

        # --- προσθήκη των προβλέψεων στο vol2_final -- ΔΙΚΗ ΤΟΥ στήλη,
        # ώστε να μη σβήνει τις προβλέψεις άλλων μοντέλων/scripts ---
        preds_for_merge = hourly[["timestamp", "predicted"]].rename(
            columns={"predicted": "predicted_hybrid_v2"}
        )
        preds_for_merge["timestamp"] = pd.to_datetime(preds_for_merge["timestamp"])
        vol2_final = vol2_final.merge(preds_for_merge, on="timestamp", how="left")

    vol2_final = merge_into_vol2_final(vol2_final, VOL2_FINAL_PATH)
    vol2_final.to_csv(VOL2_FINAL_PATH, index=False)
    print(f"\nΑποθηκεύτηκε/ενημερώθηκε το vol2_final (σωρευτικό, ΔΕΝ σβήνει προβλέψεις "
          f"άλλων scripts): {VOL2_FINAL_PATH}")
    print(f"  Στήλες: {list(vol2_final.columns)}")
    pred_cols = [c for c in vol2_final.columns if c.startswith("predicted_")]
    for c in pred_cols:
        print(f"  {c}: {vol2_final[c].notna().sum()} προβλέψεις")