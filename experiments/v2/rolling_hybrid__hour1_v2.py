"""
rolling_evaluation_hour1_v2_allhours.py
=========================================
Παραλλαγή του rolling_evaluation_hour1_v2.py: ΤΟ ΙΔΙΟ feature set/
hyperparameters/lag_1h-lag_2h λογική, ΑΛΛΑ το μοντέλο πλέον εκπαιδεύεται
σε ΟΛΕΣ τις ώρες (24x περισσότερα δεδομένα ανά rolling window), όχι
μόνο σε δεδομένα ώρας 01:00. Η στήλη "hour" παραμένει feature, ώστε το
μοντέλο να μπορεί να διαχωρίσει τη συμπεριφορά ανά ώρα.

Η ΠΡΟΒΛΕΨΗ γίνεται ΜΟΝΟ για την ώρα 01:00 κάθε ημέρας (ίδιο test set με
το hour1_v2.py) -- σκοπός είναι να δούμε αν η μεγαλύτερη ποικιλία
δεδομένων training βοηθάει το v2 μοντέλο (που αποδίδει χειρότερα από το
original στη σύγκριση hour-specific).

lag_2h: κανονικό, καμία διαρροή (shift(2) σε όλο το ωριαίο dataset πριν
  το split -- ίδιο όπως πριν).
lag_1h: στο TEST = πραγματική πρόβλεψη του μοντέλου ώρας 0 (στήλη
  hour_specific του vol2_final.csv, hour==0). Στο TRAINING = πραγματική
  τιμή + Gaussian θόρυβος std=HOUR0_NOISE_STD=13.44 (ίδιο με πριν,
  εφαρμόζεται σε όλες τις γραμμές πριν το split, όχι μόνο στην ώρα 1).

Rolling training: MONTHS_USED (18) πριν από κάθε ημέρα, ΟΛΕΣ οι ώρες.
Eval window (προβλέψεις μόνο ώρας 1): 2025-09-01 -> 2026-06-30.

Οι προβλέψεις γράφονται ΣΤΟ ΙΔΙΟ ΣΗΜΕΙΟ με πριν: vol2_final.csv, στήλη
"hour_specific", ίδιοι timestamps ώρας 01:00 -- προσοχή, θα αντικαταστήσει
όποιες τιμές είχε γράψει εκεί το rolling_evaluation_hour1_v2.py.
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
VOL2_FINAL_PATH = DATASET_PATH.parent / "vol2_final.csv"

EVAL_START = "2025-09-01"
EVAL_END = "2026-06-30"

MONTHS_USED = 18
MIN_TRAIN_DAYS = 40   # ελάχιστες ΔΙΑΚΡΙΤΕΣ ημέρες στο training window (όχι γραμμές)

QUANTILE_ALPHA = 0.5

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILE_ALPHA,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

HOUR0_NOISE_STD = 13.44   # RMSE του μοντέλου ώρας 0 (v2) -- σταθερό εδώ
NOISE_SEED = 42

REGULAR_MODEL_COLUMN = "predicted_hybrid_v2"   # για τη σύγκριση στο τέλος

EXCLUDE_ALWAYS = {
    "timestamp", "mcp_eur_per_mwh", "date",
    "net_out",       # ενσωματωμένο στο net_load_total_mw
    "outage_mw",     # δεν βοηθούσε
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ίδιο base feature engineering με το rolling_evaluation_hybrid_v2.py
    + lag_1h (θορυβώδες για training) + lag_2h. Εφαρμόζεται στο ΠΛΗΡΕΣ
    ωριαίο dataset (όλες οι ώρες) -- όχι φιλτραρισμένο."""
    df = df.copy()
    df["rolling_mean_1day"] = df["mcp_eur_per_mwh"].shift(24).rolling(24).mean()
    df["rolling_mean_7days"] = df["mcp_eur_per_mwh"].shift(24).rolling(24 * 7).mean()
    df["lag_24h"] = df["mcp_eur_per_mwh"].shift(24)
    df["lag_25h"] = df["mcp_eur_per_mwh"].shift(25)
    df["lag_48h"] = df["mcp_eur_per_mwh"].shift(48)
    df["lag_168h"] = df["mcp_eur_per_mwh"].shift(168)

    df["rolling_max_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).max()
    df["rolling_min_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).min()

    df["net_load_total_mw"] = df["load_forecast_mw"] + df["net_out"] - df["res_forecast_mw"]

    df["lag_2h"] = df["mcp_eur_per_mwh"].shift(2)

    lag_1h_actual = df["mcp_eur_per_mwh"].shift(1)
    rng = np.random.default_rng(NOISE_SEED)
    noise = rng.normal(loc=0.0, scale=HOUR0_NOISE_STD, size=len(df))
    df["lag_1h"] = lag_1h_actual + noise

    return df.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_ALWAYS]


def load_hour0_predictions(path: Path) -> pd.Series:
    if not path.exists():
        print(f"[Προσοχή] Δεν βρέθηκε {path} -- καμία διαθέσιμη πρόβλεψη ώρας 0.")
        return pd.Series(dtype=float)
    vol2 = pd.read_csv(path, usecols=lambda c: c in ("timestamp", "hour", "hour_specific"))
    if "hour_specific" not in vol2.columns:
        print(f"[Προσοχή] Το {path} δεν έχει στήλη hour_specific ακόμα.")
        return pd.Series(dtype=float)
    vol2["timestamp"] = pd.to_datetime(vol2["timestamp"])
    hour0 = vol2[(vol2["hour"] == 0) & vol2["hour_specific"].notna()].copy()
    hour0["date"] = hour0["timestamp"].dt.normalize()
    return hour0.set_index("date")["hour_specific"]


def load_regular_model_predictions(path: Path, column: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", column])
    vol2 = pd.read_csv(path, usecols=lambda c: c in ("timestamp", "hour", column))
    if column not in vol2.columns:
        print(f"[Προσοχή] Το {path} δεν έχει στήλη {column} -- καμία σύγκριση με το κανονικό μοντέλο.")
        return pd.DataFrame(columns=["timestamp", column])
    vol2["timestamp"] = pd.to_datetime(vol2["timestamp"])
    return vol2[(vol2["hour"] == 1) & vol2[column].notna()][["timestamp", column]]


def merge_into_vol2_final(new_data: pd.DataFrame, path: Path) -> pd.DataFrame:
    new_data = new_data.set_index("timestamp")
    if path.exists():
        existing = pd.read_csv(path)
        existing["timestamp"] = pd.to_datetime(existing["timestamp"])
        existing = existing.set_index("timestamp")
        full_index = existing.index.union(new_data.index)
        existing = existing.reindex(full_index)
        for col in new_data.columns:
            if col not in existing.columns:
                existing[col] = np.nan
            existing.loc[new_data.index, col] = new_data[col]
        result = existing
    else:
        result = new_data
    return result.sort_index().reset_index()


def run_hour1_rolling(df_full: pd.DataFrame, feature_cols: list, hour0_preds: pd.Series) -> pd.DataFrame:
    """df_full περιέχει ΟΛΕΣ τις ώρες -- το training window παίρνει ΟΛΕΣ
    τις γραμμές (όλων των ωρών) μέσα στο rolling παράθυρο, ενώ το test
    row είναι πάντα η ώρα 01:00 της ημέρας."""
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = min(pd.Timestamp(EVAL_END), df_full["timestamp"].max())
    if pd.Timestamp(EVAL_END) > df_full["timestamp"].max():
        print(f"[Προσοχή] EVAL_END ({EVAL_END}) > μέγιστη διαθέσιμη ημερομηνία "
              f"({df_full['timestamp'].max().date()}). Κόβεται αυτόματα.")

    df_full = df_full.set_index("timestamp").sort_index()
    records = []
    test_days = pd.date_range(eval_start, eval_end, freq="D")

    for i, day in enumerate(test_days, start=1):
        train_start = day - pd.DateOffset(months=MONTHS_USED)
        train = df_full.loc[train_start:day - pd.Timedelta(seconds=1)]

        test_timestamp = day + pd.Timedelta(hours=1)
        if test_timestamp not in df_full.index:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (λείπει η ώρα 1)")
            continue

        train_days_available = train.index.normalize().nunique()
        if train_days_available < MIN_TRAIN_DAYS:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται "
                  f"(ελλιπές training, {train_days_available} ημέρες, {len(train)} γραμμές)")
            continue
        if day not in hour0_preds.index:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (λείπει πρόβλεψη ώρας 0)")
            continue

        X_train, y_train = train[feature_cols], train["mcp_eur_per_mwh"]

        X_test = df_full.loc[[test_timestamp], feature_cols].copy()
        X_test["lag_1h"] = hour0_preds.loc[day]
        y_test = df_full.loc[test_timestamp, "mcp_eur_per_mwh"]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = float(model.predict(X_test)[0])

        err = abs(y_pred - y_test)
        records.append({"timestamp": test_timestamp, "actual": y_test, "predicted": y_pred})
        print(f"[{i}/{len(test_days)}] {day.date()} 01:00 -> actual={y_test:.2f}  "
              f"pred={y_pred:.2f}  |err|={err:.2f}  (train: {len(train)} γραμμές / "
              f"{train_days_available} ημέρες)")

    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering (v2 base + lag_2h πραγματικό + lag_1h θορυβώδες, όλες οι ώρες)...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    print(f"Φόρτωση προβλέψεων ώρας 0 από: {VOL2_FINAL_PATH}")
    hour0_preds = load_hour0_predictions(VOL2_FINAL_PATH)
    print(f"Διαθέσιμες προβλέψεις ώρας 0: {len(hour0_preds)}")

    print(f"Γραμμές συνολικά διαθέσιμες (όλες οι ώρες): {len(df)}")

    print(f"\nRolling ΩΡΑ-1 evaluation (v2 base, training σε ΟΛΕΣ τις ώρες) "
          f"{EVAL_START} -> {EVAL_END}...\n")
    preds = run_hour1_rolling(df, feature_cols, hour0_preds)

    vol2_final = pd.DataFrame(columns=["timestamp", "hour_specific"])
    if preds.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε EVAL_START/EVAL_END/dataset/vol2_final.")
    else:
        mae = float(np.mean(np.abs(preds["predicted"] - preds["actual"])))
        rmse = float(np.sqrt(np.mean((preds["predicted"] - preds["actual"]) ** 2)))
        print("\n" + "=" * 50)
        print(f"ΩΡΑ 01:00 (all-hours training, v2 base) -- {len(preds)} ημέρες, {EVAL_START} -> {EVAL_END}")
        print("=" * 50)
        print(f"MAE  = {mae:.2f} EUR/MWh")
        print(f"RMSE = {rmse:.2f} EUR/MWh")

        # --- σύγκριση με το κανονικό (multi-hour) μοντέλο, ίδια ώρα/ημέρες ---
        regular = load_regular_model_predictions(VOL2_FINAL_PATH, REGULAR_MODEL_COLUMN)
        merged = preds.merge(regular, on="timestamp", how="inner")
        print("\n" + "=" * 50)
        print(f"ΣΥΓΚΡΙΣΗ με το κανονικό μοντέλο ({REGULAR_MODEL_COLUMN}), ίδιες {len(merged)} ημέρες")
        print("=" * 50)
        if merged.empty:
            print("Δεν βρέθηκαν προβλέψεις του κανονικού μοντέλου για σύγκριση.")
        else:
            reg_mae = float((merged[REGULAR_MODEL_COLUMN] - merged["actual"]).abs().mean())
            reg_rmse = float(((merged[REGULAR_MODEL_COLUMN] - merged["actual"]) ** 2).mean() ** 0.5)
            hs_mae = float((merged["predicted"] - merged["actual"]).abs().mean())
            hs_rmse = float(((merged["predicted"] - merged["actual"]) ** 2).mean() ** 0.5)
            print(f"Κανονικό μοντέλο:      MAE={reg_mae:.2f}  RMSE={reg_rmse:.2f}")
            print(f"Hour-specific (all-hours training): MAE={hs_mae:.2f}  RMSE={hs_rmse:.2f}")
            print(f"Διαφορά MAE:  {reg_mae - hs_mae:+.2f} EUR/MWh "
                  f"({'βελτίωση' if hs_mae < reg_mae else 'χειρότερο'})")
            print(f"Διαφορά RMSE: {reg_rmse - hs_rmse:+.2f} EUR/MWh "
                  f"({'βελτίωση' if hs_rmse < reg_rmse else 'χειρότερο'})")

        out_path = DATASET_PATH.parent / "rolling_evaluation_hour1_v2_allhours_predictions.csv"
        preds.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν οι αναλυτικές προβλέψεις: {out_path}")

        vol2_final = preds[["timestamp", "predicted"]].rename(columns={"predicted": "hour_specific"})

    vol2_final = merge_into_vol2_final(vol2_final, VOL2_FINAL_PATH)
    vol2_final.to_csv(VOL2_FINAL_PATH, index=False)
    print(f"\nΑποθηκεύτηκε/ενημερώθηκε το vol2_final: {VOL2_FINAL_PATH}")
    print(f"  hour_specific: {vol2_final['hour_specific'].notna().sum()} προβλέψεις συνολικά (σωρευτικά)")