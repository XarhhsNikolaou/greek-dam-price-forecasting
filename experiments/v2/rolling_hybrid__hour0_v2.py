"""
rolling_evaluation_hour0_v2.py
================================
Ειδικό μοντέλο ΜΟΝΟ για την ώρα 00:00, βασισμένο στο feature set/
hyperparameters του rolling_evaluation_hybrid_v2.py (γενικό μοντέλο --
η ώρα 0 δεν ανήκει στο PEAK_HOURS του v2, άρα ούτως ή άλλως θα έπαιρνε
πρόβλεψη από το γενικό μοντέλο, όχι το zone-specific -- οπότε εδώ δεν
χρειάζεται καν hybrid routing, μόνο ΕΝΑ μοντέλο).

ΕΞΤΡΑ features lag_1h / lag_2h (mcp_eur_per_mwh.shift(1)/.shift(2)):
ΓΙΑΤΙ ΕΙΝΑΙ ΑΣΦΑΛΗ ΜΟΝΟ ΓΙΑ ΩΡΑ 0, ΟΧΙ ΓΙΑ ΑΛΛΕΣ ΩΡΕΣ:
  Η day-ahead δημοπρασία για την ημέρα D κλείνει γύρω στο μεσημέρι της
  D-1 και καθορίζει ΟΛΕΣ τις 24 ώρες της D ταυτόχρονα. Άρα για ώρα h>0
  της D, το lag_1h/lag_2h θα αναφερόταν σε ώρα (h-1)/(h-2) της ΙΔΙΑΣ D
  -- κομμάτι της ΙΔΙΑΣ, ακόμα άγνωστης δημοπρασίας -> leakage.
  Για ώρα 0 όμως, lag_1h/lag_2h αναφέρονται στις ώρες 23:00/22:00 της
  D-1 -- που ανήκουν σε ΠΡΟΗΓΟΥΜΕΝΗ, ήδη διεκπεραιωμένη δημοπρασία
  (αυτή της D-1, που έκλεισε ένα μερόνυχτο νωρίτερα). Άρα γνωστά εκ
  των προτέρων, καμία διαρροή.

Rolling training: MONTHS_USED (18) πριν από κάθε ημέρα, ΜΟΝΟ σε
δεδομένα ώρας 0 (ένα δείγμα/ημέρα -- η lag_1h/lag_2h δεν έχουν καν
νόημα για άλλες ώρες, οπότε δεν έχει νόημα να εκπαιδευτεί πάνω σε αυτές).

Eval window: 2025-09-01 -> 2026-06-30 -- το "υπόλοιπο" της περιόδου
Ιούλιος 2025-Ιούνιος 2026, αφού το καλοκαίρι (Ιούλ-Αύγ 2025) το καλύπτει
το αντίστοιχο rolling_evaluation_hour0_original.py (εκεί αποδίδει
καλύτερα το original hybrid).

Οι προβλέψεις γράφονται στο vol2_final.csv, στήλη "hour_specific" --
ΙΔΙΑ στήλη με το rolling_evaluation_hour0_original.py, αφού καλύπτουν
συμπληρωματικές (όχι επικαλυπτόμενες) περιόδους -- μαζί φτιάχνουν ΕΝΑ
πλήρες σετ προβλέψεων ώρας 0 για όλο το testing window.

Reporting: MAE και RMSE μόνο για την ώρα 00:00, στο eval window αυτού
του script.
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
MIN_TRAIN_DAYS = 40   # αντίστοιχο του MIN_TRAIN_HOURS=1000 στα multi-hour scripts, σε ημέρες

QUANTILE_ALPHA = 0.5

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILE_ALPHA,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

EXCLUDE_ALWAYS = {
    "timestamp", "mcp_eur_per_mwh", "date",
    "net_out",       # ενσωματωμένο στο net_load_total_mw (ίδιο με hybrid_v2)
    "outage_mw",     # δεν βοηθούσε (ίδιο με hybrid_v2)
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ίδιο base feature engineering με το rolling_evaluation_hybrid_v2.py
    + lag_1h/lag_2h."""
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

    # ΕΞΤΡΑ -- ασφαλή ΜΟΝΟ για ώρα 0 (βλ. docstring)
    df["lag_1h"] = df["mcp_eur_per_mwh"].shift(1)
    df["lag_2h"] = df["mcp_eur_per_mwh"].shift(2)

    return df.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in EXCLUDE_ALWAYS]


def merge_into_vol2_final(new_data: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Ίδια λογική merge με τα υπόλοιπα rolling_evaluation_*.py -- σωρευτικό,
    δεν σβήνει στήλες που έγραψαν άλλα scripts (ή προηγούμενα runs)."""
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
            # ΜΟΝΟ οι γραμμές που το new_data όντως καλύπτει -- ΟΧΙ blanket
            # overwrite με reindexed (γεμάτο NaN εκτός new_data) column,
            # αλλιώς σβήνει προϋπάρχουσες τιμές άλλων runs στην ΙΔΙΑ στήλη
            # (π.χ. δύο hour0 scripts που γράφουν και τα δύο "hour_specific"
            # σε διαφορετικές, μη επικαλυπτόμενες περιόδους).
            existing.loc[new_data.index, col] = new_data[col]
        result = existing
    else:
        result = new_data
    return result.sort_index().reset_index()


def run_hour0_rolling(df_h0: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = min(pd.Timestamp(EVAL_END), df_h0["timestamp"].max())
    if pd.Timestamp(EVAL_END) > df_h0["timestamp"].max():
        print(f"[Προσοχή] EVAL_END ({EVAL_END}) > μέγιστη διαθέσιμη ημερομηνία ώρας-0 "
              f"({df_h0['timestamp'].max().date()}). Κόβεται αυτόματα.")

    df_h0 = df_h0.set_index("timestamp").sort_index()
    records = []
    test_days = pd.date_range(eval_start, eval_end, freq="D")

    for i, day in enumerate(test_days, start=1):
        train_start = day - pd.DateOffset(months=MONTHS_USED)
        train = df_h0.loc[train_start:day - pd.Timedelta(seconds=1)]

        if day not in df_h0.index:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (λείπει η ώρα 0)")
            continue
        if len(train) < MIN_TRAIN_DAYS:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (ελλιπές training, {len(train)} ημέρες)")
            continue

        X_train, y_train = train[feature_cols], train["mcp_eur_per_mwh"]
        X_test = df_h0.loc[[day], feature_cols]
        y_test = df_h0.loc[day, "mcp_eur_per_mwh"]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = float(model.predict(X_test)[0])

        err = abs(y_pred - y_test)
        records.append({"timestamp": day, "actual": y_test, "predicted": y_pred})
        print(f"[{i}/{len(test_days)}] {day.date()} 00:00 -> actual={y_test:.2f}  "
              f"pred={y_pred:.2f}  |err|={err:.2f}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering (v2 base + lag_1h/lag_2h)...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    df_h0 = df[df["hour"] == 0].copy()
    print(f"Γραμμές ώρας 00:00 διαθέσιμες: {len(df_h0)}")

    print(f"\nRolling ΩΡΑ-0 evaluation (hybrid_v2 base) {EVAL_START} -> {EVAL_END}...\n")
    preds = run_hour0_rolling(df_h0, feature_cols)

    vol2_final = pd.DataFrame(columns=["timestamp", "hour_specific"])
    if preds.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε EVAL_START/EVAL_END/dataset.")
    else:
        mae = float(np.mean(np.abs(preds["predicted"] - preds["actual"])))
        rmse = float(np.sqrt(np.mean((preds["predicted"] - preds["actual"]) ** 2)))
        print("\n" + "=" * 50)
        print(f"ΩΡΑ 00:00 -- MAE/RMSE ({len(preds)} ημέρες, {EVAL_START} -> {EVAL_END}, hybrid_v2 base)")
        print("=" * 50)
        print(f"MAE  = {mae:.2f} EUR/MWh")
        print(f"RMSE = {rmse:.2f} EUR/MWh")

        out_path = DATASET_PATH.parent / "rolling_evaluation_hour0_v2_predictions.csv"
        preds.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν οι αναλυτικές προβλέψεις: {out_path}")

        vol2_final = preds[["timestamp", "predicted"]].rename(columns={"predicted": "hour_specific"})

    vol2_final = merge_into_vol2_final(vol2_final, VOL2_FINAL_PATH)
    vol2_final.to_csv(VOL2_FINAL_PATH, index=False)
    print(f"\nΑποθηκεύτηκε/ενημερώθηκε το vol2_final: {VOL2_FINAL_PATH}")
    print(f"  hour_specific: {vol2_final['hour_specific'].notna().sum()} προβλέψεις συνολικά "
          f"(σωρευτικά με το rolling_evaluation_hour0_original.py, αν έχει ήδη τρέξει)")