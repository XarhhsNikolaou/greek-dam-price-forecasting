"""
rolling_evaluation_hour3_original.py
======================================
Ειδικό μοντέλο ΜΟΝΟ για την ώρα 03:00, βασισμένο στο feature set/
hyperparameters του rolling_evaluation_hybrid.py (original -- outage_mw
μέσα). Καλύπτει ΜΟΝΟ Ιούλιο-Αύγουστο 2025 -- συμπληρωματικό του
rolling_evaluation_hour3_v2.py.

ΕΞΤΡΑ features lag_1h / lag_2h -- ίδια λογική με την ώρα 2, μετατοπισμένη
κατά μία ώρα:
  Για ώρα 3 της ημέρας D:
    lag_1h αναφέρεται σε ώρα 2 της ΙΔΙΑΣ D -> ΧΡΕΙΑΖΕΤΑΙ το μοντέλο ώρας 2.
    lag_2h αναφέρεται σε ώρα 1 της ΙΔΙΑΣ D -> ΧΡΕΙΑΖΕΤΑΙ το μοντέλο ώρας 1.

  Στο TEST: lag_1h = πραγματική πρόβλεψη μοντέλου ώρας 2, lag_2h =
    πραγματική πρόβλεψη μοντέλου ώρας 1 (και τα δύο από τη στήλη
    hour_specific του vol2_final.csv, φιλτραρισμένα ανά hour).
  Στο TRAINING: lag_1h = πραγματική τιμή ώρας 2 + θόρυβος std=HOUR2_
    NOISE_STD (RMSE μοντέλου ώρας 2, original = 7.30 -- όπως μόλις
    αναφέρθηκε), lag_2h = πραγματική τιμή ώρας 1 + θόρυβος std=HOUR1_
    NOISE_STD (RMSE μοντέλου ώρας 1, original = 6.98).

Στο τέλος συγκρίνει με το "κανονικό" (multi-hour) μοντέλο, ΙΔΙΑ ώρα (3)
ΙΔΙΕΣ ημέρες, μέσω της στήλης predicted_hybrid_original -- ΠΛΕΟΝ με
ρητό diagnostic: αν λείπουν μέρες από τη στήλη σύγκρισης, τυπώνεται
ΠΟΣΕΣ και ΠΟΙΟ διάστημα, ώστε να είναι ξεκάθαρο αν πρόκειται για κενό
δεδομένων (π.χ. ημιτελές run του αντίστοιχου multi-hour script) και όχι
bug στη σύγκριση.

Rolling training: MONTHS_USED (18), ΜΟΝΟ δεδομένα ώρας 3.
Eval window: 2025-07-01 -> 2025-08-31.

Οι προβλέψεις γράφονται στο vol2_final.csv, στήλη "hour_specific".
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

EVAL_START = "2025-07-01"
EVAL_END = "2025-08-31"

MONTHS_USED = 18
MIN_TRAIN_DAYS = 40

QUANTILE_ALPHA = 0.5

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILE_ALPHA,
    max_depth=6,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

HOUR1_NOISE_STD = 6.98   # RMSE μοντέλου ώρας 1 (original)
HOUR2_NOISE_STD = 7.30   # RMSE μοντέλου ώρας 2 (original)
NOISE_SEED = 42

REGULAR_MODEL_COLUMN = "predicted_hybrid_original"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rolling_mean_1day"] = df["mcp_eur_per_mwh"].shift(24).rolling(24).mean()
    df["rolling_mean_7days"] = df["mcp_eur_per_mwh"].shift(24).rolling(24 * 7).mean()
    df["lag_24h"] = df["mcp_eur_per_mwh"].shift(24)
    df["lag_25h"] = df["mcp_eur_per_mwh"].shift(25)
    df["lag_48h"] = df["mcp_eur_per_mwh"].shift(48)
    df["lag_168h"] = df["mcp_eur_per_mwh"].shift(168)

    df["rolling_max_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).max()
    df["rolling_min_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).min()

    rng = np.random.default_rng(NOISE_SEED)

    # lag_1h <- ρόλος ώρας 2 (η ώρα ακριβώς πριν)
    lag_1h_actual = df["mcp_eur_per_mwh"].shift(1)
    noise1 = rng.normal(loc=0.0, scale=HOUR2_NOISE_STD, size=len(df))
    df["lag_1h"] = lag_1h_actual + noise1

    # lag_2h <- ρόλος ώρας 1
    lag_2h_actual = df["mcp_eur_per_mwh"].shift(2)
    noise2 = rng.normal(loc=0.0, scale=HOUR1_NOISE_STD, size=len(df))
    df["lag_2h"] = lag_2h_actual + noise2

    return df.dropna().reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]


def load_hour_predictions(path: Path, hour: int) -> pd.Series:
    if not path.exists():
        print(f"[Προσοχή] Δεν βρέθηκε {path} -- καμία διαθέσιμη πρόβλεψη ώρας {hour}.")
        return pd.Series(dtype=float)
    vol2 = pd.read_csv(path, usecols=lambda c: c in ("timestamp", "hour", "hour_specific"))
    if "hour_specific" not in vol2.columns:
        print(f"[Προσοχή] Το {path} δεν έχει στήλη hour_specific ακόμα.")
        return pd.Series(dtype=float)
    vol2["timestamp"] = pd.to_datetime(vol2["timestamp"])
    sub = vol2[(vol2["hour"] == hour) & vol2["hour_specific"].notna()].copy()
    sub["date"] = sub["timestamp"].dt.normalize()
    return sub.set_index("date")["hour_specific"]


def load_regular_model_predictions(path: Path, column: str, hour: int) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", column])
    vol2 = pd.read_csv(path, usecols=lambda c: c in ("timestamp", "hour", column))
    if column not in vol2.columns:
        print(f"[Προσοχή] Το {path} δεν έχει στήλη {column} -- καμία σύγκριση με το κανονικό μοντέλο.")
        return pd.DataFrame(columns=["timestamp", column])
    vol2["timestamp"] = pd.to_datetime(vol2["timestamp"])
    return vol2[(vol2["hour"] == hour) & vol2[column].notna()][["timestamp", column]]


def diagnose_regular_model_coverage(preds: pd.DataFrame, regular: pd.DataFrame, column: str) -> None:
    """Εξηγεί ΓΙΑΤΙ η σύγκριση μπορεί να καλύπτει λιγότερες μέρες απ' όσες
    αξιολόγησε το hour-specific μοντέλο -- ΔΕΝ είναι bug, είναι κενό στα
    διαθέσιμα δεδομένα του {column} (π.χ. ημιτελές run του αντίστοιχου
    multi-hour rolling_evaluation script)."""
    our_days = set(preds["timestamp"])
    reg_days = set(regular["timestamp"])
    missing = sorted(our_days - reg_days)
    print(f"Ημέρες hour-specific αξιολογήθηκαν: {len(our_days)}")
    print(f"Ημέρες με {column} διαθέσιμο (ίδια ώρα): {len(reg_days & our_days)}")
    if missing:
        print(f"[Προσοχή] Λείπουν {len(missing)} ημέρες από το {column} μέσα στο eval window αυτού "
              f"του script -- {missing[0].date()} έως {missing[-1].date()}. Αυτό ΔΕΝ είναι bug της "
              f"σύγκρισης· σημαίνει ότι το rolling_evaluation_hybrid{'_v2' if column.endswith('v2') else ''}.py "
              f"δεν έχει (ακόμα) προβλέψεις για αυτές τις μέρες στο vol2_final.csv -- ξανατρέξτο.")


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


def run_hour3_rolling(df_h3: pd.DataFrame, feature_cols: list,
                       hour1_preds: pd.Series, hour2_preds: pd.Series) -> pd.DataFrame:
    eval_start = pd.Timestamp(EVAL_START)
    eval_end = min(pd.Timestamp(EVAL_END), df_h3["timestamp"].max())
    if pd.Timestamp(EVAL_END) > df_h3["timestamp"].max():
        print(f"[Προσοχή] EVAL_END ({EVAL_END}) > μέγιστη διαθέσιμη ημερομηνία ώρας-3 "
              f"({df_h3['timestamp'].max().date()}). Κόβεται αυτόματα.")

    df_h3 = df_h3.set_index("timestamp").sort_index()
    records = []
    test_days = pd.date_range(eval_start, eval_end, freq="D")

    for i, day in enumerate(test_days, start=1):
        train_start = day - pd.DateOffset(months=MONTHS_USED)
        train = df_h3.loc[train_start:day - pd.Timedelta(seconds=1)]

        test_timestamp = day + pd.Timedelta(hours=3)
        if test_timestamp not in df_h3.index:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (λείπει η ώρα 3)")
            continue
        if len(train) < MIN_TRAIN_DAYS:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (ελλιπές training, {len(train)} ημέρες)")
            continue
        if day not in hour1_preds.index or day not in hour2_preds.index:
            print(f"[{i}/{len(test_days)}] {day.date()} -> παραλείπεται (λείπει πρόβλεψη ώρας 1 ή 2)")
            continue

        X_train, y_train = train[feature_cols], train["mcp_eur_per_mwh"]

        X_test = df_h3.loc[[test_timestamp], feature_cols].copy()
        X_test["lag_1h"] = hour2_preds.loc[day]
        X_test["lag_2h"] = hour1_preds.loc[day]
        y_test = df_h3.loc[test_timestamp, "mcp_eur_per_mwh"]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = float(model.predict(X_test)[0])

        err = abs(y_pred - y_test)
        records.append({"timestamp": test_timestamp, "actual": y_test, "predicted": y_pred})
        print(f"[{i}/{len(test_days)}] {day.date()} 03:00 -> actual={y_test:.2f}  "
              f"pred={y_pred:.2f}  |err|={err:.2f}")

    return pd.DataFrame(records)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering (original base + lag_1h/lag_2h θορυβώδη)...")
    df = build_features(df)
    feature_cols = get_feature_columns(df)
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    print(f"Φόρτωση προβλέψεων ωρών 1 και 2 από: {VOL2_FINAL_PATH}")
    hour1_preds = load_hour_predictions(VOL2_FINAL_PATH, hour=1)
    hour2_preds = load_hour_predictions(VOL2_FINAL_PATH, hour=2)
    print(f"Διαθέσιμες προβλέψεις: ώρα1={len(hour1_preds)}  ώρα2={len(hour2_preds)}")

    df_h3 = df[df["hour"] == 3].copy()
    print(f"Γραμμές ώρας 03:00 διαθέσιμες: {len(df_h3)}")

    print(f"\nRolling ΩΡΑ-3 evaluation (original base) {EVAL_START} -> {EVAL_END}...\n")
    preds = run_hour3_rolling(df_h3, feature_cols, hour1_preds, hour2_preds)

    vol2_final = pd.DataFrame(columns=["timestamp", "hour_specific"])
    if preds.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε EVAL_START/EVAL_END/dataset/vol2_final.")
    else:
        mae = float(np.mean(np.abs(preds["predicted"] - preds["actual"])))
        rmse = float(np.sqrt(np.mean((preds["predicted"] - preds["actual"]) ** 2)))
        print("\n" + "=" * 50)
        print(f"ΩΡΑ 03:00 (hour-specific, original base) -- {len(preds)} ημέρες, {EVAL_START} -> {EVAL_END}")
        print("=" * 50)
        print(f"MAE  = {mae:.2f} EUR/MWh")
        print(f"RMSE = {rmse:.2f} EUR/MWh")

        regular = load_regular_model_predictions(VOL2_FINAL_PATH, REGULAR_MODEL_COLUMN, hour=3)
        print("\n" + "=" * 50)
        print(f"ΣΥΓΚΡΙΣΗ με το κανονικό μοντέλο ({REGULAR_MODEL_COLUMN})")
        print("=" * 50)
        diagnose_regular_model_coverage(preds, regular, REGULAR_MODEL_COLUMN)
        merged = preds.merge(regular, on="timestamp", how="inner")
        if merged.empty:
            print("Δεν βρέθηκαν προβλέψεις του κανονικού μοντέλου για σύγκριση.")
        else:
            reg_mae = float((merged[REGULAR_MODEL_COLUMN] - merged["actual"]).abs().mean())
            reg_rmse = float(((merged[REGULAR_MODEL_COLUMN] - merged["actual"]) ** 2).mean() ** 0.5)
            hs_mae = float((merged["predicted"] - merged["actual"]).abs().mean())
            hs_rmse = float(((merged["predicted"] - merged["actual"]) ** 2).mean() ** 0.5)
            print(f"\nΚανονικό μοντέλο:   MAE={reg_mae:.2f}  RMSE={reg_rmse:.2f}")
            print(f"Hour-specific:       MAE={hs_mae:.2f}  RMSE={hs_rmse:.2f}")
            print(f"Διαφορά MAE:  {reg_mae - hs_mae:+.2f} EUR/MWh "
                  f"({'βελτίωση' if hs_mae < reg_mae else 'χειρότερο'})")
            print(f"Διαφορά RMSE: {reg_rmse - hs_rmse:+.2f} EUR/MWh "
                  f"({'βελτίωση' if hs_rmse < reg_rmse else 'χειρότερο'})")

        out_path = DATASET_PATH.parent / "rolling_evaluation_hour3_original_predictions.csv"
        preds.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν οι αναλυτικές προβλέψεις: {out_path}")

        vol2_final = preds[["timestamp", "predicted"]].rename(columns={"predicted": "hour_specific"})

    vol2_final = merge_into_vol2_final(vol2_final, VOL2_FINAL_PATH)
    vol2_final.to_csv(VOL2_FINAL_PATH, index=False)
    print(f"\nΑποθηκεύτηκε/ενημερώθηκε το vol2_final: {VOL2_FINAL_PATH}")
    print(f"  hour_specific: {vol2_final['hour_specific'].notna().sum()} προβλέψεις συνολικά (σωρευτικά)")