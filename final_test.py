"""
final_test.py
================
Η ΤΕΛΕΥΤΑΙΑ δοκιμή αυτής της σειράς πειραμάτων (βλ. README για πλήρες
ιστορικό): hybrid μοντέλο (zone model στις ώρες αιχμής + γενικό μοντέλο
σε όλες τις ώρες), με το ΖΩΝΗΣ μοντέλο να παίρνει επιπλέον 4 BINARY
(threshold) features που δεν έχει το γενικό μοντέλο:

    - is_max_24h_over    (1 αν rolling_max_24h > MAX_THRESHOLD)
    - is_max_1week_over  (1 αν rolling_max_1week > MAX_THRESHOLD)
    - is_min_24h_under   (1 αν rolling_min_24h < MIN_THRESHOLD)
    - is_min_1week_under (1 αν rolling_min_1week < MIN_THRESHOLD)

ΙΣΤΟΡΙΚΟ: η ΣΥΝΕΧΗΣ εκδοχή αυτών των features (rolling max/min ως raw
τιμές, όχι binary) δοκιμάστηκε πρώτα -- ασαφές/ασυνεπές αποτέλεσμα σε 3
μεμονωμένα test windows. Η υπόθεση εδώ: ίσως η συνεχής τιμή "μπερδεύει"
το μοντέλο σε ήσυχους μήνες (προσθέτει θόρυβο χωρίς αντίστοιχο όφελος),
ενώ ένα binary threshold θα "ενεργοποιείται" μόνο σε πραγματικά ακραίες
περιόδους. Σε προκαταρκτικό τεστ (πάλι 3 μεμονωμένα windows) το binary
νίκησε σε 2/3 αλλά έχασε αισθητά στο τρίτο -- ίδιο, ασυνεπές μοτίβο.
Αυτό το script τρέχει το ΠΛΉΡΕΣ rolling evaluation για την οριστική
απάντηση.

ΣΗΜΑΝΤΙΚΟ (leakage): anchor = shift(24) παντού, ίδιο invariant με το
build_features() του rolling_evaluation.py -- ασφαλές για ΚΑΘΕ ώρα της
προβλεπόμενης μέρας D (βλ. εκεί για πλήρη εξήγηση του leakage bug που
βρέθηκε και διορθώθηκε σήμερα).

ΜΕΤΑ ΑΠΟ ΑΥΤΟ: το σημερινό session κλείνει, ό,τι κι αν δείξει αυτό το
αποτέλεσμα. Επόμενο βήμα (στο τέλος του έτους, με πλήρη 3ετή δεδομένα σε
όλους τους μήνες): έλεγχος αν το 3ετές training window φέρνει πιο κοντά
στον στόχο MAE=14 σε σχέση με το 2ετές, plus νέες ιδέες (καιρός,
ελληνικές αργίες, συστηματικό hyperparameter tuning -- βλ. README
ROADMAP).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from rolling_evaluation import (
    DATASET_PATH,
    EVAL_YEAR,
    MIN_TRAIN_HOURS,
    MONTHS_USED,
    SEASON_MAP,
    build_features,
    summarize,
)

warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EVAL_START = None   # π.χ. "2025-01-01" -- None = 1 Ιανουαρίου του EVAL_YEAR
EVAL_END = None      # π.χ. "2025-12-31" -- None = 31 Δεκεμβρίου του EVAL_YEAR

PEAK_HOURS = {10, 11, 12, 18, 19, 20}
MAX_THRESHOLD = 180   # €/MWh -- πάνω από αυτό, θεωρείται "πρόσφατα ακραία υψηλή" περίοδος
MIN_THRESHOLD = 30    # €/MWh -- κάτω από αυτό, θεωρείται "πρόσφατα ακραία χαμηλή" περίοδος
ZONE_EXTRA_FEATURES = ["is_max_24h_over", "is_max_1week_over", "is_min_24h_under", "is_min_1week_under"]

XGB_PARAMS = dict(
    objective="reg:squarederror",
    max_depth=6,
    learning_rate=0.05,
    n_estimators=400,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method="hist",
)


def add_zone_extra_features(df: pd.DataFrame) -> pd.DataFrame:
    """Προσθέτει τα 4 binary threshold features πάνω στο ήδη διορθωμένο
    (leak-free) build_features() του rolling_evaluation.py. Anchor =
    shift(24), ίδιο invariant με όλα τα υπόλοιπα rolling features --
    ασφαλές για ΚΑΘΕ ώρα της προβλεπόμενης μέρας D."""
    df = df.copy()
    s24 = df["mcp_eur_per_mwh"].shift(24)
    rolling_max_24h = s24.rolling(24).max()
    rolling_min_24h = s24.rolling(24).min()
    rolling_max_1week = s24.rolling(24 * 7).max()
    rolling_min_1week = s24.rolling(24 * 7).min()

    df["is_max_24h_over"] = np.where(rolling_max_24h.isna(), np.nan, (rolling_max_24h > MAX_THRESHOLD).astype(float))
    df["is_max_1week_over"] = np.where(rolling_max_1week.isna(), np.nan, (rolling_max_1week > MAX_THRESHOLD).astype(float))
    df["is_min_24h_under"] = np.where(rolling_min_24h.isna(), np.nan, (rolling_min_24h < MIN_THRESHOLD).astype(float))
    df["is_min_1week_under"] = np.where(rolling_min_1week.isna(), np.nan, (rolling_min_1week < MIN_THRESHOLD).astype(float))

    # τα ενδιάμεσα rolling_max/min_* χρειάζονται μόνο για τον υπολογισμό
    # των binary flags -- δεν μπαίνουν οι ίδιες οι συνεχείς τιμές στο
    # μοντέλο εδώ (αυτό δοκιμάστηκε ξεχωριστά, βλ. ιστορικό στο docstring)
    return df.dropna(subset=["is_max_24h_over", "is_max_1week_over",
                              "is_min_24h_under", "is_min_1week_under"]).reset_index(drop=True)


def _resolve_eval_window(df: pd.DataFrame, eval_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-03-01")
    end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-03-31")
    max_available = df["timestamp"].max()
    if end > max_available:
        print(f"[Προσοχή] EVAL_END ({end.date()}) > μέγιστη διαθέσιμη ημερομηνία "
              f"στο dataset ({max_available.date()}). Κόβεται αυτόματα.")
        end = max_available.normalize()
    return start, end


def run_final_test(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    general_features = [c for c in df.columns if c not in
                         ("timestamp", "mcp_eur_per_mwh", "net_out") + tuple(ZONE_EXTRA_FEATURES)]
    zone_features = general_features + ZONE_EXTRA_FEATURES
    train_hours = MONTHS_USED * 30 * 24

    eval_start, eval_end = _resolve_eval_window(df, eval_year)

    results = []
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

        model_all = xgb.XGBRegressor(**XGB_PARAMS)
        model_all.fit(df_train[general_features], df_train["mcp_eur_per_mwh"])

        model_zone = None
        df_train_zone = df_train[is_zone_train]
        if len(df_train_zone) >= 100:
            model_zone = xgb.XGBRegressor(**XGB_PARAMS)
            model_zone.fit(df_train_zone[zone_features], df_train_zone["mcp_eur_per_mwh"])

        y_pred = model_all.predict(df_test[general_features])
        if model_zone is not None and is_zone_test.any():
            zone_rows = df_test[is_zone_test]
            y_pred[is_zone_test] = model_zone.predict(zone_rows[zone_features])

        y_test = df_test["mcp_eur_per_mwh"].values
        mae = float(np.mean(np.abs(y_pred - y_test)))

        results.append({
            "date": current_day.date(),
            "month": current_day.month,
            "season": SEASON_MAP[current_day.month],
            "mae": mae,
        })

        for ts, actual, pred, hr in zip(df_test["timestamp"].values, y_test, y_pred, df_test["hour"].values):
            hourly_records.append({"timestamp": ts, "actual": actual, "predicted": pred,
                                    "hour": hr, "in_zone": hr in PEAK_HOURS})

        print(f"[{i}/{n_days}] {current_day.date()} -> MAE: {mae:.2f}")
        current_day += pd.Timedelta(days=1)

    return pd.DataFrame(results), pd.DataFrame(hourly_records)


def summarize_zone(hourly: pd.DataFrame) -> None:
    print("\n--- MAE εντός vs εκτός ζώνης ωρών αιχμής ---")
    err = (hourly["predicted"] - hourly["actual"]).abs()
    print(f"MAE εντός ζώνης ({sorted(PEAK_HOURS)}): {err[hourly['in_zone']].mean():.2f}")
    print(f"MAE εκτός ζώνης: {err[~hourly['in_zone']].mean():.2f}")


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering (βασικό, leak-free)...")
    df = build_features(df)
    print("Feature engineering (extra rolling max/min, μόνο για το zone model)...")
    df = add_zone_extra_features(df)

    print(f"\nFinal test -- hybrid + zone-only rolling max/min, έτος {EVAL_YEAR}...\n")
    results, hourly = run_final_test(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε -- έλεγξε EVAL_YEAR/EVAL_START/EVAL_END/dataset.")
    else:
        summarize(results)
        summarize_zone(hourly)

        out_path = DATASET_PATH.parent / f"final_test_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")

        hourly_path = DATASET_PATH.parent / f"final_test_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"Αποθηκεύτηκαν τα ωριαία αποτελέσματα: {hourly_path}")