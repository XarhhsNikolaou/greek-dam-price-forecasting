"""
oracle_gpd_diagnostic.py
===========================
ΔΙΑΓΝΩΣΤΙΚΟ (oracle) τεστ -- ΟΧΙ πραγματικό, deployable μοντέλο.

Ιδέα: τρέχουμε το ΚΛΑΣΙΚΟ, ενιαίο XGBoost regressor (χωρίς κανένα
classifier/ζώνες ώρας) σε ΟΛΕΣ τις ώρες μιας μέρας. Εντοπίζουμε τις ώρες
όπου η απόκλιση |πρόβλεψη - πραγματική| ξεπερνά ένα όριο (π.χ. 50 €/MWh)
-- ΧΡΗΣΙΜΟΠΟΙΩΝΤΑΣ την πραγματική τιμή, κάτι που ΔΕΝ θα ήταν διαθέσιμο σε
πραγματικό day-ahead σενάριο (data leakage, σκόπιμο εδώ).

Σε αυτές τις "oracle-identified" ώρες, αντικαθιστούμε την πρόβλεψη με
conditional GPD point estimate, και μετράμε πόσο βελτιώνεται το MAE.

Σκοπός: να δούμε το ΤΑΒΑΝΙ βελτίωσης που θα μπορούσε να προσφέρει το GPD
--   ΑΝ είχαμε τέλειο τρόπο να εντοπίζουμε εκ των προτέρων ποιες ώρες θα
"αστοχήσει" το κλασικό μοντέλο. Αν δείξει σημαντική βελτίωση, αξίζει να
επενδύσουμε σε ΜΗ-διαρρέοντα τρόπο εντοπισμού (π.χ. feature σφιχτότητας
συστήματος). Αν δεν βοηθήσει ούτε καν εδώ, δεν αξίζει να συνεχίσουμε προς
αυτή την κατεύθυνση.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from conditional_gpd import fit_conditional_gpd, predict_point_price
from rolling_evaluation import (
    DATASET_PATH,
    EVAL_END,
    EVAL_START,
    EVAL_YEAR,
    MIN_TRAIN_HOURS,
    MONTHS_USED,
    SEASON_MAP,
    build_features,
    summarize,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
MISS_THRESHOLD = 50            # |error| πάνω από αυτό -> "oracle miss"
EXTREME_PERCENTILE_HIGH = 95
EXTREME_PERCENTILE_LOW = 5
MIN_EXTREME_SAMPLES = 30

BASELINE_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=0.5,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)


def _standardize(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean()
    std = X_train.std().replace(0, 1.0)
    return ((X_train - mean) / std).values, ((X_test - mean) / std).values


def run_oracle_diagnostic(df: pd.DataFrame, eval_year: int) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-10-01")
    eval_end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-10-31")

    results = []
    current_day = eval_start
    n_days = (eval_end - eval_start).days + 1
    i = 0
    total_oracle_misses = 0
    total_hours_evaluated = 0

    while current_day <= eval_end:
        i += 1
        test_start = current_day
        test_end = current_day + pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        train_start = train_end - pd.Timedelta(hours=train_hours - 1)

        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)]
        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)]

        if len(df_test) < 24 or len(df_train) < MIN_TRAIN_HOURS:
            print(f"[{i}/{n_days}] {current_day.date()} -> παραλείπεται (ελλιπή δεδομένα)")
            current_day += pd.Timedelta(days=1)
            continue

        X_train = df_train[feature_cols]
        y_train = df_train["mcp_eur_per_mwh"]
        X_test = df_test[feature_cols]
        y_test = df_test["mcp_eur_per_mwh"].values

        # --- Βήμα 1: ΚΛΑΣΙΚΟ, ενιαίο XGBoost σε όλες τις ώρες ---
        baseline_model = xgb.XGBRegressor(**BASELINE_PARAMS)
        baseline_model.fit(X_train, y_train)
        y_pred_baseline = baseline_model.predict(X_test)

        baseline_mae = float(np.mean(np.abs(y_pred_baseline - y_test)))

        # --- Βήμα 2 (ORACLE, χρησιμοποιεί πραγματική τιμή -- ΜΟΝΟ ΔΙΑΓΝΩΣΤΙΚΑ) ---
        abs_error = np.abs(y_pred_baseline - y_test)
        oracle_miss_mask = abs_error > MISS_THRESHOLD
        n_oracle_misses = int(oracle_miss_mask.sum())
        total_oracle_misses += n_oracle_misses
        total_hours_evaluated += len(y_test)

        y_pred_corrected = y_pred_baseline.copy()

        if n_oracle_misses > 0:
            threshold_high = np.percentile(y_train, EXTREME_PERCENTILE_HIGH)
            threshold_low = np.percentile(y_train, EXTREME_PERCENTILE_LOW)

            high_mask_train = y_train.values >= threshold_high
            low_mask_train = y_train.values <= threshold_low

            for idx in np.where(oracle_miss_mask)[0]:
                actual_val = y_test[idx]
                x_row = X_test.iloc[[idx]]

                if actual_val >= threshold_high and high_mask_train.sum() >= MIN_EXTREME_SAMPLES:
                    Xs_train_h, xs_row = _standardize(X_train[high_mask_train], x_row)
                    y_exceed_h = y_train[high_mask_train].values - threshold_high
                    fit_h = fit_conditional_gpd(Xs_train_h, y_exceed_h)
                    pred = predict_point_price(threshold_high, xs_row, fit_h["beta"], fit_h["shape"])[0]
                    y_pred_corrected[idx] = pred

                elif actual_val <= threshold_low and low_mask_train.sum() >= MIN_EXTREME_SAMPLES:
                    Xs_train_l, xs_row = _standardize(X_train[low_mask_train], x_row)
                    y_exceed_l = threshold_low - y_train[low_mask_train].values
                    fit_l = fit_conditional_gpd(Xs_train_l, y_exceed_l)
                    pred_exceed = predict_point_price(0, xs_row, fit_l["beta"], fit_l["shape"])[0]
                    y_pred_corrected[idx] = threshold_low - pred_exceed
                # else: η αστοχία δεν αντιστοιχεί σε καμία από τις δύο ουρές
                # (σπάνιο) -- κρατάμε την αρχική πρόβλεψη XGBoost.

        corrected_mae = float(np.mean(np.abs(y_pred_corrected - y_test)))

        results.append({
            "date": current_day.date(),
            "month": current_day.month,
            "season": SEASON_MAP[current_day.month],
            "baseline_mae": baseline_mae,
            "corrected_mae": corrected_mae,
            "n_oracle_misses": n_oracle_misses,
            "mae": corrected_mae,  # για συμβατότητα με summarize()
        })

        flag = f"  <- {n_oracle_misses} oracle misses διορθώθηκαν" if n_oracle_misses > 0 else ""
        print(f"[{i}/{n_days}] {current_day.date()} -> baseline: {baseline_mae:.2f}, "
              f"corrected: {corrected_mae:.2f}{flag}")

        current_day += pd.Timedelta(days=1)

    print(f"\nΣυνολικά oracle misses: {total_oracle_misses} / {total_hours_evaluated} ώρες "
          f"({100 * total_oracle_misses / total_hours_evaluated:.2f}%)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering...")
    df = build_features(df)

    print("Oracle GPD diagnostic...\n")
    results = run_oracle_diagnostic(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε.")
    else:
        print("\n" + "=" * 60)
        print("ΣΥΝΟΨΗ")
        print("=" * 60)
        print(f"Μέσο baseline MAE (χωρίς GPD διόρθωση):  {results['baseline_mae'].mean():.2f}")
        print(f"Μέσο corrected MAE (με oracle GPD):      {results['corrected_mae'].mean():.2f}")
        print(f"Βελτίωση:                                {results['baseline_mae'].mean() - results['corrected_mae'].mean():+.2f}")
        print(f"\nΜέρες με τουλάχιστον 1 oracle miss: {(results['n_oracle_misses'] > 0).sum()} / {len(results)}")

        out_path = DATASET_PATH.parent / f"oracle_gpd_diagnostic_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")