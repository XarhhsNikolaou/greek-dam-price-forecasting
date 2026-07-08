"""
two_stage_model.py
====================
Two-stage προσέγγιση για το πρόβλημα των spikes:

    1. Classifier: για κάθε ώρα, προβλέπει αν θα είναι "ακραία" τιμή
       (top EXTREME_PERCENTILE% της training περιόδου) ή "κανονική".
    2. Δύο ξεχωριστά regressors:
        - regressor_normal: εκπαιδευμένο ΜΟΝΟ σε κανονικές ώρες
        - regressor_extreme: εκπαιδευμένο ΜΟΝΟ σε ακραίες ώρες
    3. Το τελικό point forecast προέρχεται από τον regressor που ταιριάζει
       με την απόφαση του classifier για κάθε ώρα.

Ίδια λογική rolling evaluation με το rolling_evaluation.py (ρεαλιστικό
day-ahead σενάριο, νέο μοντέλο κάθε μέρα, μόνο ιστορικά δεδομένα).
"""

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
    plot_mae_distribution,
    plot_month_actual_vs_predicted,
    summarize,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EXTREME_PERCENTILE = 95     # top 5% -> "ακραία" ώρα (ίδιο όριο με το spike_analysis.py)
MIN_EXTREME_SAMPLES = 30    # ελάχιστο πλήθος ακραίων δειγμάτων για να εκπαιδευτεί
                            # ξεχωριστός regressor_extreme· αλλιώς fallback

CLASSIFIER_PARAMS = dict(
    objective="binary:logistic",
    max_depth=4,
    learning_rate=0.1,
    n_estimators=150,
)

REGRESSOR_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=0.5,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

MAE_THRESHOLD = 14
PLOT_MONTH = None


def run_two_stage_evaluation(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start = pd.Timestamp(f"{eval_year}-01-01")
    eval_end = pd.Timestamp(f"{eval_year}-12-31")

    results = []
    hourly_records = []
    current_day = eval_start
    n_days = (eval_end - eval_start).days + 1
    i = 0
    n_fallback_days = 0

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
        y_test = df_test["mcp_eur_per_mwh"]

        # --- Στάδιο 1: όριο "ακραίας" τιμής + classifier ---
        threshold = np.percentile(y_train, EXTREME_PERCENTILE)
        is_extreme_train = (y_train >= threshold).astype(int)

        use_two_stage = is_extreme_train.sum() >= MIN_EXTREME_SAMPLES

        if use_two_stage:
            classifier = xgb.XGBClassifier(**CLASSIFIER_PARAMS)
            classifier.fit(X_train, is_extreme_train)
            predicted_extreme = classifier.predict(X_test).astype(bool)

            reg_normal = xgb.XGBRegressor(**REGRESSOR_PARAMS)
            reg_normal.fit(X_train[is_extreme_train == 0], y_train[is_extreme_train == 0])

            reg_extreme = xgb.XGBRegressor(**REGRESSOR_PARAMS)
            reg_extreme.fit(X_train[is_extreme_train == 1], y_train[is_extreme_train == 1])

            pred_normal_all = reg_normal.predict(X_test)
            pred_extreme_all = reg_extreme.predict(X_test)
            y_pred = np.where(predicted_extreme, pred_extreme_all, pred_normal_all)
        else:
            n_fallback_days += 1
            reg_fallback = xgb.XGBRegressor(**REGRESSOR_PARAMS)
            reg_fallback.fit(X_train, y_train)
            y_pred = reg_fallback.predict(X_test)

        mae = float(np.mean(np.abs(y_pred - y_test.values)))

        results.append({
            "date": current_day.date(),
            "month": current_day.month,
            "season": SEASON_MAP[current_day.month],
            "mae": mae,
            "used_two_stage": use_two_stage,
        })

        for ts, actual, pred in zip(df_test["timestamp"].values, y_test.values, y_pred):
            hourly_records.append({"timestamp": ts, "actual": actual, "predicted": pred})

        print(f"[{i}/{n_days}] {current_day.date()} -> MAE: {mae:.2f}"
              f"{'' if use_two_stage else '  (fallback: λίγα extreme δείγματα)'}")

        current_day += pd.Timedelta(days=1)

    print(f"\nΜέρες με fallback στο απλό μοντέλο: {n_fallback_days}/{i}")
    return pd.DataFrame(results), pd.DataFrame(hourly_records)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering...")
    df = build_features(df)

    print(f"Two-stage rolling evaluation για το έτος {EVAL_YEAR}...\n")
    results, hourly = run_two_stage_evaluation(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε το EVAL_YEAR/dataset.")
    else:
        summarize(results)
        out_path = DATASET_PATH.parent / f"two_stage_evaluation_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")

        hourly_path = DATASET_PATH.parent / f"two_stage_evaluation_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"Αποθηκεύτηκαν τα ωριαία αποτελέσματα: {hourly_path}")

        plot_path = DATASET_PATH.parent / f"two_stage_mae_distribution_{EVAL_YEAR}.png"
        plot_mae_distribution(results, MAE_THRESHOLD, plot_path)

        month_plot_path = DATASET_PATH.parent / f"two_stage_actual_vs_predicted_month_{EVAL_YEAR}.png"
        plot_month_actual_vs_predicted(hourly, results, EVAL_YEAR, PLOT_MONTH, month_plot_path)