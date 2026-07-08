"""
hour_bucket_model.py
======================
Εναλλακτική στο two_stage_model.py: αντί για μαθημένο classifier, η
δρομολόγηση σε ξεχωριστό μοντέλο γίνεται ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΑ από την ίδια την
ώρα της ημέρας (πάντα γνωστή εκ των προτέρων, μηδενικό ρίσκο λάθους
ταξινόμησης):

    - LOW_HOURS  (π.χ. 8, 9, 10)   -> ξεχωριστό μοντέλο για χαμηλές τιμές
    - HIGH_HOURS (π.χ. 19, 20, 21) -> ξεχωριστό μοντέλο για υψηλές τιμές
    - υπόλοιπες ώρες               -> το "κανονικό" μοντέλο

Περιλαμβάνει επίσης σύγκριση με ΑΠΛΟ dummy baseline (μέση τιμή του
training σε κάθε ζώνη) ειδικά για τις ζώνες LOW/HIGH -- αν το XGBoost δεν
ξεπερνά καν αυτό το dummy εκεί, αυτό είναι ισχυρή ένδειξη ότι το πρόβλημα
είναι στα ίδια τα features/δεδομένα, όχι στο μοντέλο.

ΣΗΜΕΙΩΣΗ: οι ζώρες LOW/HIGH ορίστηκαν εμπειρικά (τυπική συμπεριφορά
ζήτησης/φορτίου). Δεν καλύπτουν απαραίτητα ΟΛΑ τα ακραία περιστατικά
(π.χ. το spike της 30/6/2026 έγινε στις 8:00, μέσα στη ζώνη "χαμηλού
φορτίου" -- πιθανό να προκλήθηκε από την πλευρά της προσφοράς, όχι
ζήτησης). Γι' αυτό παράγονται και τα histograms ώρας μεγίστου/ελαχίστου,
ώστε να ελέγχεται οπτικά πόσο καλά καλύπτουν οι ζώνες την πραγματικότητα.
"""

import numpy as np
import pandas as pd
import xgboost as xgb

from rolling_evaluation import (
    DATASET_PATH,
    EVAL_END,
    EVAL_START,
    EVAL_YEAR,
    MIN_TRAIN_HOURS,
    MONTHS_USED,
    SEASON_MAP,
    build_features,
    plot_mae_distribution,
    plot_month_actual_vs_predicted,
    summarize,
)
from spike_analysis import hour_of_daily_extreme_histogram

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
LOW_HOURS = {10, 11, 12}
HIGH_HOURS = {16, 17}

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


def _bucket(hour: int) -> str:
    if hour in LOW_HOURS:
        return "low"
    if hour in HIGH_HOURS:
        return "high"
    return "normal"


def run_hour_bucket_evaluation(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-11-01")
    eval_end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-11-30")

    results = []
    hourly_records = []
    # Συλλέγουμε επίσης το σφάλμα του dummy baseline (μέση τιμή training),
    # ξεχωριστά ανά ζώνη, για τη σύγκριση XGBoost-vs-dummy.
    dummy_errors = {"low": [], "high": []}
    model_errors = {"low": [], "high": []}

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

        df_train["bucket"] = df_train["hour"].apply(_bucket)
        df_test["bucket"] = df_test["hour"].apply(_bucket)

        y_pred_full = np.zeros(len(df_test))
        test_bucket_arr = df_test["bucket"].values

        for bucket in ("low", "high", "normal"):
            train_mask = df_train["bucket"] == bucket
            test_mask = test_bucket_arr == bucket
            if test_mask.sum() == 0:
                continue

            X_train_b = df_train.loc[train_mask, feature_cols]
            y_train_b = df_train.loc[train_mask, "mcp_eur_per_mwh"]
            X_test_b = df_test.loc[test_mask, feature_cols]
            y_test_b = df_test.loc[test_mask, "mcp_eur_per_mwh"]

            model = xgb.XGBRegressor(**REGRESSOR_PARAMS)
            model.fit(X_train_b, y_train_b)
            pred_b = model.predict(X_test_b)
            y_pred_full[test_mask] = pred_b

            if bucket in ("low", "high"):
                dummy_pred = np.full(len(y_test_b), y_train_b.mean())
                dummy_errors[bucket].extend(np.abs(dummy_pred - y_test_b.values).tolist())
                model_errors[bucket].extend(np.abs(pred_b - y_test_b.values).tolist())

        y_test_full = df_test["mcp_eur_per_mwh"].values
        mae = float(np.mean(np.abs(y_pred_full - y_test_full)))

        results.append({
            "date": current_day.date(),
            "month": current_day.month,
            "season": SEASON_MAP[current_day.month],
            "mae": mae,
        })

        for ts, actual, pred in zip(df_test["timestamp"].values, y_test_full, y_pred_full):
            hourly_records.append({"timestamp": ts, "actual": actual, "predicted": pred})

        print(f"[{i}/{n_days}] {current_day.date()} -> MAE: {mae:.2f}")

        current_day += pd.Timedelta(days=1)

    print("\n" + "=" * 50)
    print("XGBoost vs Dummy (μέση τιμή training) -- ανά ζώνη")
    print("=" * 50)
    for bucket in ("low", "high"):
        if model_errors[bucket]:
            print(f"{bucket.upper()}: XGBoost MAE = {np.mean(model_errors[bucket]):.2f}"
                  f"   |   Dummy MAE = {np.mean(dummy_errors[bucket]):.2f}")

    return pd.DataFrame(results), pd.DataFrame(hourly_records)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering...")
    df = build_features(df)

    print(f"Hour-bucket rolling evaluation...\n")
    results, hourly = run_hour_bucket_evaluation(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε το EVAL_YEAR/EVAL_START/EVAL_END/dataset.")
    else:
        summarize(results)
        out_path = DATASET_PATH.parent / f"hour_bucket_evaluation_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")

        hourly_path = DATASET_PATH.parent / f"hour_bucket_evaluation_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"Αποθηκεύτηκαν τα ωριαία αποτελέσματα: {hourly_path}")

        plot_path = DATASET_PATH.parent / f"hour_bucket_mae_distribution_{EVAL_YEAR}.png"
        plot_mae_distribution(results, MAE_THRESHOLD, plot_path)

        month_plot_path = DATASET_PATH.parent / f"hour_bucket_actual_vs_predicted_month_{EVAL_YEAR}.png"
        plot_month_actual_vs_predicted(hourly, results, EVAL_YEAR, PLOT_MONTH, month_plot_path)

        max_hist_path = DATASET_PATH.parent / "hour_of_daily_max_histogram.png"
        hour_of_daily_extreme_histogram(hourly, "max", max_hist_path)

        min_hist_path = DATASET_PATH.parent / "hour_of_daily_min_histogram.png"
        hour_of_daily_extreme_histogram(hourly, "min", min_hist_path)
