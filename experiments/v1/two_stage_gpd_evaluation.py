"""
two_stage_gpd_model.py
=========================
Πλήρες two-stage μοντέλο πρόβλεψης, με GPD ειδικά για τις ουρές:

    1. Classifier (3 κλάσεις): "low" (κάτω από bottom X%), "high" (πάνω
       από top X%), "normal" (ενδιάμεσα) -- βασισμένο σε πραγματικά
       percentiles τιμής, ΟΧΙ σε ώρα ημέρας (σε αντίθεση με το
       hour_bucket_model.py). Αυτό ταιριάζει καλύτερα με τη φύση του GPD,
       που χρειάζεται πραγματικές υπερβάσεις τιμής, όχι ώρες.
    2. Ανάλογα με την κλάση:
        - "high"   -> conditional GPD (upper tail) για το point estimate
        - "low"    -> conditional GPD (lower tail, mirrored) για το point estimate
        - "normal" -> απλό XGBoost regressor (reg:quantileerror, a=0.5)

Ίδια rolling day-ahead λογική με τα προηγούμενα scripts.
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
    plot_mae_distribution,
    plot_month_actual_vs_predicted,
    summarize,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
EXTREME_PERCENTILE_HIGH = 95   # top 5% -> "high"
EXTREME_PERCENTILE_LOW = 5     # bottom 5% -> "low"
MIN_EXTREME_SAMPLES = 30       # ελάχιστο πλήθος δειγμάτων ανά ακραία κλάση

CLASSIFIER_PARAMS = dict(
    objective="multi:softmax",
    num_class=3,               # 0=normal, 1=low, 2=high
    max_depth=4,
    learning_rate=0.1,
    n_estimators=150,
)

NORMAL_REGRESSOR_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=0.5,
    max_depth=5,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

MAE_THRESHOLD = 14
PLOT_MONTH = None


def _standardize(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean()
    std = X_train.std().replace(0, 1.0)
    return ((X_train - mean) / std).values, ((X_test - mean) / std).values


def run_gpd_evaluation(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-10-01")
    eval_end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-10-31")

    results = []
    hourly_records = []
    current_day = eval_start
    n_days = (eval_end - eval_start).days + 1
    i = 0
    n_gpd_fallback = 0

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

        # --- Ορισμός ορίων + 3-class ετικέτες (0=normal, 1=low, 2=high) ---
        threshold_high = np.percentile(y_train, EXTREME_PERCENTILE_HIGH)
        threshold_low = np.percentile(y_train, EXTREME_PERCENTILE_LOW)

        label_train = np.zeros(len(y_train), dtype=int)
        label_train[y_train.values >= threshold_high] = 2
        label_train[y_train.values <= threshold_low] = 1

        n_high = (label_train == 2).sum()
        n_low = (label_train == 1).sum()
        use_gpd_high = n_high >= MIN_EXTREME_SAMPLES
        use_gpd_low = n_low >= MIN_EXTREME_SAMPLES

        # --- Classifier ---
        classifier = xgb.XGBClassifier(**CLASSIFIER_PARAMS)
        classifier.fit(X_train, label_train)
        predicted_label = classifier.predict(X_test)

        # --- Normal regressor (πάντα εκπαιδεύεται στα "normal" δεδομένα) ---
        normal_mask = label_train == 0
        reg_normal = xgb.XGBRegressor(**NORMAL_REGRESSOR_PARAMS)
        reg_normal.fit(X_train[normal_mask], y_train[normal_mask])
        pred_normal_all = reg_normal.predict(X_test)

        y_pred = pred_normal_all.copy()

        # --- HIGH tail: conditional GPD (ή fallback σε XGBoost αν λίγα δείγματα) ---
        if use_gpd_high:
            high_mask = label_train == 2
            Xs_train_h, Xs_test_h = _standardize(X_train[high_mask], X_test)
            y_exceed_h = y_train[high_mask].values - threshold_high
            fit_h = fit_conditional_gpd(Xs_train_h, y_exceed_h)
            pred_high_all = predict_point_price(threshold_high, Xs_test_h, fit_h["beta"], fit_h["shape"])
        else:
            n_gpd_fallback += 1
            reg_high = xgb.XGBRegressor(**NORMAL_REGRESSOR_PARAMS)
            reg_high.fit(X_train, y_train)
            pred_high_all = reg_high.predict(X_test)

        # --- LOW tail: conditional GPD (mirrored) ---
        if use_gpd_low:
            low_mask = label_train == 1
            Xs_train_l, Xs_test_l = _standardize(X_train[low_mask], X_test)
            y_exceed_l = threshold_low - y_train[low_mask].values
            fit_l = fit_conditional_gpd(Xs_train_l, y_exceed_l)
            pred_low_exceed = predict_point_price(0, Xs_test_l, fit_l["beta"], fit_l["shape"])
            pred_low_all = threshold_low - pred_low_exceed
        else:
            reg_low = xgb.XGBRegressor(**NORMAL_REGRESSOR_PARAMS)
            reg_low.fit(X_train, y_train)
            pred_low_all = reg_low.predict(X_test)

        y_pred[predicted_label == 2] = pred_high_all[predicted_label == 2]
        y_pred[predicted_label == 1] = pred_low_all[predicted_label == 1]

        mae = float(np.mean(np.abs(y_pred - y_test.values)))

        results.append({
            "date": current_day.date(),
            "month": current_day.month,
            "season": SEASON_MAP[current_day.month],
            "mae": mae,
        })

        for ts, actual, pred in zip(df_test["timestamp"].values, y_test.values, y_pred):
            hourly_records.append({"timestamp": ts, "actual": actual, "predicted": pred})

        print(f"[{i}/{n_days}] {current_day.date()} -> MAE: {mae:.2f}")

        current_day += pd.Timedelta(days=1)

    print(f"\nΜέρες με fallback (λίγα ακραία δείγματα για GPD): {n_gpd_fallback}/{i}")
    return pd.DataFrame(results), pd.DataFrame(hourly_records)


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering...")
    df = build_features(df)

    print(f"GPD two-stage rolling evaluation...\n")
    results, hourly = run_gpd_evaluation(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε.")
    else:
        summarize(results)
        out_path = DATASET_PATH.parent / f"gpd_evaluation_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")

        hourly_path = DATASET_PATH.parent / f"gpd_evaluation_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"Αποθηκεύτηκαν τα ωριαία αποτελέσματα: {hourly_path}")

        plot_path = DATASET_PATH.parent / f"gpd_mae_distribution_{EVAL_YEAR}.png"
        plot_mae_distribution(results, MAE_THRESHOLD, plot_path)

        month_plot_path = DATASET_PATH.parent / f"gpd_actual_vs_predicted_month_{EVAL_YEAR}.png"
        plot_month_actual_vs_predicted(hourly, results, EVAL_YEAR, PLOT_MONTH, month_plot_path)