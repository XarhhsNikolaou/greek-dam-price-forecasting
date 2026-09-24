"""
hyperparameter_search.py
===========================
Grid/random search πάνω στα hyperparameters του XGBoost, χρησιμοποιώντας
το ΗΔΗ ΕΠΙΒΕΒΑΙΩΜΕΝΟ feature set (build_features() από το rolling_evaluation.py
-- ΧΩΡΙΣ τα load/residual-load/delta_price features, που δοκιμάστηκαν και
απορρίφθηκαν ως ασταθή/ασυνεπή σε πολλαπλά test windows).

ΓΙΑΤΙ ΠΟΛΛΑΠΛΑ TEST WINDOWS (όχι μόνο ένα):
Είδαμε ήδη ότι η αξιολόγηση σε ένα μόνο test window μπορεί να δώσει
παραπλανητικά αποτελέσματα (π.χ. τα load features φαίνονταν να βοηθάνε
πολύ στο H1 2026 αλλά χειροτέρευαν σταθερά στο H1/H2 2025). Το ίδιο ρίσκο
υπάρχει και στο hyperparameter tuning -- ένας συνδυασμός μπορεί να είναι
απλά "τυχερός" σε ένα συγκεκριμένο διάστημα. Γι' αυτό εδώ κάθε συνδυασμός
αξιολογείται σε 3 ΞΕΧΩΡΙΣΤΑ test windows, και κατατάσσεται με βάση τον
ΜΕΣΟ όρο MAE σε όλα -- πιο robust επιλογή, λιγότερο ρίσκο overfitting σε
ένα window.

ΧΡΗΣΗ:
    python hyperparameter_search.py

Τα αποτελέσματα αποθηκεύονται σε hyperparameter_search_results.csv, με τα
top-N να τυπώνονται στην οθόνη.
"""

import itertools
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from rolling_evaluation import DATASET_PATH, build_features

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
TRAIN_WINDOW_YEARS = 3

# 3 ξεχωριστά test windows -- ίδια λογική με τα multi-window tests που ήδη
# κάναμε σήμερα για exports/outages/load features. Άλλαξέ τα αν θες.
TEST_WINDOWS = [
    ("H1 2025", "2025-01-01", "2025-07-01"),
    ("H2 2025", "2025-07-01", "2026-01-01"),
    ("H1 2026", "2026-01-01", "2026-07-01"),
]

PARAM_GRID = {
    "max_depth": [3, 4, 5, 6, 7],
    "n_estimators": [150, 200, 250, 300, 350],
    "learning_rate": [0.05],
    "subsample": [1.0],
}

# Πλήρες grid = πολλές χιλιάδες συνδυασμοί -- πολύ αργό. Κάνουμε τυχαίο
# δείγμα (RandomizedSearch-style). Ανέβασε το N_SAMPLES αν έχεις χρόνο.
N_SAMPLES = 80
RANDOM_SEED = 42

OUTPUT_PATH = Path("hyperparameter_search_results.csv")
TOP_N_TO_PRINT = 15


def load_dataset() -> pd.DataFrame:
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    print("Feature engineering (build_features από rolling_evaluation.py)...")
    df = build_features(df)
    return df


def evaluate_params(df: pd.DataFrame, feature_cols: list[str], params: dict) -> dict:
    """Εκπαιδεύει/αξιολογεί ένα συνδυασμό hyperparameters σε ΚΑΘΕ test
    window ξεχωριστά, και επιστρέφει το MAE ανά window + τον μέσο όρο."""
    per_window_mae = {}
    for name, start_str, end_str in TEST_WINDOWS:
        test_start = pd.Timestamp(start_str)
        test_end = pd.Timestamp(end_str)
        train_start = test_start - pd.DateOffset(years=TRAIN_WINDOW_YEARS)

        train = df[(df["timestamp"] >= train_start) & (df["timestamp"] < test_start)]
        test = df[(df["timestamp"] >= test_start) & (df["timestamp"] < test_end)]

        if len(train) < 1000 or len(test) < 24:
            per_window_mae[name] = np.nan
            continue

        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_SEED,
            tree_method="hist",
            **params,
        )
        model.fit(train[feature_cols], train["mcp_eur_per_mwh"])
        pred = model.predict(test[feature_cols])
        per_window_mae[name] = mean_absolute_error(test["mcp_eur_per_mwh"], pred)

    result = {**params}
    for name, _, _ in TEST_WINDOWS:
        result[f"mae_{name}"] = per_window_mae[name]
    result["mae_avg"] = float(np.nanmean(list(per_window_mae.values())))
    return result


def run_search(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]

    keys = list(PARAM_GRID.keys())
    all_combos = list(itertools.product(*PARAM_GRID.values()))
    print(f"Πλήρες grid: {len(all_combos)} συνδυασμοί -- κάνω τυχαίο δείγμα {N_SAMPLES}.")

    random.seed(RANDOM_SEED)
    sampled = random.sample(all_combos, min(N_SAMPLES, len(all_combos)))

    results = []
    t0 = time.time()
    for i, combo in enumerate(sampled):
        params = dict(zip(keys, combo))
        res = evaluate_params(df, feature_cols, params)
        results.append(res)
        elapsed = time.time() - t0
        print(f"[{i + 1}/{len(sampled)}] {elapsed:.0f}s -- mae_avg={res['mae_avg']:.3f}  params={params}")

    results_df = pd.DataFrame(results).sort_values("mae_avg").reset_index(drop=True)
    return results_df


if __name__ == "__main__":
    df = load_dataset()
    results_df = run_search(df)

    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nΑποθηκεύτηκαν όλα τα αποτελέσματα: {OUTPUT_PATH}")

    print("\n" + "=" * 70)
    print(f"TOP {TOP_N_TO_PRINT} συνδυασμοί (κατά μέσο MAE στα {len(TEST_WINDOWS)} test windows)")
    print("=" * 70)
    cols_to_show = list(PARAM_GRID.keys()) + [f"mae_{n}" for n, _, _ in TEST_WINDOWS] + ["mae_avg"]
    print(results_df[cols_to_show].head(TOP_N_TO_PRINT).to_string(index=False))

    best = results_df.iloc[0]
    print("\nΚαλύτερος συνδυασμός:")
    for k in PARAM_GRID:
        print(f"  {k} = {best[k]}")
    print(f"  -> mae_avg = {best['mae_avg']:.3f}")