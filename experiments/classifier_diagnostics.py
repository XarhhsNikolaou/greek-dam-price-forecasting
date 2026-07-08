"""
classifier_diagnostics.py
============================
Διαγνωστικό: τι features χρησιμοποιεί σήμερα ο classifier (low/normal/high)
για να αποφασίσει, και πόσο βαρύνει το καθένα -- πριν αποφασίσουμε τι νέο
feature ή αλλαγή χρειάζεται.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

import two_stage_gpd_model as gpdm
from rolling_evaluation import build_features

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REFERENCE_DATE = "2025-10-01"  # τέλος του training window που εξετάζουμε
MONTH_FOR_CANDIDATE_HOURS = 10  # ποιες υποψήφιες ώρες/μήνα να χρησιμοποιήσουμε


def inspect_classifier(df: pd.DataFrame, reference_date: str, month: int) -> None:
    train_hours = gpdm.MONTHS_USED * 30 * 24
    train_end = pd.Timestamp(reference_date) - pd.Timedelta(hours=1)
    train_start = train_end - pd.Timedelta(hours=train_hours - 1)

    df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)].copy()
    df_train["hour_extreme_freq"] = df_train.apply(
        lambda r: gpdm.HOUR_EXTREME_FREQUENCY[int(r["month"])][int(r["hour"])], axis=1
    )

    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    feature_cols_full = feature_cols + ["hour_extreme_freq"]

    X_train = df_train[feature_cols_full]
    y_train = df_train["mcp_eur_per_mwh"]

    threshold_high = y_train.quantile(gpdm.EXTREME_PERCENTILE_HIGH / 100)
    threshold_low = y_train.quantile(gpdm.EXTREME_PERCENTILE_LOW / 100)

    label_train = (y_train >= threshold_high).astype(int) * 2 + (y_train <= threshold_low).astype(int) * 1
    label_train = label_train.values

    candidate_hours = gpdm.CANDIDATE_HOURS_BY_MONTH.get(month)
    if candidate_hours is not None:
        cand_mask = df_train["hour"].isin(candidate_hours).values
        X_fit, y_fit = X_train[cand_mask], label_train[cand_mask]
    else:
        X_fit, y_fit = X_train, label_train

    print(f"Δείγματα εκπαίδευσης classifier: {len(X_fit)}")
    print(f"Κατανομή κλάσεων: {pd.Series(y_fit).value_counts().to_dict()}")

    classifier = xgb.XGBClassifier(**gpdm.CLASSIFIER_PARAMS)
    classifier.fit(X_fit, y_fit)

    importances = pd.Series(classifier.feature_importances_, index=feature_cols_full)
    importances = importances.sort_values(ascending=False)

    print("\nFeature importance (φθίνουσα σειρά):")
    print(importances.to_string())

    plt.figure(figsize=(10, 8))
    importances.plot(kind="barh", color="steelblue")
    plt.gca().invert_yaxis()
    plt.title(f"Feature importance -- classifier (μήνας {month}, reference {reference_date})")
    plt.xlabel("Importance (gain)")
    plt.tight_layout()
    out_path = Path(gpdm.DATASET_PATH).parent / "classifier_feature_importance.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nΑποθηκεύτηκε: {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {gpdm.DATASET_PATH}")
    df = pd.read_csv(gpdm.DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = build_features(df)

    inspect_classifier(df, REFERENCE_DATE, MONTH_FOR_CANDIDATE_HOURS)