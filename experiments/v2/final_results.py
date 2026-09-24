"""
compute_final_predictions.py
==============================
Συνθέτει την ΤΕΛΙΚΗ πρόβλεψη ανά ώρα, διαβάζοντας τις ήδη υπολογισμένες
στήλες του vol2_final.csv, και υπολογίζει το τελικό MAE/RMSE για τον
τελευταίο χρόνο (2025-07-01 -> 2026-06-30), το οποίο δίνουμε ως
αποτέλεσμα.

Κανόνας σύνθεσης (ανά ώρα, ανά περίοδο):
  - Ώρες 00:00-02:00 (0,1,2):            ΠΑΝΤΑ hour-specific
  - Ώρες 03:00-05:00 (3,4,5):
      * Ιούλιος-Αύγουστος 2025 (original) -> hour-specific
      * Σεπτέμβριος 2025 - Ιούνιος 2026 (v2) -> κανονικό (predicted_hybrid_v2)
  - Ώρες 06:00-23:00:
      * Ιούλιος-Αύγουστος 2025 -> κανονικό (predicted_hybrid_original)
      * Σεπτέμβριος 2025 - Ιούνιος 2026 -> κανονικό (predicted_hybrid_v2)

Δηλαδή το "κανονικό" μοντέλο εναλλάσσεται ανάλογα με την περίοδο
(original για Ιούλ-Αύγ, v2 για Σεπ-Ιούν) -- ίδιο split με τα δύο
"κανονικά" μοντέλα σε όλα τα προηγούμενα scripts.

Γράφει μια νέα στήλη "final_prediction" στο vol2_final.csv (χωρίς να
πειράξει τις υπόλοιπες) και τυπώνει το συνολικό MAE/RMSE για το
2025-07-01 -> 2026-06-30, καθώς και breakdown ανά ζώνη ωρών/περίοδο για
έλεγχο κάλυψης.

Επιπλέον, γράφει ΞΕΧΩΡΙΣΤΟ αρχείο (FINAL_OUTPUT_PATH) με ΜΟΝΟ 2 στήλες
-- "actual" (mcp_eur_per_mwh) και "predicted" (final_prediction) -- για
όλες τις έγκυρες γραμμές του eval window.
"""

from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]  # repository root

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VOL2_FINAL_PATH = Path(str(REPO / "Dataset_Creation" / "processed" / "vol2_final.csv"))
FINAL_OUTPUT_PATH = VOL2_FINAL_PATH.parent / "final_actual_vs_predicted.csv"

FINAL_EVAL_START = "2025-07-01"
FINAL_EVAL_END = "2026-06-30"

ORIGINAL_PERIOD_START = "2025-07-01"
ORIGINAL_PERIOD_END = "2025-08-31 23:59:59"

HOUR_SPECIFIC_ALWAYS = {0, 1, 2}
HOUR_SPECIFIC_ORIGINAL_ONLY = {3, 4, 5}


def build_final_prediction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    in_original_period = df["timestamp"].between(ORIGINAL_PERIOD_START, ORIGINAL_PERIOD_END)
    regular = np.where(in_original_period, df["predicted_hybrid_original"], df["predicted_hybrid_v2"])

    hour = df["hour"]
    is_always_hs = hour.isin(HOUR_SPECIFIC_ALWAYS)
    is_original_only_hs = hour.isin(HOUR_SPECIFIC_ORIGINAL_ONLY) & in_original_period

    final_pred = np.where(
        is_always_hs | is_original_only_hs,
        df["hour_specific"],
        regular,
    )

    df["final_prediction"] = final_pred

    # ετικέτα πηγής, χρήσιμη για έλεγχο/debug
    source = np.select(
        [is_always_hs, is_original_only_hs],
        ["hour_specific", "hour_specific"],
        default="regular",
    )
    df["final_prediction_source"] = source
    return df


def report(df: pd.DataFrame) -> pd.DataFrame:
    window = df[df["timestamp"].between(FINAL_EVAL_START, FINAL_EVAL_END)].copy()
    print(f"Σύνολο γραμμών στο eval window ({FINAL_EVAL_START} -> {FINAL_EVAL_END}): {len(window)}")

    missing = window["final_prediction"].isna()
    if missing.any():
        print(f"[Προσοχή] {missing.sum()} γραμμές χωρίς final_prediction (κενές πηγές) -- εξαιρούνται από το MAE.")
        gaps = window.loc[missing, ["timestamp", "hour"]]
        print(gaps.groupby("hour").size().rename("missing_count"))

    valid = window.dropna(subset=["final_prediction", "mcp_eur_per_mwh"])
    err = valid["final_prediction"] - valid["mcp_eur_per_mwh"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())

    print("\n" + "=" * 60)
    print(f"ΤΕΛΙΚΟ ΑΠΟΤΕΛΕΣΜΑ -- {FINAL_EVAL_START} -> {FINAL_EVAL_END} ({len(valid)} γραμμές)")
    print("=" * 60)
    print(f"MAE  = {mae:.2f} EUR/MWh")
    print(f"RMSE = {rmse:.2f} EUR/MWh")

    print("\n--- Breakdown ανά πηγή πρόβλεψης ---")
    for src, sub in valid.groupby("final_prediction_source"):
        e = sub["final_prediction"] - sub["mcp_eur_per_mwh"]
        print(f"{src:15s}: n={len(sub):5d}  MAE={e.abs().mean():.2f}  RMSE={np.sqrt((e**2).mean()):.2f}")

    print("\n--- Breakdown ανά ζώνη ωρών ---")
    zone = np.select(
        [valid["hour"].isin([0, 1, 2]), valid["hour"].isin([3, 4, 5])],
        ["00-02 (hour-specific)", "03-05 (mixed)"],
        default="06-23 (regular)",
    )
    valid = valid.assign(zone=zone)
    for z, sub in valid.groupby("zone"):
        e = sub["final_prediction"] - sub["mcp_eur_per_mwh"]
        print(f"{z:24s}: n={len(sub):5d}  MAE={e.abs().mean():.2f}  RMSE={np.sqrt((e**2).mean()):.2f}")

    return valid


if __name__ == "__main__":
    print(f"Φόρτωση: {VOL2_FINAL_PATH}")
    df = pd.read_csv(VOL2_FINAL_PATH)
    df = build_final_prediction(df)

    valid = report(df)

    df.to_csv(VOL2_FINAL_PATH, index=False)
    print(f"\nΑποθηκεύτηκε το vol2_final.csv με τις νέες στήλες "
          f"'final_prediction' / 'final_prediction_source'.")

    out = valid[["mcp_eur_per_mwh", "final_prediction"]].rename(
        columns={"mcp_eur_per_mwh": "actual", "final_prediction": "predicted"}
    )
    out.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f"Αποθηκεύτηκε το {FINAL_OUTPUT_PATH} ({len(out)} γραμμές, στήλες: actual, predicted).")