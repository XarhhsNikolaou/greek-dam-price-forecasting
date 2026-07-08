"""
spike_analysis.py
===================
Ανάλυση πάνω στα ΗΔΗ υπολογισμένα ωριαία αποτελέσματα του rolling
evaluation (rolling_evaluation_hourly_<year>.csv) — δεν χρειάζεται να
ξανατρέξει το μοντέλο.

Σκοπός: να ποσοτικοποιήσει αυτό που φαίνεται καθαρά στο γράφημα
actual-vs-predicted — ότι το μοντέλο αποτυγχάνει συστηματικά ακριβώς στα
ακραία σημεία (πραγματικά spikes που δεν προβλέπει, και "ψεύτικα" spikes
που προβλέπει αλλά δεν έρχονται).

Παράγει:
    1. MAE στις "ακραίες" ώρες (top X% τιμών) έναντι του υπόλοιπου —
       πόσο μεγαλύτερο είναι το σφάλμα εκεί, και πόσο μερίδιο του
       συνολικού σφάλματος προέρχεται από αυτές τις λίγες ώρες.
    2. Histogram: σε ποια ώρα της ημέρας τείνει να εμφανίζεται το
       ημερήσιο μέγιστο τιμής, σε όλο τον χρόνο αξιολόγησης.
    3. Καταμέτρηση "false alarms" (το μοντέλο προβλέπει ακραία τιμή, η
       πραγματική δεν είναι) και "misses" (η πραγματική είναι ακραία, η
       πρόβλεψη όχι).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
HOURLY_RESULTS_PATH = Path(
    r"C:\Users\harry\Desktop\Projects\Price Forecasting Project\Dataset_Creation\processed\rolling_evaluation_hourly_2025.csv"
)
EXTREME_PERCENTILES = [90, 95]  # ελέγχουμε top 10% ΚΑΙ top 5%


def load_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["abs_error"] = (df["predicted"] - df["actual"]).abs()
    return df


def quantify_extreme_error(df: pd.DataFrame, percentiles: list[int]) -> None:
    print("=" * 60)
    print("1. MAE σε ακραίες τιμές έναντι υπόλοιπου")
    print("=" * 60)

    overall_mae = df["abs_error"].mean()
    print(f"Συνολικό MAE (όλες οι ώρες): {overall_mae:.2f}\n")

    for pct in percentiles:
        threshold = np.percentile(df["actual"], pct)
        extreme = df[df["actual"] >= threshold]
        rest = df[df["actual"] < threshold]

        extreme_mae = extreme["abs_error"].mean()
        rest_mae = rest["abs_error"].mean()
        ratio = extreme_mae / rest_mae if rest_mae > 0 else float("inf")

        # Πόσο % του ΣΥΝΟΛΙΚΟΥ αθροίσματος σφάλματος προέρχεται από αυτές
        # τις λίγες ώρες (όχι μόνο μέσος όρος -- και συνεισφορά στο σύνολο).
        total_abs_error_sum = df["abs_error"].sum()
        extreme_contribution_pct = 100 * extreme["abs_error"].sum() / total_abs_error_sum

        print(f"--- Top {100 - pct}% τιμών (actual >= {threshold:.1f} €/MWh) ---")
        print(f"  Πλήθος ωρών: {len(extreme)} / {len(df)} ({100 * len(extreme) / len(df):.1f}%)")
        print(f"  MAE σε αυτές τις ώρες: {extreme_mae:.2f}")
        print(f"  MAE στο υπόλοιπο: {rest_mae:.2f}")
        print(f"  Λόγος (πόσο χειρότερο): {ratio:.2f}x")
        print(f"  Συνεισφορά στο συνολικό άθροισμα σφάλματος: {extreme_contribution_pct:.1f}%")
        print()


def hour_of_daily_extreme_histogram(df: pd.DataFrame, which: str, out_path: Path) -> pd.Series:
    """which = 'max' ή 'min' -- σε ποια ώρα εμφανίζεται το ημερήσιο μέγιστο
    ή ελάχιστο τιμής, σε όλο το έτος αξιολόγησης."""
    label = "μεγίστου" if which == "max" else "ελαχίστου"
    print("=" * 60)
    print(f"Ώρα εμφάνισης του ημερήσιου {label}")
    print("=" * 60)

    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    if which == "max":
        idx = df.groupby("date")["actual"].idxmax()
    else:
        idx = df.groupby("date")["actual"].idxmin()
    extreme_hours = df.loc[idx, "hour"]

    counts = extreme_hours.value_counts().sort_index()
    print(counts)
    print(f"\nΠιο συχνή ώρα ημερήσιου {label}: {counts.idxmax()}:00 ({counts.max()} μέρες)")

    plt.figure(figsize=(10, 6))
    color = "darkorange" if which == "max" else "steelblue"
    plt.bar(counts.index, counts.values, color=color, edgecolor="black")
    plt.title(f"Σε ποια ώρα εμφανίζεται το ημερήσιο {label} τιμής")
    plt.xlabel("Ώρα ημέρας")
    plt.ylabel("Πλήθος ημερών")
    plt.xticks(range(0, 24))
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"\nΑποθηκεύτηκε το γράφημα: {out_path}")
    plt.show()

    return counts


def false_alarms_and_misses(df: pd.DataFrame, pct: int = 95) -> None:
    print("=" * 60)
    print(f"3. False alarms / Misses (όριο: top {100 - pct}%)")
    print("=" * 60)

    threshold = np.percentile(df["actual"], pct)

    actual_extreme = df["actual"] >= threshold
    predicted_extreme = df["predicted"] >= threshold

    false_alarms = ((~actual_extreme) & predicted_extreme).sum()
    misses = (actual_extreme & (~predicted_extreme)).sum()
    hits = (actual_extreme & predicted_extreme).sum()
    total_actual_extreme = actual_extreme.sum()
    total_predicted_extreme = predicted_extreme.sum()

    print(f"Όριο ακραίας τιμής: {threshold:.1f} €/MWh")
    print(f"Πραγματικές ακραίες ώρες: {total_actual_extreme}")
    print(f"Προβλεπόμενες ως ακραίες ώρες: {total_predicted_extreme}")
    print(f"  Σωστές προβλέψεις ακραίας τιμής (hits): {hits}")
    print(f"  Ψευδείς συναγερμοί (false alarms): {false_alarms}")
    print(f"  Χαμένα πραγματικά spikes (misses): {misses}")
    if total_actual_extreme > 0:
        print(f"  Recall (πόσα πραγματικά spikes πιάστηκαν): {100 * hits / total_actual_extreme:.1f}%")
    if total_predicted_extreme > 0:
        print(f"  Precision (πόσες 'ειδοποιήσεις' ήταν σωστές): {100 * hits / total_predicted_extreme:.1f}%")


if __name__ == "__main__":
    print(f"Φόρτωση: {HOURLY_RESULTS_PATH}\n")
    df = load_hourly(HOURLY_RESULTS_PATH)

    quantify_extreme_error(df, EXTREME_PERCENTILES)

    max_hist_path = HOURLY_RESULTS_PATH.parent / "hour_of_daily_max_histogram.png"
    hour_of_daily_extreme_histogram(df, "max", max_hist_path)

    min_hist_path = HOURLY_RESULTS_PATH.parent / "hour_of_daily_min_histogram.png"
    hour_of_daily_extreme_histogram(df, "min", min_hist_path)

    false_alarms_and_misses(df, pct=95)
