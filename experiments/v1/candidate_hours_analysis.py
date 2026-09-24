"""
candidate_hours_analysis.py
==============================
Υπολογίζει, ανά ημερολογιακό μήνα (συγκεντρώνοντας ΟΛΑ τα χρόνια του
dataset μαζί -- π.χ. όλοι οι Ιανουάριοι μαζί, 2023-2026), σε ποιες ώρες
εμφανίζεται στην πραγματικότητα το ημερήσιο μέγιστο και το ημερήσιο
ελάχιστο τιμής. Από αυτή την κατανομή, παράγει ΑΥΤΟΜΑΤΑ ένα σύνολο
υποψηφίων ωρών ανά μήνα που καλύπτει τουλάχιστον COVERAGE_TARGET (π.χ.
85%) των ιστορικών ακραίων περιστατικών (μέγιστο + ελάχιστο μαζί).

Αυτό αντικαθιστά τα χειροκίνητα/εμπειρικά παράθυρα που είχαμε πριν
(βελτιστοποιημένα για διαφορετικό σκοπό -- ξεχωριστούς regressors ανά
ζώνη) με κάτι πιο συστηματικό: "ποιες ώρες πραγματικά χρειάζεται να
προσέχουμε ανά μήνα, βάσει ιστορικού".

Έξοδος: ένα dict CANDIDATE_HOURS_BY_MONTH έτοιμο να μπει στο
two_stage_gpd_model.py, + γραφήματα ανά μήνα για οπτικό έλεγχο.
"""

from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]  # repository root

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_PATH = Path(
    str(REPO / "Dataset_Creation" / "processed" / "final_dataset.csv")
)
COVERAGE_TARGET = 0.85  # ποσοστό ιστορικών ακραίων που θέλουμε να καλύπτουν οι υποψήφιες ώρες


def hours_covering_target(hour_counts: pd.Series, target: float) -> set[int]:
    """Δέχεται value_counts() ωρών (φθίνουσα σειρά συχνότητας), επιστρέφει
    το ΜΙΚΡΟΤΕΡΟ σύνολο ωρών που καλύπτει >= target ποσοστό του συνόλου."""
    total = hour_counts.sum()
    cumulative = 0
    selected = set()
    for hour, count in hour_counts.items():
        selected.add(int(hour))
        cumulative += count
        if cumulative / total >= target:
            break
    return selected


def analyze_month(df: pd.DataFrame, month: int) -> dict:
    month_df = df[df["timestamp"].dt.month == month].copy()
    month_df["date"] = month_df["timestamp"].dt.date
    month_df["hour"] = month_df["timestamp"].dt.hour

    idx_max = month_df.groupby("date")["mcp_eur_per_mwh"].idxmax()
    idx_min = month_df.groupby("date")["mcp_eur_per_mwh"].idxmin()

    max_hours = month_df.loc[idx_max, "hour"]
    min_hours = month_df.loc[idx_min, "hour"]

    # Καλύπτουμε ΚΑΙ τα δύο άκρα -- ενώνουμε τις εμφανίσεις τους σε μία
    # κοινή κατανομή, ώστε το υποψήφιο σύνολο να καλύπτει και τα δύο.
    combined = pd.concat([max_hours, min_hours])
    hour_counts = combined.value_counts().sort_values(ascending=False)

    candidate_hours = hours_covering_target(hour_counts, COVERAGE_TARGET)

    return {
        "month": month,
        "n_days": month_df["date"].nunique(),
        "max_hour_counts": max_hours.value_counts().sort_index(),
        "min_hour_counts": min_hours.value_counts().sort_index(),
        "candidate_hours": candidate_hours,
    }


def plot_month(analysis: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5))
    max_counts = analysis["max_hour_counts"].reindex(range(24), fill_value=0)
    min_counts = analysis["min_hour_counts"].reindex(range(24), fill_value=0)

    width = 0.4
    ax.bar(max_counts.index - width / 2, max_counts.values, width=width, label="Ημερήσιο μέγιστο", color="darkorange")
    ax.bar(min_counts.index + width / 2, min_counts.values, width=width, label="Ημερήσιο ελάχιστο", color="steelblue")

    for h in analysis["candidate_hours"]:
        ax.axvline(h, color="red", alpha=0.15, linewidth=8)

    ax.set_title(f"Μήνας {analysis['month']:02d} -- Υποψήφιες ώρες (κόκκινο): {sorted(analysis['candidate_hours'])}")
    ax.set_xlabel("Ώρα ημέρας")
    ax.set_ylabel("Πλήθος ημερών")
    ax.set_xticks(range(24))
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    print(f"Φόρτωση: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    candidate_hours_by_month = {}

    for month in range(1, 13):
        analysis = analyze_month(df, month)
        candidate_hours_by_month[month] = sorted(analysis["candidate_hours"])

        print(f"\n=== Μήνας {month:02d} ({analysis['n_days']} ημέρες ιστορικά) ===")
        print(f"Υποψήφιες ώρες (κάλυψη >= {COVERAGE_TARGET*100:.0f}%): {sorted(analysis['candidate_hours'])}")

        out_path = DATASET_PATH.parent / f"candidate_hours_month_{month:02d}.png"
        plot_month(analysis, out_path)

    print("\n" + "=" * 60)
    print("ΕΤΟΙΜΟ dict για επικόλληση στο two_stage_gpd_model.py:")
    print("=" * 60)
    print("CANDIDATE_HOURS_BY_MONTH = {")
    for month, hours in candidate_hours_by_month.items():
        print(f"    {month}: {{{', '.join(map(str, hours))}}},")
    print("}")