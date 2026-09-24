"""
evt_analysis.py
=================
ΒΗΜΑ 1 του EVT (Extreme Value Theory) πλάνου: στατικό, ΜΗ εξαρτημένο από
features, fit μιας Generalized Pareto Distribution (GPD) πάνω στις
υπερβάσεις ενός υψηλού ορίου (Peaks-Over-Threshold μέθοδος) -- τόσο για
την ΠΑΝΩ ουρά (spikes υψηλής τιμής) όσο και για την ΚΑΤΩ ουρά (πολύ
χαμηλές/αρνητικές τιμές).

Αυτό είναι ΔΙΑΓΝΩΣΤΙΚΟ βήμα -- δεν κάνει ακόμα προβλέψεις με features.
Σκοπός: να καταλάβουμε πόσο "βαριά" είναι πραγματικά η ουρά της ελληνικής
αγοράς DAM, πριν προχωρήσουμε στο πιο σύνθετο conditional μοντέλο (όπου η
scale παράμετρος της GPD θα εξαρτάται από features).

Θεωρία (πολύ σύντομα): το θεώρημα Pickands-Balkema-de Haan λέει ότι, για
αρκετά υψηλό όριο u, η κατανομή των υπερβάσεων (X-u | X>u) συγκλίνει
ΠΑΝΤΑ σε GPD, ανεξάρτητα από την αρχική κατανομή του X. Η παράμετρος
σχήματος (shape, ξ) λέει πόσο "βαριά" είναι η ουρά:
    ξ > 0: βαριά ουρά (Pareto-type)  -- ακραίες τιμές θεωρητικά απεριόριστες
    ξ = 0: εκθετική ουρά
    ξ < 0: φραγμένη ουρά -- υπάρχει πρακτικό ανώτατο όριο
"""

from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]  # repository root

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_PATH = Path(
    str(REPO / "Dataset_Creation" / "processed" / "final_dataset.csv")
)
UPPER_PERCENTILE = 95  # όριο u για την πάνω ουρά (top 5%)
LOWER_PERCENTILE = 5   # όριο u για την κάτω ουρά (bottom 5%)


def fit_gpd_tail(values: np.ndarray, threshold: float, tail: str) -> dict:
    """tail = 'upper' ή 'lower'. Επιστρέφει dict με shape, scale, threshold,
    και τις ίδιες τις υπερβάσεις (χρήσιμο για τα γραφήματα μετά)."""
    if tail == "upper":
        exceedances = values[values > threshold] - threshold
    else:
        exceedances = threshold - values[values < threshold]

    shape, loc, scale = stats.genpareto.fit(exceedances, floc=0)

    return {
        "tail": tail,
        "threshold": threshold,
        "shape": shape,
        "scale": scale,
        "n_exceedances": len(exceedances),
        "exceedances": exceedances,
    }


def interpret_shape(shape: float) -> str:
    if shape > 0.05:
        return "ΒΑΡΙΑ ουρά (Pareto-type) -- ακραίες τιμές θεωρητικά χωρίς πρακτικό όριο"
    elif shape < -0.05:
        return "ΦΡΑΓΜΕΝΗ ουρά -- υπάρχει πρακτικό ανώτατο/κατώτατο όριο"
    else:
        return "περίπου ΕΚΘΕΤΙΚΗ ουρά (ούτε ιδιαίτερα βαριά, ούτε φραγμένη)"


def print_summary(fit_result: dict) -> None:
    tail_label = "ΠΑΝΩ (υψηλές τιμές / spikes)" if fit_result["tail"] == "upper" else "ΚΑΤΩ (χαμηλές/αρνητικές τιμές)"
    print(f"\n--- GPD fit: {tail_label} ---")
    print(f"Όριο (threshold): {fit_result['threshold']:.2f} €/MWh")
    print(f"Πλήθος υπερβάσεων: {fit_result['n_exceedances']}")
    print(f"Shape (ξ): {fit_result['shape']:.3f}  -> {interpret_shape(fit_result['shape'])}")
    print(f"Scale (σ): {fit_result['scale']:.2f}")

    # Return level: αναμενόμενη υπέρβαση στο 99ο εκατοστημόριο ΤΩΝ ΥΠΕΡΒΑΣΕΩΝ
    # (δηλαδή: "1 στις 100 φορές που ξεπερνάμε το όριο, η υπέρβαση θα είναι
    # τουλάχιστον τόσο μεγάλη")
    q99 = stats.genpareto.ppf(0.99, c=fit_result["shape"], scale=fit_result["scale"])
    if fit_result["tail"] == "upper":
        print(f"99ο εκατοστημόριο υπέρβασης: +{q99:.1f} €/MWh πάνω από το όριο"
              f"  (δηλ. τιμή ≈ {fit_result['threshold'] + q99:.1f} €/MWh)")
    else:
        print(f"99ο εκατοστημόριο υπέρβασης: -{q99:.1f} €/MWh κάτω από το όριο"
              f"  (δηλ. τιμή ≈ {fit_result['threshold'] - q99:.1f} €/MWh)")


def plot_gpd_fit(fit_result: dict, out_path: Path) -> None:
    exceedances = fit_result["exceedances"]
    shape, scale = fit_result["shape"], fit_result["scale"]
    tail_label = "Πάνω ουρά (spikes)" if fit_result["tail"] == "upper" else "Κάτω ουρά (χαμηλές τιμές)"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Αριστερά: histogram υπερβάσεων + fitted GPD pdf
    x = np.linspace(0, exceedances.max(), 200)
    pdf = stats.genpareto.pdf(x, c=shape, scale=scale)
    axes[0].hist(exceedances, bins=30, density=True, alpha=0.6, color="steelblue", edgecolor="black")
    axes[0].plot(x, pdf, color="red", linewidth=2, label=f"GPD fit (ξ={shape:.2f}, σ={scale:.1f})")
    axes[0].set_title(f"{tail_label}: Κατανομή υπερβάσεων vs GPD fit")
    axes[0].set_xlabel("Μέγεθος υπέρβασης (€/MWh)")
    axes[0].set_ylabel("Πυκνότητα")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Δεξιά: Q-Q plot (empirical vs θεωρητικά quantiles) -- goodness of fit
    empirical_q = np.sort(exceedances)
    n = len(empirical_q)
    theoretical_q = stats.genpareto.ppf((np.arange(1, n + 1) - 0.5) / n, c=shape, scale=scale)
    axes[1].scatter(theoretical_q, empirical_q, alpha=0.5, s=15, color="steelblue")
    max_val = max(empirical_q.max(), theoretical_q.max())
    axes[1].plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Τέλεια ταύτιση")
    axes[1].set_title(f"{tail_label}: Q-Q plot (goodness of fit)")
    axes[1].set_xlabel("Θεωρητικά quantiles (GPD)")
    axes[1].set_ylabel("Πραγματικά quantiles")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Αποθηκεύτηκε το γράφημα: {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Φόρτωση: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    prices = df["mcp_eur_per_mwh"].values

    upper_threshold = np.percentile(prices, UPPER_PERCENTILE)
    lower_threshold = np.percentile(prices, LOWER_PERCENTILE)

    upper_fit = fit_gpd_tail(prices, upper_threshold, "upper")
    lower_fit = fit_gpd_tail(prices, lower_threshold, "lower")

    print("=" * 60)
    print("EVT / GPD ΔΙΑΓΝΩΣΤΙΚΗ ΑΝΑΛΥΣΗ")
    print("=" * 60)
    print_summary(upper_fit)
    print_summary(lower_fit)

    plot_gpd_fit(upper_fit, DATASET_PATH.parent / "gpd_fit_upper_tail.png")
    plot_gpd_fit(lower_fit, DATASET_PATH.parent / "gpd_fit_lower_tail.png")