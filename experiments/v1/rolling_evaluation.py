"""
rolling_evaluation.py
======================
Αξιολογεί το μοντέλο με ΠΟΛΛΑΠΛΕΣ, ρεαλιστικές day-ahead προβλέψεις σε
ολόκληρο ένα έτος — όχι σε ένα μόνο τυχαίο 24ωρο (που έχει μεγάλη
διακύμανση ανάλογα με τη μέρα, όπως ήδη είδαμε: 50, 10, 7 MAE σε
διαφορετικές μέρες).

Για κάθε μέρα D του έτους αξιολόγησης:
    - train window: τελευταίοι `MONTHS_USED` μήνες πριν από τη μέρα D
      (rolling — μετακινείται μαζί με τη μέρα, όχι σταθερό ιστορικό)
    - test window: ακριβώς η μέρα D (24 ώρες)
    - εκπαιδεύει ΕΝΑ μοντέλο, προβλέπει τη μέρα, υπολογίζει MAE

Στο τέλος, τα ημερήσια MAE συγκεντρώνονται ανά μήνα, ανά εποχή, και
συνολικά για το έτος. Μέρες με γνωστά κενά δεδομένων (π.χ. το 14ήμερο
Ιανουαρίου 2025) παραλείπονται αυτόματα.
"""

from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[2]  # repository root

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_PATH = Path(str(REPO / "Dataset_Creation" / "processed" / "final_dataset.csv"))

EVAL_YEAR = 2026          # ποιο ημερολογιακό έτος αξιολογούμε (άλλαξέ το αν θες)
EVAL_START = None         # π.χ. "2025-01-01" -- None = 1 Ιανουαρίου του EVAL_YEAR
EVAL_END = None           # π.χ. "2025-12-31" -- None = 31 Δεκεμβρίου του EVAL_YEAR
                          # (κόβεται αυτόματα στη μέγιστη διαθέσιμη ημερομηνία του dataset)
MONTHS_USED = 18          # μήνες rolling training window πριν από κάθε μέρα
MIN_TRAIN_HOURS = 1000    # ελάχιστο πλήθος ωρών training, αλλιώς η μέρα παραλείπεται
MAE_THRESHOLD = 14        # "αποδεκτό" όριο MAE (σύμφωνα με πρακτική βιομηχανίας, ~10-14)
PLOT_MONTH = None         # 1-12 για συγκεκριμένο μήνα, ή None για αυτόματη επιλογή
                          # του μήνα με το μεγαλύτερο μέσο MAE (ο "χειρότερος" μήνας)

QUANTILE_ALPHA = 0.5     # 0.5 = median (μαθηματικά MAE-optimal στόχος).
                          # Δοκίμασε 0.7-0.8 αργότερα αν θες να στοχεύσεις
                          # ρητά ψηλότερα (καλύτερο recall σε spikes, με
                          # αντιστάθμισμα περισσότερα false alarms).

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILE_ALPHA,
    # ΔΙΟΡΘΩΣΗ: πριν, objective='reg:squarederror' εκπαίδευε το μοντέλο να
    # στοχεύει τη ΜΕΣΗ τιμή (mean), ενώ το MAE ελαχιστοποιείται μαθηματικά
    # από τη ΔΙΑΜΕΣΟ (median) -- αναντιστοιχία ανάμεσα σε στόχο εκπαίδευσης
    # και μετρική αξιολόγησης. Με quantile_alpha=0.5, το μοντέλο στοχεύει
    # ρητά τη median, ευθυγραμμισμένο με το MAE που μας ενδιαφέρει.
    max_depth=5,
    learning_rate=0.05,
    n_estimators=150,
    subsample=1.0,
)

SEASON_MAP = {
    12: "Χειμώνας", 1: "Χειμώνας", 2: "Χειμώνας",
    3: "Άνοιξη", 4: "Άνοιξη", 5: "Άνοιξη",
    6: "Καλοκαίρι", 7: "Καλοκαίρι", 8: "Καλοκαίρι",
    9: "Φθινόπωρο", 10: "Φθινόπωρο", 11: "Φθινόπωρο",
}


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ίδιο feature engineering με το feature_engineering.py, με τη
    ΔΙΟΡΘΩΣΗ leakage στα rolling means (χρειάζονται shift(1) πριν το
    rolling, αλλιώς το 'rolling_mean_1day' της ώρας t περιλαμβάνει και
    την ίδια την τιμή-στόχο mcp[t]).

    ΝΕΑ (προστέθηκαν αφού βρέθηκε εμπειρικά σημαντική βελτίωση, 3y rolling
    test σε πολλαπλά test windows, ~1-3.5 MAE βελτίωση): rolling max/min
    24ω και rolling skewness/kurtosis 24ω. Το max/min κάνει το μεγαλύτερο
    μέρος της δουλειάς (δίνει στο μοντέλο το "εύρος" της αγοράς τις
    τελευταίες 24 ώρες, κάτι που ούτε το lag_24h ούτε το rolling_mean
    πιάνουν καθαρά). Το skew/kurtosis μόνα τους είναι ασθενή (σε ένα test
    window μάλιστα ελαφρώς αρνητικά), αλλά σε συνδυασμό με το max/min
    έδωσαν το καλύτερο αποτέλεσμα σε όλα τα test windows -- γι' αυτό
    κρατιούνται όλα μαζί, ως ομάδα, όχι μεμονωμένα."""
    df = df.copy()
    df["rolling_mean_1day"] = df["mcp_eur_per_mwh"].shift(24).rolling(24).mean()
    df["rolling_mean_7days"] = df["mcp_eur_per_mwh"].shift(24).rolling(24 * 7).mean()
    df["lag_24h"] = df["mcp_eur_per_mwh"].shift(24)
    df["lag_25h"] = df["mcp_eur_per_mwh"].shift(25)
    df["lag_48h"] = df["mcp_eur_per_mwh"].shift(48)
    df["lag_168h"] = df["mcp_eur_per_mwh"].shift(168)

    df["delta_price"] = df["lag_24h"] - df["lag_25h"]
    df["delta_price_1week"] = df["lag_24h"] - df["lag_168h"]

    _s = df["mcp_eur_per_mwh"].shift(24)
    df["rolling_max_24h"] = _s.rolling(24).max()
    df["rolling_min_24h"] = _s.rolling(24).min()
    df["rolling_skew_24h"] = _s.rolling(24).skew()
    df["rolling_kurt_24h"] = _s.rolling(24).kurt()

    return df.dropna().reset_index(drop=True)


def _resolve_eval_window(df: pd.DataFrame, eval_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-10-01")
    end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-10-31")

    max_available = df["timestamp"].max()
    if end > max_available:
        print(f"[Προσοχή] EVAL_END ({end.date()}) > μέγιστη διαθέσιμη ημερομηνία "
              f"στο dataset ({max_available.date()}). Κόβεται αυτόματα.")
        end = max_available.normalize()
    return start, end


def run_rolling_evaluation(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start, eval_end = _resolve_eval_window(df, eval_year)

    results = []
    hourly_records = []  # timestamp, actual, predicted -- για το γράφημα ανά μήνα
    current_day = eval_start
    n_days = (eval_end - eval_start).days + 1
    i = 0

    while current_day <= eval_end:
        i += 1
        test_start = current_day
        test_end = current_day + pd.Timedelta(hours=23)
        train_end = test_start - pd.Timedelta(hours=1)
        train_start = train_end - pd.Timedelta(hours=train_hours - 1)

        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)]
        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)]

        # Παράλειψη μερών με γνωστά κενά δεδομένων (λιγότερες από 24 ώρες
        # test data), ή ανεπαρκές training ιστορικό.
        if len(df_test) < 24 or len(df_train) < MIN_TRAIN_HOURS:
            print(f"[{i}/{n_days}] {current_day.date()} -> παραλείπεται (ελλιπή δεδομένα)")
            current_day += pd.Timedelta(days=1)
            continue

        X_train = df_train[feature_cols]
        y_train = df_train["mcp_eur_per_mwh"]
        X_test = df_test[feature_cols]
        y_test = df_test["mcp_eur_per_mwh"]

        model = xgb.XGBRegressor(**XGB_PARAMS)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
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

    return pd.DataFrame(results), pd.DataFrame(hourly_records)


def summarize(results: pd.DataFrame) -> None:
    print("\n" + "=" * 50)
    print(f"ΣΥΝΟΨΗ — {len(results)} αξιολογημένες μέρες")
    print("=" * 50)

    print("\n--- Ανά μήνα ---")
    monthly = results.groupby("month")["mae"].agg(["mean", "std", "count"])
    print(monthly)

    print("\n--- Ανά εποχή ---")
    seasonal = results.groupby("season")["mae"].agg(["mean", "std", "count"])
    print(seasonal)

    print("\n--- Σύνολο έτους ---")
    print(f"Μέσο MAE: {results['mae'].mean():.2f}")
    print(f"Τυπική απόκλιση: {results['mae'].std():.2f}")
    print(f"Ελάχιστο/Μέγιστο: {results['mae'].min():.2f} / {results['mae'].max():.2f}")


def plot_mae_distribution(results: pd.DataFrame, threshold: float, out_path: Path) -> None:
    """Histogram της κατανομής του ημερήσιου MAE, με κάθετη γραμμή στο
    'αποδεκτό' όριο (π.χ. 14, πρακτική βιομηχανίας κατά το workshop).
    Τυπώνει επίσης πόσες/τι ποσοστό μέρες είναι μέσα στο όριο."""
    import matplotlib.pyplot as plt

    within = (results["mae"] <= threshold).sum()
    total = len(results)
    pct = 100 * within / total

    print(f"\n--- Μέρες εντός ορίου MAE <= {threshold} ---")
    print(f"{within} / {total} μέρες ({pct:.1f}%)")

    plt.figure(figsize=(10, 6))
    plt.hist(results["mae"], bins=30, color="steelblue", edgecolor="black", alpha=0.8)
    plt.axvline(threshold, color="red", linestyle="--", linewidth=2,
                label=f"Όριο αποδοχής (MAE = {threshold})")
    plt.title(f"Κατανομή ημερήσιου MAE — {pct:.1f}% των ημερών εντός ορίου")
    plt.xlabel("MAE (€/MWh)")
    plt.ylabel("Πλήθος ημερών")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Αποθηκεύτηκε το γράφημα: {out_path}")
    plt.show()



def plot_month_actual_vs_predicted(hourly: pd.DataFrame, daily_results: pd.DataFrame,
                                    year: int, month: "int | None", out_path: Path) -> None:
    """Γράφημα πραγματικής vs προβλεπόμενης τιμής, ώρα-προς-ώρα, για έναν
    ολόκληρο μήνα. Αν δεν δοθεί συγκεκριμένος μήνας, επιλέγεται αυτόματα
    ο μήνας με το μεγαλύτερο μέσο ημερήσιο MAE (ο "χειρότερος" μήνας) —
    χρήσιμο ακριβώς για να δούμε αν πολλά spikes συγκεντρώνονται εκεί."""
    import matplotlib.pyplot as plt

    hourly = hourly.copy()
    hourly["timestamp"] = pd.to_datetime(hourly["timestamp"])

    if month is None:
        monthly_mae = daily_results.groupby("month")["mae"].mean()
        month = int(monthly_mae.idxmax())
        print(f"\nΑυτόματη επιλογή χειρότερου μήνα: {month} (μέσο MAE: {monthly_mae.loc[month]:.2f})")

    month_data = hourly[
        (hourly["timestamp"].dt.year == year) & (hourly["timestamp"].dt.month == month)
    ].sort_values("timestamp")

    if month_data.empty:
        print(f"Δεν υπάρχουν δεδομένα για {year}-{month:02d} — παραλείπεται το γράφημα.")
        return

    plt.figure(figsize=(16, 6))
    plt.plot(month_data["timestamp"], month_data["actual"],
              label="Πραγματική Τιμή (Actual)", color="blue", linewidth=1)
    plt.plot(month_data["timestamp"], month_data["predicted"],
              label="Πρόβλεψη (Predicted)", color="red", linestyle="--", linewidth=1)
    plt.title(f"Πραγματική vs Πρόβλεψη — {year}-{month:02d}")
    plt.xlabel("Ημερομηνία")
    plt.ylabel("Τιμή (€/MWh)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Αποθηκεύτηκε το γράφημα μήνα: {out_path}")
    plt.show()


if __name__ == "__main__":
    print(f"Φόρτωση dataset από: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    print("Feature engineering...")
    df = build_features(df)

    print(f"Rolling evaluation για το έτος {EVAL_YEAR}...\n")
    results, hourly = run_rolling_evaluation(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε το EVAL_YEAR/dataset.")
    else:
        summarize(results)
        out_path = DATASET_PATH.parent / f"rolling_evaluation_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")

        hourly_path = DATASET_PATH.parent / f"rolling_evaluation_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"Αποθηκεύτηκαν τα ωριαία αποτελέσματα: {hourly_path}")

        plot_path = DATASET_PATH.parent / f"mae_distribution_{EVAL_YEAR}.png"
        plot_mae_distribution(results, MAE_THRESHOLD, plot_path)

        month_plot_path = DATASET_PATH.parent / f"actual_vs_predicted_month_{EVAL_YEAR}.png"
        plot_month_actual_vs_predicted(hourly, results, EVAL_YEAR, PLOT_MONTH, month_plot_path)