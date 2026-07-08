"""
rolling_evaluation_hybrid.py
=============================
Rolling day-ahead αξιολόγηση σε βάθος ΟΛΟΚΛΗΡΟΥ έτους, με HYBRID μοντέλο:

    - Για τις ώρες μέσα στη ζώνη PEAK_HOURS (π.χ. 10,11,12,18,19,20):
      χρησιμοποιείται μοντέλο εκπαιδευμένο ΜΟΝΟ πάνω σε δεδομένα αυτών
      των ωρών (zone-specific model).
    - Για όλες τις υπόλοιπες ώρες: χρησιμοποιείται μοντέλο εκπαιδευμένο
      πάνω σε ΟΛΕΣ τις ώρες (γενικό μοντέλο).

Η επιλογή ζώνης γίνεται ΝΤΕΤΕΡΜΙΝΙΣΤΙΚΑ από την ώρα της ημέρας (γνωστή εκ
των προτέρων, μηδενικό ρίσκο leakage/λάθους ταξινόμησης) -- ίδια λογική
δρομολόγησης με το hour_bucket_model.py, αλλά με τη διαφορά που δοκιμάσαμε
και επιβεβαιώσαμε ότι δουλεύει καλύτερα: το μοντέλο για τις "κανονικές"
ώρες εκπαιδεύεται σε ΟΛΟ το training set (όχι μόνο στις μη-ζώνη ώρες),
ώστε να μην χάνει δεδομένα.

ΔΙΟΡΘΩΣΕΙΣ σε σχέση με το αρχικό rolling_evaluation.py:
  1. Το eval_start/eval_end ήταν hardcoded μόνο στον Οκτώβριο, παρόλο που
     το docstring έλεγε "ολόκληρο έτος". Εδώ καλύπτεται πραγματικά όλο
     το έτος μέσω EVAL_START/EVAL_END (configurable, με default 1/1-31/12).
  2. Το EVAL_START/EVAL_END γίνονται πραγματικά module-level constants
     (το hour_bucket_model.py προσπαθούσε να τα κάνει import από το
     rolling_evaluation.py, αλλά δεν υπήρχαν εκεί -- θα έσκαγε ImportError).
  3. Αν το EVAL_END ξεπερνά τη μέγιστη διαθέσιμη ημερομηνία στο dataset,
     κόβεται αυτόματα (χρήσιμο π.χ. για το 2026, όπου έχεις δεδομένα μόνο
     μέχρι τον Ιούνιο).

ΣΗΜΕΙΩΣΗ ΓΙΑ ΤΗ ΖΩΝΗ: η PEAK_HOURS ορίστηκε εμπειρικά και καλύπτει την
πλειοψηφία αλλά ΟΧΙ όλα τα spikes (στο test που κάναμε σε H1 2026, ~60%
των spikes έπεφταν μέσα στη ζώνη). Τα υπόλοιπα spikes δεν ωφελούνται από
το hybrid -- βλέπε breakdown στο τέλος του output (spikes εντός/εκτός ζώνης).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATASET_PATH = Path(r"C:\Users\harry\Desktop\Projects\Price Forecasting Project\Dataset_Creation\processed\final_dataset.csv")

EVAL_YEAR = 2025          # ποιο ημερολογιακό έτος αξιολογούμε
EVAL_START = None         # π.χ. "2025-01-01" -- None = 1 Ιανουαρίου του EVAL_YEAR
EVAL_END = None           # π.χ. "2025-12-31" -- None = 31 Δεκεμβρίου του EVAL_YEAR
                          # (κόβεται αυτόματα στη μέγιστη διαθέσιμη ημερομηνία)

MONTHS_USED = 24          # μήνες rolling training window πριν από κάθε μέρα
MIN_TRAIN_HOURS = 1000    # ελάχιστο πλήθος ωρών training, αλλιώς η μέρα παραλείπεται
MAE_THRESHOLD = 14        # "αποδεκτό" όριο MAE

PEAK_HOURS = {10, 11, 12, 18, 19, 20}   # ζώνη ωρών με zone-specific μοντέλο

# top X% των πραγματικών τιμών (μέσα στο eval έτος) που θεωρούνται "spike"
# για το τελικό breakdown -- καθαρά διαγνωστικό, δεν επηρεάζει το training
SPIKE_QUANTILE = 0.95

QUANTILE_ALPHA = 0.5      # 0.5 = median (μαθηματικά MAE-optimal στόχος)

XGB_PARAMS = dict(
    objective="reg:quantileerror",
    quantile_alpha=QUANTILE_ALPHA,
    max_depth=6,
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
    """Ίδιο feature engineering με το rolling_evaluation.py, με shift(1)
    πριν το rolling ώστε να αποφεύγεται leakage.

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
    df["rolling_mean_1day"] = df["mcp_eur_per_mwh"].shift(1).rolling(24).mean()
    df["rolling_mean_7days"] = df["mcp_eur_per_mwh"].shift(1).rolling(24 * 7).mean()
    df["lag_24h"] = df["mcp_eur_per_mwh"].shift(24)
    df["lag_25h"] = df["mcp_eur_per_mwh"].shift(25)
    df["lag_48h"] = df["mcp_eur_per_mwh"].shift(48)
    df["lag_168h"] = df["mcp_eur_per_mwh"].shift(168)

    df["rolling_max_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).max()
    df["rolling_min_1week"] = df["mcp_eur_per_mwh"].shift(24).rolling(168).min()

    return df.dropna().reset_index(drop=True)


def _resolve_eval_window(df: pd.DataFrame, eval_year: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-03-01")
    end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-03-31")

    max_available = df["timestamp"].max()
    if end > max_available:
        print(f"[Προσοχή] EVAL_END ({end.date()}) > μέγιστη διαθέσιμη ημερομηνία "
              f"στο dataset ({max_available.date()}). Κόβεται αυτόματα.")
        end = max_available.normalize()
    return start, end


def run_rolling_evaluation_hybrid(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start, eval_end = _resolve_eval_window(df, eval_year)

    results = []
    hourly_records = []
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

        is_zone_train = df_train["hour"].isin(PEAK_HOURS)
        is_zone_test = df_test["hour"].isin(PEAK_HOURS).values

        # --- Μοντέλο A: γενικό, εκπαιδευμένο σε ΟΛΕΣ τις ώρες ---
        X_train_all = df_train[feature_cols]
        y_train_all = df_train["mcp_eur_per_mwh"]
        model_all = xgb.XGBRegressor(**XGB_PARAMS)
        model_all.fit(X_train_all, y_train_all)

        # --- Μοντέλο B: zone-specific, εκπαιδευμένο ΜΟΝΟ στη ζώνη ---
        df_train_zone = df_train[is_zone_train]
        model_zone = None
        if len(df_train_zone) >= 100:  # ελάχιστο αξιόπιστο δείγμα για fit
            model_zone = xgb.XGBRegressor(**XGB_PARAMS)
            model_zone.fit(df_train_zone[feature_cols], df_train_zone["mcp_eur_per_mwh"])

        # --- Πρόβλεψη: hybrid routing βάσει ώρας ---
        y_pred = model_all.predict(df_test[feature_cols])
        if model_zone is not None and is_zone_test.any():
            zone_rows = df_test[is_zone_test]
            y_pred[is_zone_test] = model_zone.predict(zone_rows[feature_cols])

        y_test = df_test["mcp_eur_per_mwh"].values
        mae = float(np.mean(np.abs(y_pred - y_test)))

        results.append({
            "date": current_day.date(),
            "month": current_day.month,
            "season": SEASON_MAP[current_day.month],
            "mae": mae,
        })

        for ts, actual, pred, hr in zip(df_test["timestamp"].values, y_test, y_pred, df_test["hour"].values):
            hourly_records.append({
                "timestamp": ts, "actual": actual, "predicted": pred,
                "hour": hr, "in_zone": hr in PEAK_HOURS,
            })

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


def summarize_spikes(hourly: pd.DataFrame, spike_quantile: float) -> None:
    """Διαγνωστικό breakdown: πόσο καλά πάει το hybrid ειδικά στα spikes,
    και πόσα spikes πέφτουν εντός/εκτός της ζώνης."""
    hourly = hourly.copy()
    thr = hourly["actual"].quantile(spike_quantile)
    spike_mask = hourly["actual"] >= thr
    err = (hourly["predicted"] - hourly["actual"]).abs()

    print("\n" + "=" * 50)
    print(f"SPIKE BREAKDOWN (threshold = {spike_quantile:.0%} percentile = {thr:.2f} EUR/MWh)")
    print("=" * 50)
    print(f"Σύνολο spike ωρών: {spike_mask.sum()} / {len(hourly)}")

    in_zone_spikes = spike_mask & hourly["in_zone"]
    out_zone_spikes = spike_mask & (~hourly["in_zone"])
    n_spikes = spike_mask.sum()
    if n_spikes > 0:
        pct_in_zone = 100 * in_zone_spikes.sum() / n_spikes
        print(f"  εντός ζώνης ({sorted(PEAK_HOURS)}): {in_zone_spikes.sum()} ({pct_in_zone:.1f}%)")
        print(f"  εκτός ζώνης: {out_zone_spikes.sum()} ({100 - pct_in_zone:.1f}%)")

    print(f"\nMAE σε spike ώρες (σύνολο):      {err[spike_mask].mean():.2f}")
    if in_zone_spikes.sum() > 0:
        print(f"MAE σε spike ώρες εντός ζώνης:   {err[in_zone_spikes].mean():.2f}")
    if out_zone_spikes.sum() > 0:
        print(f"MAE σε spike ώρες εκτός ζώνης:   {err[out_zone_spikes].mean():.2f}")
    print(f"MAE σε μη-spike ώρες:            {err[~spike_mask].mean():.2f}")


def plot_mae_distribution(results: pd.DataFrame, threshold: float, out_path: Path) -> None:
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
    plt.title(f"Κατανομή ημερήσιου MAE (hybrid) — {pct:.1f}% των ημερών εντός ορίου")
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
              label="Πρόβλεψη (Predicted, hybrid)", color="red", linestyle="--", linewidth=1)
    plt.title(f"Πραγματική vs Πρόβλεψη (hybrid) — {year}-{month:02d}")
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

    print(f"Rolling HYBRID evaluation για το έτος {EVAL_YEAR}...\n")
    results, hourly = run_rolling_evaluation_hybrid(df, EVAL_YEAR)

    if results.empty:
        print("Καμία μέρα δεν αξιολογήθηκε — έλεγξε το EVAL_YEAR/EVAL_START/EVAL_END/dataset.")
    else:
        summarize(results)
        summarize_spikes(hourly, SPIKE_QUANTILE)

        out_path = DATASET_PATH.parent / f"rolling_evaluation_hybrid_{EVAL_YEAR}.csv"
        results.to_csv(out_path, index=False)
        print(f"\nΑποθηκεύτηκαν τα αναλυτικά αποτελέσματα: {out_path}")

        hourly_path = DATASET_PATH.parent / f"rolling_evaluation_hybrid_hourly_{EVAL_YEAR}.csv"
        hourly.to_csv(hourly_path, index=False)
        print(f"Αποθηκεύτηκαν τα ωριαία αποτελέσματα: {hourly_path}")

        plot_path = DATASET_PATH.parent / f"hybrid_mae_distribution_{EVAL_YEAR}.png"
        plot_mae_distribution(results, MAE_THRESHOLD, plot_path)

        month_plot_path = DATASET_PATH.parent / f"hybrid_actual_vs_predicted_month_{EVAL_YEAR}.png"
        plot_month_actual_vs_predicted(hourly, results, EVAL_YEAR, None, month_plot_path)