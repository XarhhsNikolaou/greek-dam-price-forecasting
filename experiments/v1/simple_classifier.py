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

# Ο classifier βλέπει ΜΟΝΟ αυτά τα 3 features αντί για όλο το feature_cols
# + extra_cols, σε ΟΛΟΥΣ τους μήνες (καμία per-month εξαίρεση). Το normal
# regressor και το GPD ΔΕΝ επηρεάζονται -- βλέπουν πάντα όλα τα κανονικά
# features. Υπόθεση: "ημέρα" = day_of_week, "net ζήτηση" = net_load_proxy_mw.
CLASSIFIER_FEATURE_COLS = ["day_of_week", "hour", "net_load_proxy_mw"]

# Ζώνη ωρών αιχμής -- ΙΔΙΑ με το rolling_evaluation_hybrid.py, που έδινε
# τα βέλτιστα αποτελέσματα ως standalone μοντέλο. Χρησιμοποιείται εδώ ΜΟΝΟ
# για το "normal" (base) μοντέλο: ό,τι ώρα ΔΕΝ ταξινομηθεί ως ακραία από
# τον classifier παίρνει πλέον πρόβλεψη από το hybrid (γενικό + zone-
# specific), αντί για το ενιαίο XGBoost που είχαμε πριν.
PEAK_HOURS = {10, 11, 12, 18, 19, 20}
MIN_ZONE_TRAIN_SAMPLES = 100  # ελάχιστο δείγμα για να εκπαιδευτεί το zone model, αλλιώς fallback σε model_all

# Υποψήφιες ώρες ανά μήνα -- ΠΛΕΟΝ data-driven, από το
# candidate_hours_analysis.py: οι ώρες που καλύπτουν το 85% των ιστορικών
# ημερήσιων μέγιστων + ελάχιστων τιμών (όλα τα χρόνια του dataset μαζί).
# Αντικαθιστά τα προηγούμενα χειροκίνητα/εμπειρικά παράθυρα.
CANDIDATE_HOURS_BY_MONTH = {
    1: {1, 2, 3, 4, 7, 8, 11, 12, 13, 17, 18, 19, 23},
    2: {2, 3, 7, 8, 10, 11, 12, 13, 17, 18, 19},
    3: {8, 9, 10, 11, 12, 13, 17, 18, 19},
    4: {7, 10, 11, 12, 13, 14, 19, 20},
    5: {9, 10, 11, 12, 13, 14, 19, 20},
    6: {9, 10, 11, 12, 13, 14, 19, 20, 21},
    7: {9, 10, 11, 12, 13, 19, 20, 21},
    8: {9, 10, 11, 12, 13, 19, 20},
    9: {10, 11, 12, 13, 14, 19},
    10: {3, 7, 10, 11, 12, 13, 18, 19},
    11: {2, 3, 4, 10, 11, 16, 17, 18, 23},
    12: {2, 3, 4, 7, 10, 11, 12, 16, 17, 18, 19},
}

# Ιστορική συχνότητα (0-1) που η κάθε ώρα υπήρξε ημερήσιο ΜΕΓΙΣΤΟ ή
# ΕΛΑΧΙΣΤΟ, ΞΕΧΩΡΙΣΤΑ (όχι πια συνδυασμένα σε ένα νούμερο) -- ώστε ο
# classifier να ξέρει ρητά ΠΡΟΣ ΠΟΙΑ ΚΑΤΕΥΘΥΝΣΗ τείνει κάθε ώρα, όχι απλά
# "τείνει να είναι ακραία εν γένει". π.χ. τον Ιούνιο, η ώρα 9 έχει
# freq_as_min=0.108 αλλά freq_as_max=0 -- ήταν συχνά ελάχιστο, ποτέ μέγιστο.
# ΣΗΜΕΙΩΣΗ (διαφάνεια): υπολογίστηκε από ΟΛΟ το dataset (2023-2026), άρα
# για αξιολόγηση εντός του 2025 χρησιμοποιεί τεχνικά και λίγη μελλοντική
# πληροφορία (2026) -- ίδια παραδοχή με τα CANDIDATE_HOURS_BY_MONTH.
FREQ_AS_MAX = {
    1: {0: 0.0091, 1: 0.0182, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0091, 6: 0.0273, 7: 0.1, 8: 0.0818, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0182, 16: 0.0364, 17: 0.3727, 18: 0.1818, 19: 0.1182, 20: 0.0182, 21: 0.0, 22: 0.0091, 23: 0.0},
    2: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0088, 6: 0.0265, 7: 0.0796, 8: 0.0796, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0177, 17: 0.1504, 18: 0.3894, 19: 0.2212, 20: 0.0088, 21: 0.0088, 22: 0.0088, 23: 0.0},
    3: {0: 0.0081, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0163, 6: 0.0163, 7: 0.0325, 8: 0.0488, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0569, 18: 0.4065, 19: 0.3577, 20: 0.0407, 21: 0.0, 22: 0.0163, 23: 0.0},
    4: {0: 0.0083, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.05, 7: 0.0667, 8: 0.0083, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0333, 19: 0.35, 20: 0.4333, 21: 0.0417, 22: 0.0, 23: 0.0083},
    5: {0: 0.0161, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0081, 5: 0.0081, 6: 0.0161, 7: 0.0081, 8: 0.0081, 9: 0.0081, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0161, 19: 0.0645, 20: 0.75, 21: 0.0161, 22: 0.0403, 23: 0.0403},
    6: {0: 0.0333, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0083, 8: 0.0083, 9: 0.0, 10: 0.0083, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.025, 17: 0.0083, 18: 0.0333, 19: 0.075, 20: 0.6667, 21: 0.0667, 22: 0.025, 23: 0.0417},
    7: {0: 0.0108, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0108, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0108, 16: 0.0, 17: 0.0, 18: 0.043, 19: 0.1398, 20: 0.6882, 21: 0.0968, 22: 0.0, 23: 0.0},
    8: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0323, 19: 0.2258, 20: 0.6882, 21: 0.0323, 22: 0.0108, 23: 0.0108},
    9: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0115, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0115, 18: 0.0115, 19: 0.8851, 20: 0.0805, 21: 0.0, 22: 0.0, 23: 0.0},
    10: {0: 0.0109, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0652, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0109, 16: 0.0217, 17: 0.0543, 18: 0.4022, 19: 0.4348, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0},
    11: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0222, 6: 0.0, 7: 0.0333, 8: 0.0222, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0, 13: 0.0, 14: 0.0, 15: 0.0222, 16: 0.3889, 17: 0.2667, 18: 0.1889, 19: 0.0444, 20: 0.0111, 21: 0.0, 22: 0.0, 23: 0.0},
    12: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0215, 7: 0.0753, 8: 0.0215, 9: 0.0215, 10: 0.0, 11: 0.0108, 12: 0.0108, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.2581, 17: 0.2796, 18: 0.1183, 19: 0.129, 20: 0.0323, 21: 0.0108, 22: 0.0108, 23: 0.0},
}

FREQ_AS_MIN = {
    1: {0: 0.0273, 1: 0.0364, 2: 0.1091, 3: 0.2818, 4: 0.0818, 5: 0.0273, 6: 0.0091, 7: 0.0, 8: 0.0, 9: 0.0364, 10: 0.0455, 11: 0.1364, 12: 0.0818, 13: 0.0636, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0091, 23: 0.0545},
    2: {0: 0.0442, 1: 0.0177, 2: 0.0531, 3: 0.1504, 4: 0.0354, 5: 0.0, 6: 0.0088, 7: 0.0, 8: 0.0, 9: 0.0354, 10: 0.0708, 11: 0.1504, 12: 0.2035, 13: 0.1681, 14: 0.0088, 15: 0.0177, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0088, 22: 0.0, 23: 0.0265},
    3: {0: 0.0163, 1: 0.0081, 2: 0.0, 3: 0.0325, 4: 0.0, 5: 0.0, 6: 0.0081, 7: 0.0, 8: 0.0081, 9: 0.0488, 10: 0.1707, 11: 0.2764, 12: 0.1707, 13: 0.1951, 14: 0.0244, 15: 0.0081, 16: 0.0081, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0163, 22: 0.0, 23: 0.0081},
    4: {0: 0.0167, 1: 0.0, 2: 0.0167, 3: 0.025, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0083, 10: 0.1083, 11: 0.125, 12: 0.1667, 13: 0.3333, 14: 0.125, 15: 0.0333, 16: 0.0333, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0083},
    5: {0: 0.0, 1: 0.0, 2: 0.0161, 3: 0.0403, 4: 0.0081, 5: 0.0, 6: 0.0, 7: 0.0081, 8: 0.0, 9: 0.0968, 10: 0.1935, 11: 0.0887, 12: 0.121, 13: 0.25, 14: 0.1452, 15: 0.0161, 16: 0.0081, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0081},
    6: {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0083, 4: 0.0, 5: 0.0083, 6: 0.025, 7: 0.0, 8: 0.0167, 9: 0.1083, 10: 0.2667, 11: 0.1417, 12: 0.0667, 13: 0.1583, 14: 0.15, 15: 0.0333, 16: 0.0, 17: 0.0083, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0083},
    7: {0: 0.0108, 1: 0.0, 2: 0.0215, 3: 0.043, 4: 0.0753, 5: 0.0215, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.086, 10: 0.172, 11: 0.2258, 12: 0.172, 13: 0.129, 14: 0.043, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0},
    8: {0: 0.0, 1: 0.0, 2: 0.0108, 3: 0.0645, 4: 0.0215, 5: 0.0108, 6: 0.0108, 7: 0.0, 8: 0.0, 9: 0.0968, 10: 0.2473, 11: 0.2258, 12: 0.1075, 13: 0.1613, 14: 0.043, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0},
    9: {0: 0.0, 1: 0.0, 2: 0.0575, 3: 0.0345, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0, 9: 0.0345, 10: 0.092, 11: 0.1839, 12: 0.1839, 13: 0.2874, 14: 0.1264, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0},
    10: {0: 0.0, 1: 0.0109, 2: 0.0217, 3: 0.1413, 4: 0.0435, 5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0109, 9: 0.0326, 10: 0.0761, 11: 0.1522, 12: 0.2065, 13: 0.2391, 14: 0.0435, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0217},
    11: {0: 0.0222, 1: 0.0444, 2: 0.0667, 3: 0.2333, 4: 0.0556, 5: 0.0, 6: 0.0111, 7: 0.0, 8: 0.0, 9: 0.0, 10: 0.2444, 11: 0.1556, 12: 0.0444, 13: 0.0, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0111, 22: 0.0, 23: 0.1111},
    12: {0: 0.0, 1: 0.0215, 2: 0.0753, 3: 0.3441, 4: 0.129, 5: 0.0108, 6: 0.0215, 7: 0.0, 8: 0.0108, 9: 0.0108, 10: 0.0753, 11: 0.1505, 12: 0.0645, 13: 0.0215, 14: 0.0, 15: 0.0, 16: 0.0, 17: 0.0, 18: 0.0, 19: 0.0, 20: 0.0, 21: 0.0, 22: 0.0, 23: 0.0645},
}


CLASSIFIER_PARAMS = dict(
    objective="multi:softmax",
    num_class=3,               # 0=normal, 1=low, 2=high
    max_depth=4,
    learning_rate=0.1,
    n_estimators=150,
)

# Βάρος ανά κλάση κατά την εκπαίδευση classifier -- οι κλάσεις low/high
# είναι σπάνιες (~5% η καθεμία), οπότε χωρίς ειδικό βάρος ο classifier
# τείνει να "προτιμά" να κάνει λάθος εκεί παρά στο normal. Μεγαλύτερο
# βάρος -> μεγαλύτερο recall σε low/high (με αντιστάθμισμα πιθανά
# περισσότερα false positives, εξισορροπείται από το GPD_CONFIDENCE_THRESHOLD).
CLASS_WEIGHT = {0: 1.0, 1: 3.0, 2: 3.0}

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

# --- Adaptive decision threshold (βασισμένο στο freq_as_max / freq_as_min) ---
# Απαιτούμενη πιθανότητα classifier για ταξινόμηση ως ακραία ώρα (ξεχωριστά
# ανά κατεύθυνση):
#   required_high = THRESH_MAX - (THRESH_MAX - THRESH_MIN) * freq_as_max
#   required_low  = THRESH_MAX - (THRESH_MAX - THRESH_MIN) * freq_as_min
# Ώρες με χαμηλή ιστορική συχνότητα (freq~0) χρειάζονται THRESH_MAX
# πιθανότητα (δύσκολο, λιγότερα false positives εκεί). Ώρες με υψηλή
# συχνότητα (freq~1) χρειάζονται μόνο THRESH_MIN (πιο εύκολο).
THRESH_MAX = 0.55
THRESH_MIN = 0.20

# "100-0" switch: αν η πιθανότητα του classifier για μια ήδη-ταξινομημένη
# ακραία ώρα ξεπερνά αυτό το όριο, χρησιμοποιούμε GPD ΠΛΗΡΩΣ· αλλιώς
# ΠΛΗΡΩΣ το κανονικό XGBoost (όχι ενδιάμεσο blending). Πιο "καθαρό":
# αποφεύγει το πρόβλημα όπου χαμηλή σιγουριά αραιώνει μια σωστή διόρθωση.
GPD_CONFIDENCE_THRESHOLD = 0.5

# Per-month προσαρμογή του required_prob (θετικό = πιο συντηρητικό/λιγότερες
# διορθώσεις, αρνητικό = πιο επιθετικό/περισσότερες διορθώσεις). Δεφόλτ όλα
# 0 (καμία προσαρμογή) -- συμπλήρωσε βάσει των αποτελεσμάτων σου ανά μήνα
# (π.χ. +0.1 σε μήνες που ήδη πάνε καλά σαν τον Αύγουστο, -0.1 σε μήνες
# που χρειάζονται επιθετική διόρθωση σαν τον Οκτώβριο).
MONTH_CONFIDENCE_ADJUSTMENT = {
    1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
    7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0,
}


def _standardize(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = X_train.mean()
    std = X_train.std().replace(0, 1.0)
    return ((X_train - mean) / std).values, ((X_test - mean) / std).values


def _propagate_to_neighbors(
    predicted_label: np.ndarray, test_hours: np.ndarray, candidate_hours: set,
    proba_high: np.ndarray, proba_low: np.ndarray,
    allowed_high: np.ndarray, allowed_low: np.ndarray, inherit_discount: float = 0.85,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Αν μια ώρα ταξινομήθηκε ως low/high, οι ΑΜΕΣΩΣ γειτονικές ώρες
    (hour-1, hour+1) που είναι ΕΠΙΣΗΣ υποψήφιες αλλά ταξινομήθηκαν ως
    'normal' αναβαθμίζονται στην ίδια κατηγορία -- τα ακραία γεγονότα
    τείνουν να διαρκούν πάνω από 1 ώρα.

    allowed_high/allowed_low: ΣΚΛΗΡΟΣ αποκλεισμός -- αν η ώρα-γείτονας δεν
    έχει ΠΟΤΕ ιστορικά υπάρξει μέγιστο (allowed_high[idx]=False), ΔΕΝ
    επιτρέπεται να κληρονομήσει ετικέτα 'high' μέσω διάδοσης, ό,τι κι αν
    λέει ο γείτονας (αντίστοιχα για low).

    ΣΗΜΑΝΤΙΚΟ: η ώρα που "κληρονομεί" την ετικέτα κληρονομεί ΚΑΙ την
    πιθανότητα της γειτονικής ώρας που την ενεργοποίησε (με μικρή
    έκπτωση) -- αλλιώς το confidence-weighted blending θα χρησιμοποιούσε
    τη ΔΙΚΗ ΤΗΣ (χαμηλή, γι' αυτό δεν ταξινομήθηκε ανεξάρτητα) πιθανότητα
    και θα ακύρωνε ουσιαστικά τη διόρθωση GPD."""
    label = predicted_label.copy()
    p_high = proba_high.copy()
    p_low = proba_low.copy()

    for idx in range(len(label)):
        if label[idx] != 0:
            continue
        hour = test_hours[idx]
        for neighbor_hour in (hour - 1, hour + 1):
            if neighbor_hour not in candidate_hours:
                continue
            neighbor_idx_arr = np.where(test_hours == neighbor_hour)[0]
            if len(neighbor_idx_arr) == 0:
                continue
            neighbor_idx = neighbor_idx_arr[0]
            neighbor_label = predicted_label[neighbor_idx]
            if neighbor_label == 2 and not allowed_high[idx]:
                continue  # ΣΚΛΗΡΟΣ αποκλεισμός: ποτέ ιστορικά μέγιστο εδώ
            if neighbor_label == 1 and not allowed_low[idx]:
                continue  # ΣΚΛΗΡΟΣ αποκλεισμός: ποτέ ιστορικά ελάχιστο εδώ
            if neighbor_label != 0:
                label[idx] = neighbor_label
                if neighbor_label == 2:
                    p_high[idx] = proba_high[neighbor_idx] * inherit_discount
                else:
                    p_low[idx] = proba_low[neighbor_idx] * inherit_discount
                break
    return label, p_high, p_low


def run_gpd_evaluation(df: pd.DataFrame, eval_year: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_cols = [c for c in df.columns if c not in ("timestamp", "mcp_eur_per_mwh", "net_out")]
    train_hours = MONTHS_USED * 30 * 24

    eval_start = pd.Timestamp(EVAL_START) if EVAL_START else pd.Timestamp(f"{eval_year}-09-01")
    eval_end = pd.Timestamp(EVAL_END) if EVAL_END else pd.Timestamp(f"{eval_year}-09-30")

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

        df_test = df[(df["timestamp"] >= test_start) & (df["timestamp"] <= test_end)].copy()
        df_train = df[(df["timestamp"] >= train_start) & (df["timestamp"] <= train_end)].copy()

        if len(df_test) < 24 or len(df_train) < MIN_TRAIN_HOURS:
            print(f"[{i}/{n_days}] {current_day.date()} -> παραλείπεται (ελλιπή δεδομένα)")
            current_day += pd.Timedelta(days=1)
            continue

        # --- Νέα features: ιστορική συχνότητα ΞΕΧΩΡΙΣΤΑ ως μέγιστο ή ελάχιστο ---
        df_train["freq_as_max"] = df_train.apply(
            lambda r: FREQ_AS_MAX[int(r["month"])][int(r["hour"])], axis=1
        )
        df_train["freq_as_min"] = df_train.apply(
            lambda r: FREQ_AS_MIN[int(r["month"])][int(r["hour"])], axis=1
        )
        df_test["freq_as_max"] = df_test.apply(
            lambda r: FREQ_AS_MAX[int(r["month"])][int(r["hour"])], axis=1
        )
        df_test["freq_as_min"] = df_test.apply(
            lambda r: FREQ_AS_MIN[int(r["month"])][int(r["hour"])], axis=1
        )

        extra_cols = ["freq_as_max", "freq_as_min"]
        X_train = df_train[feature_cols + extra_cols]
        y_train = df_train["mcp_eur_per_mwh"]
        X_test = df_test[feature_cols + extra_cols]
        y_test = df_test["mcp_eur_per_mwh"]

        # --- Ξεχωριστό, περιορισμένο feature set ΜΟΝΟ για τον classifier ---
        X_train_clf = df_train[CLASSIFIER_FEATURE_COLS]
        X_test_clf = df_test[CLASSIFIER_FEATURE_COLS]

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

        # --- Classifier: περιορισμένος στις υποψήφιες ώρες του μήνα (αν ορίζονται) ---
        candidate_hours = CANDIDATE_HOURS_BY_MONTH.get(current_day.month)

        classifier = xgb.XGBClassifier(**CLASSIFIER_PARAMS)
        predicted_label = np.zeros(len(y_test), dtype=int)  # default: όλα "normal"
        # proba_high/proba_low: πιθανότητα ανά ώρα (0 αν εκτός υποψηφίων ή
        # δεν εκπαιδεύτηκε classifier) -- χρησιμοποιείται ΚΑΙ για το
        # adaptive threshold ΚΑΙ για το confidence-weighted blending.
        proba_high = np.zeros(len(y_test))
        proba_low = np.zeros(len(y_test))

        freq_max_test = X_test["freq_as_max"].values
        freq_min_test = X_test["freq_as_min"].values
        # Adaptive threshold, ΞΕΧΩΡΙΣΤΑ ανά κατεύθυνση: χαμηλή ιστορική
        # συχνότητα ΩΣ ΜΕΓΙΣΤΟ -> χρειάζεται μεγαλύτερη πιθανότητα για να
        # ταξινομηθεί ως high (αντίστοιχα για low με το freq_as_min).
        # + per-month προσαρμογή (πιο συντηρητικό/επιθετικό ανά μήνα).
        month_adj = MONTH_CONFIDENCE_ADJUSTMENT.get(current_day.month, 0.0)
        required_prob_high = np.clip(THRESH_MAX - (THRESH_MAX - THRESH_MIN) * freq_max_test + month_adj, 0.05, 0.95)
        required_prob_low = np.clip(THRESH_MAX - (THRESH_MAX - THRESH_MIN) * freq_min_test + month_adj, 0.05, 0.95)

        # ΣΚΛΗΡΟΣ αποκλεισμός: αν μια ώρα δεν έχει ΠΟΤΕ ιστορικά υπάρξει
        # μέγιστο (freq_as_max==0), ΔΕΝ επιτρέπεται να ταξινομηθεί ως
        # "high" -- ανεξάρτητα από το τι λέει ο classifier. Αντίστοιχα
        # για "low" με freq_as_min==0. Ποτέ αρνητική ουρά σε ώρα υψηλής
        # τιμής, ό,τι κι αν λέει η πιθανότητα.
        allowed_high = freq_max_test > 0
        allowed_low = freq_min_test > 0
        required_prob_high = np.where(allowed_high, required_prob_high, 2.0)  # 2.0 = αδύνατο (proba max=1.0)
        required_prob_low = np.where(allowed_low, required_prob_low, 2.0)

        if candidate_hours is not None:
            train_cand_mask = df_train["hour"].isin(candidate_hours).values
            test_cand_mask = df_test["hour"].isin(candidate_hours).values

            sample_weight = np.array([CLASS_WEIGHT[l] for l in label_train[train_cand_mask]])
            classifier.fit(X_train_clf[train_cand_mask], label_train[train_cand_mask], sample_weight=sample_weight)
            if test_cand_mask.sum() > 0:
                proba_cand = classifier.predict_proba(X_test_clf[test_cand_mask])  # στήλες: [0,1,2]
                proba_low[test_cand_mask] = proba_cand[:, 1]
                proba_high[test_cand_mask] = proba_cand[:, 2]

                # Απόφαση με adaptive threshold (όχι απλό argmax): ταξινομούμε
                # ως high/low ΜΟΝΟ αν η πιθανότητα ξεπερνά το απαιτούμενο
                # όριο για ΕΚΕΙΝΗ την ώρα ΚΑΙ κατεύθυνση.
                cand_idx = np.where(test_cand_mask)[0]
                for j, idx in enumerate(cand_idx):
                    if proba_high[idx] >= required_prob_high[idx]:
                        predicted_label[idx] = 2
                    elif proba_low[idx] >= required_prob_low[idx]:
                        predicted_label[idx] = 1

            # --- Διάδοση σε γειτονικές ώρες (τα spikes τείνουν να διαρκούν) ---
            predicted_label, proba_high, proba_low = _propagate_to_neighbors(
                predicted_label, df_test["hour"].values, candidate_hours,
                proba_high, proba_low, allowed_high, allowed_low,
            )
        else:
            sample_weight = np.array([CLASS_WEIGHT[l] for l in label_train])
            classifier.fit(X_train_clf, label_train, sample_weight=sample_weight)
            proba_all = classifier.predict_proba(X_test_clf)
            proba_low = proba_all[:, 1]
            proba_high = proba_all[:, 2]
            for idx in range(len(predicted_label)):
                if proba_high[idx] >= required_prob_high[idx]:
                    predicted_label[idx] = 2
                elif proba_low[idx] >= required_prob_low[idx]:
                    predicted_label[idx] = 1

        # --- Normal (base) μοντέλο: ΠΛΕΟΝ hybrid (γενικό + zone-specific),
        # αντί για ένα ενιαίο XGBoost -- ίδια λογική με το
        # rolling_evaluation_hybrid.py, που έδινε τα καλύτερά μας αποτελέσματα.
        # Το ΓΕΝΙΚΟ μοντέλο εκπαιδεύεται ΜΟΝΟ στα "normal"-labeled δεδομένα
        # (label==0), όπως και πριν -- τα ακραία δεν μπαίνουν στη γενική βάση.
        # Το ZONE μοντέλο ΟΜΩΣ εκπαιδεύεται σε ΟΛΑ τα δεδομένα της ζώνης
        # (χωρίς αποκλεισμό extremes), ΑΚΡΙΒΩΣ όπως στο αυτόνομο hybrid
        # script -- εκεί δεν υπήρχε καν η έννοια extreme/normal label.
        normal_mask = label_train == 0
        X_train_normal = X_train[normal_mask]
        y_train_normal = y_train[normal_mask]
        train_hours_all = df_train["hour"].values
        is_zone_train_all = np.isin(train_hours_all, list(PEAK_HOURS))
        is_zone_test = np.isin(df_test["hour"].values, list(PEAK_HOURS))

        model_all = xgb.XGBRegressor(**NORMAL_REGRESSOR_PARAMS)
        model_all.fit(X_train_normal, y_train_normal)
        pred_normal_all = model_all.predict(X_test)

        if is_zone_train_all.sum() >= MIN_ZONE_TRAIN_SAMPLES and is_zone_test.any():
            model_zone = xgb.XGBRegressor(**NORMAL_REGRESSOR_PARAMS)
            model_zone.fit(X_train[is_zone_train_all], y_train[is_zone_train_all])
            pred_normal_all[is_zone_test] = model_zone.predict(X_test[is_zone_test])

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

        # --- "100-0" switch: GPD ΜΟΝΟ αν η σιγουριά ξεπερνά ένα υψηλότερο
        # όριο· αλλιώς ΠΛΗΡΩΣ το κανονικό XGBoost (καθαρές διορθώσεις,
        # όχι μερικές/αραιωμένες που θα μπορούσαν να αναιρέσουν όφελος).
        high_sel = (predicted_label == 2) & (proba_high >= GPD_CONFIDENCE_THRESHOLD)
        y_pred[high_sel] = pred_high_all[high_sel]
        low_sel = (predicted_label == 1) & (proba_low >= GPD_CONFIDENCE_THRESHOLD)
        y_pred[low_sel] = pred_low_all[low_sel]

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