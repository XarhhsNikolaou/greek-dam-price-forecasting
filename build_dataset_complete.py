"""
build_dataset.py
=================
Ενιαίο script που χτίζει το dataset για την πρόβλεψη τιμών DAM, βήμα-βήμα.

Κάθε στάδιο (stage) παίρνει ως input το αρχείο του προηγούμενου σταδίου και
γράφει ΝΕΟ αρχείο (ποτέ δεν κάνει overwrite στο input του) — έτσι όλα τα
ενδιάμεσα στάδια παραμένουν διαθέσιμα, τόσο για reproducibility όσο και για
να φαίνεται η εξέλιξη του dataset (π.χ. πώς άλλαζε το MAE του μοντέλου
καθώς προσθέτονταν νέα features).

Στάδια:
    01_base                  -> το αρχικό dataset (2023-2025), όπως είναι
    02_with_2026_prices      -> + DAM 2026, + ΑΔΜΗΕ Load/RES 2026
    03_with_fuel_carbon      -> + NGAS price, + carbon price
    04_with_calendar         -> + hour, month, day_of_week, is_weekend
    05_with_flows            -> + net_out (cross-border flows) — ΜΟΝΟ ως raw/reference,
                                 όχι ακόμα ασφαλές ως model feature (βλ. στάδιο 8)
    06_with_outages          -> + unit outages/unavailability
    07_with_hydro            -> + στάθμη ταμιευτήρων υδροηλεκτρικών (ENTSO-E, εβδομαδιαία)
    08_with_net_out_forecast -> + net_out_forecast: μια ΠΡΟΒΛΕΨΗ του net_out (όχι η
                                 πραγματική τιμή) — αυτό ΕΙΝΑΙ ασφαλές ως model feature

Σημείωση: τα paths παρακάτω είναι ακόμα absolute/hardcoded — αυτό είναι
εντάξει προς το παρόν, όσο δουλεύουμε τοπικά. Θα γίνουν relative/config
μόνο στο τέλος, όταν το pipeline ετοιμαστεί για GitHub.
"""

from pathlib import Path as _Path
REPO = _Path(__file__).resolve().parents[0]  # repository root

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# CONFIG — άλλαξε ΜΟΝΟ αυτό το κομμάτι για να ταιριάζει με τους φακέλους σου
# ---------------------------------------------------------------------------
CONFIG = {
    # Root φάκελος του project
    "project_root": Path(str(REPO)),

    # Αρχικό, ήδη-καθαρό base dataset (2023-2025)
    "base_file": Path(str(REPO / "dam_prices_with_load_res_forecasts_clean (1).csv")),

    # Raw φάκελοι
    "dam_2026_dir": None,       # π.χ. project_root / "DAM 2026"
    "admie_dir": None,          # π.χ. project_root / "ΑΔΜΗΕ_Folder"
    "ngas_dir": None,           # π.χ. project_root / "Τιμές Καυσίμων"
    "carbon_file": None,        # π.χ. project_root / "Carbon Emissions Futures Historical Data.csv"
    "cross_border_dir": None,   # π.χ. project_root / "Raw Data" / "Cross_Border_Flows"
    "outages_dir": None,        # θα οριστεί όταν έχεις όλα τα raw αρχεία
    "hydro_dir": None,          # π.χ. project_root / "Raw Data" / "Hydro_Reservoirs"

    # Φάκελος όπου γράφονται τα ενδιάμεσα/τελικά processed αρχεία
    "processed_dir": None,      # π.χ. project_root / "Dataset_Creation" / "processed"

    # Flags
    "run_outages_stage": True,  # άλλαξέ το σε True όταν είναι έτοιμο το στάδιο outages

    # 1-day (24ω) καθυστέρηση ορατότητας για ΟΛΑ τα outages (planned + forced).
    # Λόγος: το raw αρχείο δεν έχει πεδίο "πότε δημοσιεύτηκε" η εγγραφή, μόνο
    # την πραγματική περίοδο του outage. Για forced (failure) outages αυτό θα
    # ήταν leakage αν χρησιμοποιούνταν ως έχει. Λύση: κάθε outage γίνεται
    # ορατό στο feature μόνο 24 ώρες μετά την καταγεγραμμένη έναρξή του.
    "outage_visibility_lag_hours": 24,

    # Πλήθος blocks για το blocked walk-forward OOF forecasting του net_out
    # (στάδιο 8) — περισσότερα blocks = πιο ρεαλιστική προσομοίωση rolling
    # retraining, αλλά πιο αργό. ~1 block/μήνα είναι καλή ισορροπία.
    "net_out_forecast_n_blocks": 40,
}

# Συμπλήρωση default τιμών με βάση το project_root, όπου δεν έχουν οριστεί ρητά.
# Όλοι οι raw φάκελοι βρίσκονται μέσα σε project_root / "Raw Data" / ...
_root = CONFIG["project_root"]
_raw = _root / "Raw Data"
CONFIG["dam_2026_dir"] = CONFIG["dam_2026_dir"] or (_raw / "DAM 2026")
CONFIG["admie_dir"] = CONFIG["admie_dir"] or (_raw / "ΑΔΜΗΕ_Folder")
CONFIG["ngas_dir"] = CONFIG["ngas_dir"] or (_raw / "Τιμές Καυσίμων")
CONFIG["carbon_file"] = CONFIG["carbon_file"] or (_raw / "Τιμές Καυσίμων" / "Carbon Emissions Futures Historical Data.csv")
CONFIG["cross_border_dir"] = CONFIG["cross_border_dir"] or (_raw / "Cross_Border_Flows")
CONFIG["outages_dir"] = CONFIG["outages_dir"] or (_raw / "Unavailability")
CONFIG["hydro_dir"] = CONFIG["hydro_dir"] or (_raw / "Hydro_Reservoirs")
CONFIG["processed_dir"] = CONFIG["processed_dir"] or (_root / "Dataset_Creation" / "processed")

CONFIG["processed_dir"].mkdir(parents=True, exist_ok=True)


def _stage_path(stage_name: str) -> Path:
    """Βοηθητική: γυρνάει το path για ένα ενδιάμεσο αρχείο σταδίου."""
    return CONFIG["processed_dir"] / f"{stage_name}.csv"


# ---------------------------------------------------------------------------
# Στάδιο 1 — Base dataset
# ---------------------------------------------------------------------------
def load_base() -> pd.DataFrame:
    print("[1/8] Φόρτωση base dataset (2023-2025)...")
    df = pd.read_csv(CONFIG["base_file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = _stage_path("01_base")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 2 — DAM 2026 + ΑΔΜΗΕ Load/RES
# ---------------------------------------------------------------------------
def _process_admie_file(file_path: Path, type_name: str) -> pd.Series:
    """
    Ανάγνωση αρχείου ΑΔΜΗΕ (15λεπτα) και μετατροπή σε ωριαία σειρά.

    ΣΗΜΕΙΩΣΗ / ΓΝΩΣΤΟΣ ΠΕΡΙΟΡΙΣΜΟΣ:
    Η ανάγνωση εδώ είναι position-based (στήλη 3 = ημερομηνία, στήλες 4-99 =
    96 τιμές 15λέπτου). Αυτό δουλεύει με τη σημερινή μορφή αρχείων ΑΔΜΗΕ, αλλά
    θα σπάσει σιωπηλά αν αλλάξει η μορφή του αρχείου. Το κρατάμε ως έχει προς
    το παρόν (δεν έχουμε δει εναλλακτική/πιο σταθερή μορφή raw αρχείου), αλλά
    αξίζει επανεξέταση αν αλλάξουν ποτέ τα raw αρχεία ΑΔΜΗΕ.
    """
    try:
        raw = pd.read_excel(file_path, header=None)
        for _, row in raw.iterrows():
            val = str(row[3])
            if "-" in val and len(val) >= 10:
                date_str = val[:10]
                values = pd.to_numeric(row[4:100], errors="coerce").values
                time_index = pd.date_range(start=date_str, periods=96, freq="15min")
                series = pd.Series(values, index=time_index, name=type_name)
                return series.resample("h").mean()
    except Exception as e:
        print(f"    !! Σφάλμα στο αρχείο {file_path.name}: {e}")
    return pd.Series(dtype=float, name=type_name)


def add_dam_2026(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/8] Ενσωμάτωση DAM 2026 + ΑΔΜΗΕ Load/RES...")
    df = df.copy()

    # --- DAM 2026 τιμές ---
    # Φιλτράρισμα βάσει ονόματος ("*DAM*"), όχι μόνο κατάληξης — έτσι δεν
    # σκάει αν τυχόν βρεθούν και Load/RES αρχεία στον ίδιο φάκελο.
    dam_files = [f for f in CONFIG["dam_2026_dir"].rglob("*.xlsx") if "DAM" in f.name.upper()] if CONFIG["dam_2026_dir"].exists() else []
    if dam_files:
        df_dam = pd.concat(
            [pd.read_excel(f, sheet_name="EL-DAM_Results") for f in dam_files],
            ignore_index=True,
        )
        df_dam["timestamp"] = pd.to_datetime(df_dam["DELIVERY_MTU"]).dt.floor("h")
        df_dam = df_dam.groupby("timestamp", as_index=False)["MCP"].mean()

        df = pd.merge(df, df_dam, on="timestamp", how="outer")
        df["mcp_eur_per_mwh"] = df["mcp_eur_per_mwh"].combine_first(df["MCP"])
        df = df.drop(columns=["MCP"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        print(f"    DAM 2026: {len(dam_files)} αρχεία ενσωματώθηκαν.")
    else:
        print("    DAM 2026: δεν βρέθηκαν αρχεία, παραλείπεται.")

    # --- ΑΔΜΗΕ Load & RES ---
    admie_dir = CONFIG["admie_dir"]
    if admie_dir.exists():
        load_files = [f for f in admie_dir.rglob("*Load*") if f.is_file() and not f.name.startswith("~")]
        res_files = [f for f in admie_dir.rglob("*RES*") if f.is_file() and not f.name.startswith("~")]

        if load_files or res_files:
            load_s = pd.concat([_process_admie_file(f, "load") for f in load_files]) if load_files else pd.Series(name="load", dtype=float)
            res_s = pd.concat([_process_admie_file(f, "res") for f in res_files]) if res_files else pd.Series(name="res", dtype=float)

            admie_df = pd.concat([load_s, res_s], axis=1).reset_index().rename(columns={"index": "timestamp"})
            df = pd.merge(df, admie_df, on="timestamp", how="left")

            if "load" in df.columns:
                df["load_forecast_mw"] = df["load_forecast_mw"].combine_first(df["load"])
                df = df.drop(columns=["load"])
            if "res" in df.columns:
                df["res_forecast_mw"] = df["res_forecast_mw"].combine_first(df["res"])
                df = df.drop(columns=["res"])

            df["net_load_proxy_mw"] = df["load_forecast_mw"] - df["res_forecast_mw"]
            print(f"    ΑΔΜΗΕ: {len(load_files)} load + {len(res_files)} RES αρχεία ενσωματώθηκαν.")
    else:
        print("    ΑΔΜΗΕ: ο φάκελος δεν βρέθηκε, παραλείπεται.")

    out_path = _stage_path("02_with_2026_prices")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 3 — Φυσικό αέριο (NGAS) + Ρύποι (carbon)
# ---------------------------------------------------------------------------
def add_fuel_and_carbon(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/8] Ενσωμάτωση τιμών φυσικού αερίου (NGAS) + ρύπων (carbon)...")
    df = df.copy()
    df["_date"] = df["timestamp"].dt.normalize()

    # --- NGAS ---
    ngas_dir = CONFIG["ngas_dir"]
    ngas_files = list(ngas_dir.rglob("*.xlsx")) if ngas_dir.exists() else []
    if ngas_files:
        df_ngas = pd.concat(
            [pd.read_excel(f, sheet_name="NGAS_Results") for f in ngas_files],
            ignore_index=True,
        )
        df_ngas["_date"] = pd.to_datetime(df_ngas["Trading Date"]).dt.normalize()
        df_ngas = df_ngas.drop_duplicates(subset=["_date"], keep="last")

        df = pd.merge(df, df_ngas[["_date", "Closing Price"]], on="_date", how="left")
        if "NGAS_Price" in df.columns:
            df["NGAS_Price"] = df["NGAS_Price"].combine_first(df["Closing Price"])
            df = df.drop(columns=["Closing Price"])
        else:
            df = df.rename(columns={"Closing Price": "NGAS_Price"})
        print(f"    NGAS: {len(ngas_files)} αρχεία ενσωματώθηκαν.")
    else:
        print("    NGAS: δεν βρέθηκαν αρχεία, παραλείπεται.")

    # --- Carbon (CO2) ---
    carbon_file = CONFIG["carbon_file"]
    if carbon_file.exists():
        df_co2 = pd.read_csv(carbon_file)
        df_co2.columns = df_co2.columns.str.strip()
        df_co2["_date"] = pd.to_datetime(df_co2["Date"], format="%m/%d/%Y", errors="coerce")

        if df_co2["Price"].dtype == object:
            df_co2["Price"] = df_co2["Price"].str.replace(",", "").astype(float)

        df_co2 = (
            df_co2[["_date", "Price"]]
            .rename(columns={"Price": "carbon_price_eur"})
            .sort_values("_date")
            .dropna(subset=["_date"])
        )

        df = pd.merge(df, df_co2, on="_date", how="left")
        df = df.sort_values("timestamp")
        # Forward/backward fill μόνο για μέρες χωρίς trading (π.χ. Σαββατοκύριακα)
        df["carbon_price_eur"] = df["carbon_price_eur"].ffill().bfill()
        print("    Carbon: ενσωματώθηκε.")
    else:
        print("    Carbon: το αρχείο δεν βρέθηκε, παραλείπεται.")

    df = df.drop(columns=["_date"]).sort_values("timestamp").reset_index(drop=True)

    out_path = _stage_path("03_with_fuel_carbon")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 4 — Calendar features (σωστά υπολογισμένα από το timestamp)
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/8] Προσθήκη calendar features (hour, month, day_of_week, is_weekend)...")
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=Δευτέρα, 6=Κυριακή
    # ΔΙΟΡΘΩΣΗ: υπολογίζεται απευθείας από το timestamp — ΟΧΙ θέση-βασισμένο
    # join με ξεχωριστό αρχείο, όπως γινόταν πριν (ρίσκο μετατόπισης γραμμών).
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    out_path = _stage_path("04_with_calendar")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 5 — Cross-border flows (net_out ΠΑΡΑΜΕΝΕΙ ως feature)
# ---------------------------------------------------------------------------
def add_cross_border_flows(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/8] Ενσωμάτωση cross-border flows (net_out)...")
    df = df.copy()

    cb_dir = CONFIG["cross_border_dir"]
    cb_files = sorted(cb_dir.glob("Cross_Border_*.csv")) if cb_dir.exists() else []

    if cb_files:
        frames = []
        for f in cb_files:
            cb = pd.read_csv(f, low_memory=False)
            # ΔΙΟΡΘΩΣΗ (σημαντική): η στήλη "timestamp" σε αυτά τα raw αρχεία
            # είναι ΧΑΛΑΣΜΕΝΗ — δεν αντιπροσωπεύει τον πραγματικό χρόνο
            # παράδοσης (αυξάνεται ανά ΓΡΑΜΜΗ του αρχείου, όχι ανά πραγματική
            # ώρα, πιθανό τεχνούργημα από "τράβηγμα" τύπου σε Excel). Η
            # ΣΩΣΤΗ στήλη είναι το "MTU" (Market Time Unit), το πραγματικό
            # διάστημα παράδοσης — επιβεβαιώθηκε ρητά ότι είναι σε CET/CEST
            # (το ίδιο το αρχείο το γράφει: π.χ. γύρω από αλλαγή ώρας βλέπουμε
            # "...01:00:00 (CET) - ...03:00:00 (CEST)").
            #
            # ΔΕΝ εφαρμόζουμε κανένα time-shift εδώ: επιβεβαιώθηκε ότι EnEx
            # (DAM), ΑΔΜΗΕ (Load/RES) ΚΑΙ ENTSO-E (cross-border) χρησιμοποιούν
            # όλα CET/CEST — άρα είναι ήδη συνεπή μεταξύ τους χωρίς μετατόπιση.
            mtu_start_str = cb["MTU"].str.split(" - ").str[0]
            # Αφαίρεση τυχόν επισήμανσης "(CET)"/"(CEST)" που εμφανίζεται
            # ρητά μόνο στις γραμμές γύρω από αλλαγή ώρας.
            mtu_start_str = mtu_start_str.str.replace(r"\s*\((CET|CEST)\)", "", regex=True)
            cb["timestamp"] = pd.to_datetime(mtu_start_str, dayfirst=True)

            # ΔΙΟΡΘΩΣΗ (σημαντική, 2ο επίπεδο): ούτε η προϋπολογισμένη στήλη
            # "net_out" ΟΥΤΕ οι wide-format στήλες ροών (π.χ. "Greece -
            # Albania") καλύπτουν παρά μόνο τις πρώτες ~5 εβδομάδες του
            # έτους — και οι δύο είναι σχεδόν άδειες μετά από αυτό. Η στήλη
            # που ΕΙΝΑΙ πλήρης σε όλο το έτος (επιβεβαιώθηκε: 87.588/87.600
            # γραμμές) είναι η "Physical Flow (MW)" σε long-format (μία
            # γραμμή ανά ζεύγος Out Area/In Area). Υπολογίζουμε το net_out
            # ΕΜΕΙΣ από αυτήν: άθροισμα εξαγωγών (Out Area == Greece) μείον
            # άθροισμα εισαγωγών (In Area == Greece) — επιβεβαιωμένο ότι
            # ταιριάζει ακριβώς με τις λίγες υπάρχουσες τιμές του πρωτότυπου
            # net_out (π.χ. 01/01/2023 00:00 -> -1993 και στα δύο).
            cb["Physical Flow (MW)"] = pd.to_numeric(
                cb["Physical Flow (MW)"].astype(str).str.replace(",", "", regex=False).str.strip(),
                errors="coerce",
            )
            export = cb[cb["Out Area"] == "Greece (GR)"].groupby("timestamp")["Physical Flow (MW)"].sum()
            imp = cb[cb["In Area"] == "Greece (GR)"].groupby("timestamp")["Physical Flow (MW)"].sum()
            net_out_series = (export - imp).rename("net_out")
            hourly = net_out_series.reset_index()
            frames.append(hourly.dropna())

        net_out = pd.concat(frames, ignore_index=True)
        net_out = net_out.groupby("timestamp", as_index=False)["net_out"].mean()

        df = pd.merge(df, net_out, on="timestamp", how="left")

        # ΔΙΟΡΘΩΣΗ: πριν, το net_out χρησιμοποιούνταν μόνο για dropna και μετά
        # πετιόταν — δεν έμπαινε ποτέ ως feature στο μοντέλο. Τώρα παραμένει.
        # Για τις ώρες χωρίς διαθέσιμο flow data, γεμίζουμε με 0 (παραδοχή:
        # "χωρίς καταγεγραμμένη ροή" -> ουδέτερη τιμή). ΑΞΙΖΕΙ ΕΠΙΒΕΒΑΙΩΣΗ:
        # αν το 0 είναι η κατάλληλη default τιμή για το δικό σου use case, ή
        # προτιμάς interpolation/ffill.
        missing_before = df["net_out"].isna().sum()
        df["net_out"] = df["net_out"].fillna(0.0)
        print(f"    net_out: {len(cb_files)} αρχεία, {missing_before} ώρες συμπληρώθηκαν με 0.")
    else:
        print("    Cross-border flows: δεν βρέθηκαν αρχεία, παραλείπεται (net_out δεν προστίθεται).")

    out_path = _stage_path("05_with_flows")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 6 — Unit outages / unavailability
# ---------------------------------------------------------------------------
def _load_outage_files(outages_dir: Path) -> pd.DataFrame:
    """
    Διαβάζει ΟΛΑ τα raw αρχεία unavailability (και οι δύο ονοματολογίες:
    'UNAVAILABILITY_Nth_...' και 'UNAVAILABILITY_OF_PRODUCTION_...') και τα
    ενώνει σε ένα ενιαίο DataFrame. Και τα δύο "στυλ" αρχείων έχουν
    πανομοιότυπη δομή στηλών (επιβεβαιώθηκε στα δεδομένα).
    """
    files = sorted(outages_dir.rglob("UNAVAILABILITY*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def add_outages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ενσωματώνει τη συνολική μη διαθέσιμη ισχύ (MW) ανά ώρα, από ΟΛΑ τα
    outages -- προγραμματισμένη συντήρηση ΚΑΙ απρόβλεπτες βλάβες (failures)
    -- με ένα 1-day VISIBILITY LAG για να αποφευχθεί leakage.

    ΙΣΤΟΡΙΚΟ / ΔΙΟΡΘΩΣΗ: η προηγούμενη έκδοση φιλτράριζε με
    Status == "Active planned", με την υπόθεση ότι το πεδίο Status
    κωδικοποιεί planned/forced. Ελέγχοντας τα πραγματικά δεδομένα βρέθηκε
    ότι αυτό ΔΕΝ ισχύει: πολλές γραμμές με Status == "Active planned"
    έχουν στην πραγματικότητα Reason == "Failure" (δηλαδή ήταν ήδη
    forced/απρόβλεπτες βλάβες, απλά όχι με το Status που περιμέναμε). Το
    πεδίο που πραγματικά διακρίνει planned/forced είναι το "Reason"
    ("Foreseen Maintenance" vs "Failure"), όχι το "Status".

    ΝΕΑ ΑΠΟΦΑΣΗ (μαζί με τον χρήστη): αντί να προσπαθήσουμε να
    φιλτράρουμε planned/forced σωστά, τα παίρνουμε ΟΛΑ μαζί (εξαιρώντας
    μόνο Status == "Cancelled", που δεν αντιπροσωπεύει πραγματική μη
    διαθεσιμότητα). Για να αποφευχθεί leakage από forced outages που δεν
    θα ήταν ρεαλιστικά γνωστά τη στιγμή μιας day-ahead πρόβλεψης, κάθε
    outage γίνεται "ορατό" στο feature μόνο `outage_visibility_lag_hours`
    ώρες ΜΕΤΑ την καταγεγραμμένη έναρξή του (βλ. CONFIG). Η πρώτη μέρα
    ενός outage δεν αποτυπώνεται καθόλου -- συνειδητό, τεκμηριωμένο
    trade-off (μικρή απώλεια πληροφορίας, μηδενικό ρίσκο leakage) αντί να
    προσπαθήσουμε να μαντέψουμε πότε ακριβώς δημοσιεύτηκε κάθε εγγραφή
    (πληροφορία που δεν υπάρχει στο raw αρχείο).

    Λοιπές αποφάσεις (αμετάβλητες από πριν):
    - Χρησιμοποιούμε τη στήλη "Time Interval (CET/CEST)" (το πραγματικό,
      μη επικαλυπτόμενο υπο-διάστημα ανά γραμμή) — ΟΧΙ το "Start & End
      Time" (που είναι απλά το συνολικό διάστημα της αναγγελίας, ίδιο σε
      πολλές γραμμές, και θα οδηγούσε σε τεράστια υπερεκτίμηση αν
      χρησιμοποιούνταν ως το πραγματικό παράθυρο κάθε τιμής MW).
    - Η στήλη "Type" (Generation unit / Production unit) αγνοείται στο
      άθροισμα — επιβεβαιώθηκε ότι αντανακλά αλλαγή σύμβασης αναφοράς στον
      χρόνο (διαφορετικός κωδικός EIC ανά εποχή), όχι διπλή καταμέτρηση της
      ίδιας μονάδας (τα διαστήματα δεν επικαλύπτονται).
    - Timestamps στο αρχείο δηλώνονται ρητά ως CET/CEST -> +1h shift σε
      τοπική ώρα Ελλάδας, ίδια λογική με τα cross-border flows.
    """
    if not CONFIG["run_outages_stage"]:
        print("[6/8] Outages stage: ΑΠΕΝΕΡΓΟΠΟΙΗΜΕΝΟ (run_outages_stage=False) — παραλείπεται.")
        return df

    lag_hours = CONFIG["outage_visibility_lag_hours"]
    print(f"[6/8] Ενσωμάτωση outages (planned + forced, με {lag_hours}h visibility lag)...")
    df = df.copy()

    outages_dir = CONFIG["outages_dir"]
    raw = _load_outage_files(outages_dir) if outages_dir and outages_dir.exists() else pd.DataFrame()

    if raw.empty:
        print("    Outages: δεν βρέθηκαν αρχεία, παραλείπεται (outage_mw δεν προστίθεται).")
        out_path = _stage_path("06_with_outages")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} γραμμές)")
        return df

    n_total = len(raw)
    outages = raw[raw["Status"] != "Cancelled"].copy()
    print(f"    Σύνολο γραμμών: {n_total}  ->  Active (planned+forced): {len(outages)}"
          f"  (αγνοήθηκαν {n_total - len(outages)} λόγω Cancelled)")

    outages["Installed (MW)"] = pd.to_numeric(outages["Installed (MW)"], errors="coerce")
    outages["Available (MW)"] = pd.to_numeric(outages["Available (MW)"], errors="coerce")
    outages["unavailable_mw"] = (outages["Installed (MW)"] - outages["Available (MW)"]).clip(lower=0)

    # Χρειαζόμαστε "Time Interval" ΚΑΙ έγκυρο unavailable_mw
    outages = outages.dropna(subset=["Time Interval (CET/CEST)", "unavailable_mw"])

    split = outages["Time Interval (CET/CEST)"].str.split(" - ", expand=True)
    # +1h: CET/CEST -> τοπική ώρα Ελλάδας (ίδιο με πριν)
    outages["interval_start"] = pd.to_datetime(split[0], dayfirst=True) + pd.Timedelta(hours=1)
    outages["interval_end"] = pd.to_datetime(split[1], dayfirst=True) + pd.Timedelta(hours=1)

    # ΝΕΟ: +lag_hours πάνω στο ήδη tz-shifted interval_start -- αυτό είναι
    # η ώρα από την οποία το outage θεωρείται "ορατό"/γνωστό στο μοντέλο.
    outages["visible_from"] = outages["interval_start"] + pd.Timedelta(hours=lag_hours)
    n_before_lag_filter = len(outages)
    outages = outages[outages["interval_end"] > outages["visible_from"]]
    print(f"    Μετά το visibility lag: {len(outages)} / {n_before_lag_filter} γραμμές παραμένουν "
          f"(οι υπόλοιπες ήταν πολύ σύντομες -- τελείωσαν πριν γίνουν καν ορατές).")

    unit_key = outages["Unit Code"].fillna(outages["Unit Name"])

    # Expand κάθε γραμμή σε ωριαία χρονοσειρά με τη σταθερή τιμή unavailable_mw,
    # ΞΕΚΙΝΩΝΤΑΣ από visible_from αντί για interval_start.
    expanded_frames = []
    for (u_key, vis_start, end, mw) in zip(unit_key, outages["visible_from"], outages["interval_end"], outages["unavailable_mw"]):
        hours = pd.date_range(vis_start.floor("h"), end - pd.Timedelta(seconds=1), freq="h")
        if len(hours) == 0:
            continue
        expanded_frames.append(pd.DataFrame({"unit": u_key, "timestamp": hours, "unavailable_mw": mw}))

    if not expanded_frames:
        print("    Outages: καμία έγκυρη γραμμή μετά το φιλτράρισμα, παραλείπεται.")
        out_path = _stage_path("06_with_outages")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} γραμμές)")
        return df

    expanded = pd.concat(expanded_frames, ignore_index=True)

    # Πρώτα ομαδοποίηση ανά (μονάδα, ώρα) -> μέσος όρος, ώστε τυχόν διπλές/
    # επικαλυπτόμενες γραμμές για ΤΗΝ ΙΔΙΑ μονάδα να μην αθροίζονται εσφαλμένα.
    per_unit_hourly = expanded.groupby(["unit", "timestamp"], as_index=False)["unavailable_mw"].mean()

    # Μετά άθροιση σε όλες τις μονάδες -> συνολική μη διαθέσιμη ισχύ ανά ώρα
    outage_mw = per_unit_hourly.groupby("timestamp", as_index=False)["unavailable_mw"].sum()
    outage_mw = outage_mw.rename(columns={"unavailable_mw": "outage_mw"})

    df = pd.merge(df, outage_mw, on="timestamp", how="left")
    missing_before = df["outage_mw"].isna().sum()
    # Ώρες χωρίς καταγεγραμμένη (ή ακόμα μη-ορατή) διακοπή -> 0.
    df["outage_mw"] = df["outage_mw"].fillna(0.0)
    print(f"    outage_mw: {len(outages)} γραμμές (planned+forced, lagged), "
          f"{missing_before} ώρες χωρίς outage συμπληρώθηκαν με 0.")

    out_path = _stage_path("06_with_outages")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 7 — Στάθμη ταμιευτήρων υδροηλεκτρικών (ENTSO-E, εβδομαδιαία)
# ---------------------------------------------------------------------------
def add_hydro_reservoir(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ενσωματώνει τη στάθμη πλήρωσης ταμιευτήρων/υδροηλεκτρικών αποθήκευσης
    (ENTSO-E dataset [16.1.D], "Water Reservoirs and Hydro Storage Plants"),
    plus την εβδομαδιαία μεταβολή της στάθμης.

    ΚΡΙΣΙΜΗ ΔΙΟΡΘΩΣΗ LEAKAGE: το ENTSO-E δίνει ΕΒΔΟΜΑΔΙΑΙΟ μέσο όρο -- ένας
    τέτοιος μέσος όρος υπολογίζεται ΜΟΝΟ αφού περάσει ολόκληρη η εβδομάδα.
    Αν χρησιμοποιούσαμε "τη στάθμη ΑΥΤΗΣ της εβδομάδας" για ώρες ΜΕΣΑ σε
    αυτήν, θα χρησιμοποιούσαμε πληροφορία από μέρες ΜΕΤΑ την υποτιθέμενη
    πρόβλεψη -- ίδιο πρόβλημα με το net_out, πιο δύσκολο να το προσέξει
    κανείς. Διόρθωση: χρησιμοποιούμε τη στάθμη της ΠΡΟΗΓΟΥΜΕΝΗΣ, ήδη
    ολοκληρωμένης εβδομάδας (shift(1) πριν το expand σε ωριαία βάση).
    """
    print("[7/8] Ενσωμάτωση στάθμης ταμιευτήρων υδροηλεκτρικών...")
    df = df.copy()

    hydro_dir = CONFIG["hydro_dir"]
    hydro_files = list(hydro_dir.rglob("*.csv")) if hydro_dir and hydro_dir.exists() else []

    if not hydro_files:
        print("    Hydro reservoir: δεν βρέθηκαν αρχεία, παραλείπεται.")
        out_path = _stage_path("07_with_hydro")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} γραμμές)")
        return df

    raw = pd.concat([pd.read_csv(f) for f in hydro_files], ignore_index=True)
    raw["week_start"] = pd.to_datetime(raw["Time Interval"].str.split(" - ").str[0], dayfirst=True)
    raw["week_end"] = pd.to_datetime(raw["Time Interval"].str.split(" - ").str[1], dayfirst=True)
    raw = raw.drop_duplicates(subset=["week_start"]).sort_values("week_start").reset_index(drop=True)
    raw["Energy (MWh)"] = pd.to_numeric(raw["Energy (MWh)"], errors="coerce")
    raw = raw.dropna(subset=["Energy (MWh)"])

    # ΚΡΙΣΙΜΟ: χρησιμοποιούμε την τιμή της ΠΡΟΗΓΟΥΜΕΝΗΣ γραμμής (ήδη
    # ολοκληρωμένη εβδομάδα) για το εύρος ημερομηνιών ΑΥΤΗΣ της γραμμής.
    raw["energy_lagged"] = raw["Energy (MWh)"].shift(1)
    # Η μεταβολή υπολογίζεται ΚΙ ΑΥΤΗ πάνω σε ήδη-lagged τιμές -> παραμένει
    # χωρίς leakage.
    raw["energy_change_lagged"] = raw["energy_lagged"].diff()

    raw = raw.dropna(subset=["energy_lagged"])

    rows = []
    for _, r in raw.iterrows():
        hours = pd.date_range(r["week_start"], r["week_end"] - pd.Timedelta(hours=1), freq="h")
        for h in hours:
            rows.append((h, r["energy_lagged"], r["energy_change_lagged"]))

    hourly = pd.DataFrame(rows, columns=["timestamp", "hydro_reservoir_mwh", "hydro_reservoir_change"])
    hourly = hourly.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    df = pd.merge(df, hourly, on="timestamp", how="left")

    # Οι πρώτες ~1-2 εβδομάδες του dataset (πριν υπάρχει καν "προηγούμενη"
    # ολοκληρωμένη εβδομάδα) γεμίζουν με την πρώτη διαθέσιμη τιμή -- μικρή,
    # τεκμηριωμένη παραδοχή μόνο για αυτό το αρχικό διάστημα.
    missing_before = df["hydro_reservoir_mwh"].isna().sum()
    df["hydro_reservoir_mwh"] = df["hydro_reservoir_mwh"].bfill().ffill()
    df["hydro_reservoir_change"] = df["hydro_reservoir_change"].bfill().ffill()

    print(f"    Hydro reservoir: {len(hydro_files)} αρχεία, {missing_before} αρχικές ώρες "
          f"συμπληρώθηκαν με την πρώτη διαθέσιμη τιμή.")

    out_path = _stage_path("07_with_hydro")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Στάδιο 8 — Πρόβλεψη (όχι η πραγματική τιμή) του net_out
# ---------------------------------------------------------------------------
def add_net_out_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Προσθέτει `net_out_forecast`: μια ΠΡΟΒΛΕΨΗ του net_out, βασισμένη ΜΟΝΟ σε
    πληροφορία που είναι πραγματικά γνωστή τη στιγμή μιας day-ahead
    πρόβλεψης. Σε αντίθεση με το ακατέργαστο `net_out` (στάδιο 5, realized
    τιμές — leak αν χρησιμοποιηθεί απευθείας), αυτό ΕΙΝΑΙ ασφαλές ως model
    feature.

    ΓΙΑΤΙ ΑΞΙΖΕΙ (βρέθηκε εμπειρικά): ένα "oracle" test με τις πραγματικές
    net_out τιμές έδειξε μικρό, ασυνεπές όφελος (μερικές φορές αρνητικό).
    Ένα πραγματικό forecasting submodel όμως, ακριβώς επειδή "καθαρίζει"
    το net_out από τον μη προβλέψιμο θόρυβο και κρατάει μόνο το
    προβλέψιμο/εποχιακό κομμάτι, έδωσε σταθερά καλύτερο MAE στο κύριο
    μοντέλο σε 5 από 6 test periods που δοκιμάστηκαν (2y/3y training
    window × H1 2025/H2 2025/H1 2026) — σε ένα σημείο μάλιστα καλύτερο κι
    από το ίδιο το oracle.

    ΜΕΘΟΔΟΛΟΓΙΑ / ΓΙΑΤΙ BLOCKED WALK-FORWARD:
    Το net_out_forecast είναι το ΜΟΝΟ feature σε αυτό το pipeline που είναι
    το ίδιο αποτέλεσμα ενός μοντέλου, όχι ένας άμεσος υπολογισμός πάνω σε
    raw δεδομένα (σε αντίθεση με τα lags/rolling means, που είναι απλά
    shift()). Αυτό σημαίνει ότι δεν μπορεί να υπολογιστεί με ένα ενιαίο
    fit πάνω σε όλο το ιστορικό dataset -- κάτι τέτοιο θα έκανε leak
    πληροφορία από το μέλλον σε παλιότερες γραμμές (το μοντέλο θα είχε
    εκπαιδευτεί και σε δεδομένα ΜΕΤΑ από κάθε δεδομένη γραμμή).

    Λύση: blocked walk-forward, ίδια λογική με το outlier classifier
    πείραμα που κάναμε νωρίτερα. Το dataset χωρίζεται σε
    `net_out_forecast_n_blocks` διαδοχικά κομμάτια. Για κάθε block (εκτός
    του πρώτου), εκπαιδεύεται ένα μοντέλο πάνω σε ΟΛΑ τα προηγούμενα
    blocks, και προβλέπει το τρέχον block. Αυτό είναι μια προσέγγιση του
    πραγματικού ημερήσιου rolling retraining (πιο αδρή -- ένα μοντέλο ανά
    block, όχι ανά μέρα -- αλλά χωρίς leakage, και πολύ πιο γρήγορο).

    Το ΠΡΩΤΟ block (warmup, χωρίς προηγούμενο ιστορικό) δεν έχει
    net_out_forecast -- αυτές οι γραμμές παίρνουν fallback = 0
    (ισοδύναμο με "καμία πληροφορία exports"), τεκμηριωμένος περιορισμός.

    Χρησιμοποιεί ΜΟΝΟ features που είναι πραγματικά γνωστά day-ahead:
    ημερολογιακά, load/RES forecasts, τιμές καυσίμων/ρύπων, και lagged/
    rolling τιμές του ΙΔΙΟΥ του net_out (shift(24)+ πριν, άρα ασφαλή).
    """
    print("[8/8] Πρόβλεψη net_out (blocked walk-forward, ασφαλές ως feature)...")
    df = df.copy()

    if "net_out" not in df.columns or df["net_out"].isna().all():
        print("    net_out_forecast: δεν υπάρχει διαθέσιμο net_out στο dataset, παραλείπεται.")
        df["net_out_forecast"] = 0.0
        out_path = _stage_path("08_with_net_out_forecast")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} γραμμές)")
        return df

    # Lagged/rolling features του net_out -- ήδη shift()-based, ασφαλή.
    df["net_out_lag24h"] = df["net_out"].shift(24)
    df["net_out_lag48h"] = df["net_out"].shift(48)
    df["net_out_lag168h"] = df["net_out"].shift(168)
    df["net_out_rollmean7d"] = df["net_out"].shift(24).rolling(24 * 7).mean()

    forecast_features = [
        "hour", "month", "day_of_week", "is_weekend",
        "load_forecast_mw", "res_forecast_mw", "NGAS_Price", "carbon_price_eur",
        "net_out_lag24h", "net_out_lag48h", "net_out_lag168h", "net_out_rollmean7d",
    ]
    forecast_features = [c for c in forecast_features if c in df.columns]

    valid = df.dropna(subset=forecast_features + ["net_out"]).copy()
    valid = valid.sort_values("timestamp").reset_index(drop=True)

    n_blocks = CONFIG["net_out_forecast_n_blocks"]
    block_bounds = np.linspace(0, len(valid), n_blocks + 1).astype(int)

    import xgboost as xgb
    reg_params = dict(objective="reg:squarederror", max_depth=5, learning_rate=0.08,
                       n_estimators=200, subsample=0.8, colsample_bytree=0.8, random_state=42)

    net_out_forecast = np.full(len(valid), np.nan)
    for b in range(1, n_blocks):
        tr_idx = slice(0, block_bounds[b])
        te_idx = slice(block_bounds[b], block_bounds[b + 1])
        if block_bounds[b + 1] - block_bounds[b] == 0:
            continue
        m = xgb.XGBRegressor(**reg_params)
        m.fit(valid.iloc[tr_idx][forecast_features], valid.iloc[tr_idx]["net_out"])
        net_out_forecast[block_bounds[b]:block_bounds[b + 1]] = m.predict(valid.iloc[te_idx][forecast_features])

    valid["net_out_forecast"] = net_out_forecast
    n_warmup = block_bounds[1]

    df = df.merge(valid[["timestamp", "net_out_forecast"]], on="timestamp", how="left")
    n_missing = df["net_out_forecast"].isna().sum()
    df["net_out_forecast"] = df["net_out_forecast"].fillna(0.0)

    # Τα βοηθητικά lag/rolling columns δεν χρειάζονται πλέον στο τελικό dataset
    df = df.drop(columns=["net_out_lag24h", "net_out_lag48h", "net_out_lag168h", "net_out_rollmean7d"])

    print(f"    net_out_forecast: {n_blocks} blocks, {n_warmup} γραμμές warmup (χωρίς forecast, ->0), "
          f"{n_missing} γραμμές συνολικά συμπληρώθηκαν με 0.")

    out_path = _stage_path("08_with_net_out_forecast")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} γραμμές)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_dataset() -> pd.DataFrame:
    df = load_base()
    df = add_dam_2026(df)
    df = add_fuel_and_carbon(df)
    df = add_calendar_features(df)
    df = add_cross_border_flows(df)
    df = add_outages(df)
    df = add_hydro_reservoir(df)
    df = add_net_out_forecast(df)

    final_path = CONFIG["processed_dir"] / "final_dataset.csv"
    df.to_csv(final_path, index=False)
    print(f"\nΟλοκληρώθηκε! Τελικό dataset: {final_path}  ({len(df)} γραμμές, {len(df.columns)} στήλες)")
    return df


if __name__ == "__main__":
    build_dataset()