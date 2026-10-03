"""
build_dataset_complete.py
=========================
Builds the dataset for Greek day-ahead (DAM) price forecasting, stage by stage.

Each stage takes the previous stage's output and writes a NEW file (never
overwriting its input), so every intermediate stage stays available for
inspection and reproducibility.

Stages:
    01_base                  -> base dataset (2023-2025) as provided
    02_with_2026_prices      -> + DAM 2026 prices (HEnEx), + ADMIE load/RES forecasts 2026
    03_with_fuel_carbon      -> + natural gas price, + EUA carbon price
    04_with_calendar         -> + hour, month, day_of_week, is_weekend
    05_with_flows            -> + net_out (realized cross-border flows) -- kept only as
                                raw/reference data, NOT safe as a model feature (stage 8)
    06_with_outages          -> + unit outages / unavailability, with a visibility lag
    07_with_hydro            -> + hydro reservoir level (ENTSO-E, weekly, previous week)
    08_with_net_out_forecast -> + net_out_forecast: a walk-forward FORECAST of net_out
                                (not the realized value) -- safe as a model feature

All timestamps are hourly. Paths are relative to this file (raw data in
"Raw Data/", outputs in "Dataset_Creation/processed/").

Usage:  python build_dataset_complete.py
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.simplefilter("ignore")


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent

CONFIG = {
    "project_root": REPO,

    # Base dataset (2023-2025): DAM prices with load/RES forecasts, already cleaned
    "base_file": REPO / "dam_prices_with_load_res_forecasts_clean (1).csv",

    # Raw data folders (None = default location under "Raw Data/", set below).
    # Two folder names are in Greek, as downloaded: "ΑΔΜΗΕ_Folder" (ADMIE)
    # and "Τιμές Καυσίμων" ("fuel prices").
    "dam_2026_dir": None,
    "admie_dir": None,
    "ngas_dir": None,
    "carbon_file": None,
    "cross_border_dir": None,
    "outages_dir": None,
    "hydro_dir": None,

    # Where the intermediate and final processed files are written
    "processed_dir": None,

    "run_outages_stage": True,

    # Outage visibility lag. The raw unavailability files have no "published
    # at" field, only the outage period itself. Using forced outages (failures)
    # from their start time would leak information a day-ahead forecaster did
    # not have, so every outage only becomes visible this many hours after its
    # recorded start. (The forecasting scripts add a further 12 h, for 36 h in
    # total, so that nothing starting after the ~12:00 D-1 forecast time is
    # visible to any hour of day D.)
    "outage_visibility_lag_hours": 24,

    # Number of blocks for the blocked walk-forward forecast of net_out
    # (stage 8). More blocks = closer to daily retraining, but slower.
    # ~1 block per month is a good balance.
    "net_out_forecast_n_blocks": 40,
}

# Fill in default locations for anything not set explicitly above.
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
    """Path of an intermediate stage file."""
    return CONFIG["processed_dir"] / f"{stage_name}.csv"


# ---------------------------------------------------------------------------
# Stage 1 — Base dataset
# ---------------------------------------------------------------------------
def load_base() -> pd.DataFrame:
    print("[1/8] Loading base dataset (2023-2025)...")
    df = pd.read_csv(CONFIG["base_file"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    out_path = _stage_path("01_base")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 2 — DAM 2026 prices + ADMIE load/RES forecasts
# ---------------------------------------------------------------------------
def _process_admie_file(file_path: Path, type_name: str) -> pd.Series:
    """
    Read one ADMIE file (15-minute values) and convert it to an hourly series.

    Known limitation: parsing is position-based (column 3 = date, columns
    4-99 = the 96 quarter-hour values). This matches the current ADMIE file
    layout but would break silently if the layout changed.
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
        print(f"    !! Error in file {file_path.name}: {e}")
    return pd.Series(dtype=float, name=type_name)


def add_dam_2026(df: pd.DataFrame) -> pd.DataFrame:
    print("[2/8] Adding DAM 2026 prices + ADMIE load/RES...")
    df = df.copy()

    # --- DAM 2026 prices ---
    # Filter on the name ("*DAM*"), not only the extension, so that any
    # load/RES files in the same folder are not read as price files.
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
        print(f"    DAM 2026: {len(dam_files)} files added.")
    else:
        print("    DAM 2026: no files found, skipped.")

    # --- ADMIE load & RES forecasts ---
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
            print(f"    ADMIE: {len(load_files)} load + {len(res_files)} RES files added.")
    else:
        print("    ADMIE: folder not found, skipped.")

    out_path = _stage_path("02_with_2026_prices")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 3 — Natural gas (NGAS) + carbon (EUA)
# ---------------------------------------------------------------------------
def add_fuel_and_carbon(df: pd.DataFrame) -> pd.DataFrame:
    print("[3/8] Adding natural gas (NGAS) and carbon prices...")
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
        print(f"    NGAS: {len(ngas_files)} files added.")
    else:
        print("    NGAS: no files found, skipped.")

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
        # Forward/backward fill only for non-trading days (e.g. weekends)
        df["carbon_price_eur"] = df["carbon_price_eur"].ffill().bfill()
        print("    Carbon: added.")
    else:
        print("    Carbon: file not found, skipped.")

    df = df.drop(columns=["_date"]).sort_values("timestamp").reset_index(drop=True)

    out_path = _stage_path("03_with_fuel_carbon")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 4 — Calendar features (computed from the timestamp)
# ---------------------------------------------------------------------------
def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    print("[4/8] Adding calendar features (hour, month, day_of_week, is_weekend)...")
    df = df.copy()

    df["hour"] = df["timestamp"].dt.hour
    df["month"] = df["timestamp"].dt.month
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0 = Monday, 6 = Sunday
    # Computed directly from the timestamp, not joined by position from a
    # separate file as in an earlier version (risk of shifted rows).
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

    out_path = _stage_path("04_with_calendar")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 5 — Cross-border flows (realized net_out, reference only)
# ---------------------------------------------------------------------------
def add_cross_border_flows(df: pd.DataFrame) -> pd.DataFrame:
    print("[5/8] Adding cross-border flows (net_out)...")
    df = df.copy()

    cb_dir = CONFIG["cross_border_dir"]
    cb_files = sorted(cb_dir.glob("Cross_Border_*.csv")) if cb_dir.exists() else []

    if cb_files:
        frames = []
        for f in cb_files:
            cb = pd.read_csv(f, low_memory=False)
            # The "timestamp" column in these raw files is broken: it advances
            # per ROW of the file, not per real hour (likely an Excel fill-down
            # artefact). The correct time is "MTU" (Market Time Unit), the
            # actual delivery interval, which the file itself marks as CET/CEST
            # around clock changes (e.g. "...01:00:00 (CET) - ...03:00:00 (CEST)").
            #
            # No time shift is applied: HEnEx (DAM), ADMIE (load/RES) and
            # ENTSO-E (flows) all use CET/CEST, so they are already aligned.
            mtu_start_str = cb["MTU"].str.split(" - ").str[0]
            # Strip the "(CET)"/"(CEST)" markers that appear only around
            # clock changes.
            mtu_start_str = mtu_start_str.str.replace(r"\s*\((CET|CEST)\)", "", regex=True)
            cb["timestamp"] = pd.to_datetime(mtu_start_str, dayfirst=True)

            # Neither the precomputed "net_out" column nor the wide-format
            # per-border columns (e.g. "Greece - Albania") cover more than the
            # first ~5 weeks of each year. The column that IS complete
            # (87,588 of 87,600 rows) is "Physical Flow (MW)" in long format,
            # one row per (Out Area, In Area) pair. net_out is computed from
            # it: total exports (Out Area == Greece) minus total imports
            # (In Area == Greece). Checked to match the few existing values of
            # the original net_out exactly (e.g. 01/01/2023 00:00 -> -1993).
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

        # Realized flows are published after delivery, so net_out is kept
        # only as the target of the stage-8 forecast, never as a model
        # feature. Hours without flow data are filled with 0 (no recorded
        # flow = neutral value).
        missing_before = df["net_out"].isna().sum()
        df["net_out"] = df["net_out"].fillna(0.0)
        print(f"    net_out: {len(cb_files)} files, {missing_before} hours filled with 0.")
    else:
        print("    Cross-border flows: no files found, skipped (net_out not added).")

    out_path = _stage_path("05_with_flows")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 6 — Unit outages / unavailability
# ---------------------------------------------------------------------------
def _load_outage_files(outages_dir: Path) -> pd.DataFrame:
    """
    Read ALL raw unavailability files (both naming styles,
    'UNAVAILABILITY_Nth_...' and 'UNAVAILABILITY_OF_PRODUCTION_...', which
    have identical columns) into one DataFrame.
    """
    files = sorted(outages_dir.rglob("UNAVAILABILITY*.csv"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def add_outages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total unavailable capacity (MW) per hour from ALL outages -- planned
    maintenance AND unplanned failures -- with a visibility lag to avoid
    leakage.

    Planned vs forced: an earlier version kept only Status == "Active
    planned", assuming Status encodes planned/forced. The data shows it does
    not: many such rows have Reason == "Failure". The field that does
    separate them is "Reason" ("Foreseen Maintenance" vs "Failure").
    Decision: keep ALL outages (excluding only Status == "Cancelled", which
    is not real unavailability), and make each one visible only
    `outage_visibility_lag_hours` after its recorded start (see CONFIG). The
    first day of every outage is therefore never seen -- a deliberate
    trade-off (a little lost information, no leakage) instead of guessing
    when each record was published, which the raw files do not say.

    Other decisions:
    - "Time Interval (CET/CEST)" is used (the real, non-overlapping
      sub-interval of each row), NOT "Start & End Time" (the whole
      announcement period, repeated across many rows, which would hugely
      overstate unavailability).
    - "Type" (Generation unit / Production unit) is ignored when summing: it
      reflects a reporting convention that changed over time (different EIC
      codes in different periods), not the same unit counted twice (the
      intervals do not overlap).
    - Times are shifted by +1 h. Note this differs from stage 5, which
      treats all sources as CET/CEST without a shift; the effect is that
      outages appear one hour later than recorded, which only delays their
      visibility further (conservative, no leakage).
    """
    if not CONFIG["run_outages_stage"]:
        print("[6/8] Outages stage disabled (run_outages_stage=False), skipped.")
        return df

    lag_hours = CONFIG["outage_visibility_lag_hours"]
    print(f"[6/8] Adding outages (planned + forced, {lag_hours} h visibility lag)...")
    df = df.copy()

    outages_dir = CONFIG["outages_dir"]
    raw = _load_outage_files(outages_dir) if outages_dir and outages_dir.exists() else pd.DataFrame()

    if raw.empty:
        print("    Outages: no files found, skipped (outage_mw not added).")
        out_path = _stage_path("06_with_outages")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} rows)")
        return df

    n_total = len(raw)
    outages = raw[raw["Status"] != "Cancelled"].copy()
    print(f"    Rows: {n_total}  ->  active (planned+forced): {len(outages)}"
          f"  ({n_total - len(outages)} cancelled rows ignored)")

    outages["Installed (MW)"] = pd.to_numeric(outages["Installed (MW)"], errors="coerce")
    outages["Available (MW)"] = pd.to_numeric(outages["Available (MW)"], errors="coerce")
    outages["unavailable_mw"] = (outages["Installed (MW)"] - outages["Available (MW)"]).clip(lower=0)

    # Need both a "Time Interval" and a valid unavailable_mw
    outages = outages.dropna(subset=["Time Interval (CET/CEST)", "unavailable_mw"])

    split = outages["Time Interval (CET/CEST)"].str.split(" - ", expand=True)
    # +1 h shift (see docstring)
    outages["interval_start"] = pd.to_datetime(split[0], dayfirst=True) + pd.Timedelta(hours=1)
    outages["interval_end"] = pd.to_datetime(split[1], dayfirst=True) + pd.Timedelta(hours=1)

    # The time from which the outage counts as known to the model.
    outages["visible_from"] = outages["interval_start"] + pd.Timedelta(hours=lag_hours)
    n_before_lag_filter = len(outages)
    outages = outages[outages["interval_end"] > outages["visible_from"]]
    print(f"    After the visibility lag: {len(outages)} / {n_before_lag_filter} rows remain "
          f"(the rest ended before becoming visible).")

    unit_key = outages["Unit Code"].fillna(outages["Unit Name"])

    # Expand each row into an hourly series with constant unavailable_mw,
    # STARTING at visible_from rather than interval_start.
    expanded_frames = []
    for (u_key, vis_start, end, mw) in zip(unit_key, outages["visible_from"], outages["interval_end"], outages["unavailable_mw"]):
        hours = pd.date_range(vis_start.floor("h"), end - pd.Timedelta(seconds=1), freq="h")
        if len(hours) == 0:
            continue
        expanded_frames.append(pd.DataFrame({"unit": u_key, "timestamp": hours, "unavailable_mw": mw}))

    if not expanded_frames:
        print("    Outages: no valid rows after filtering, skipped.")
        out_path = _stage_path("06_with_outages")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} rows)")
        return df

    expanded = pd.concat(expanded_frames, ignore_index=True)

    # Average per (unit, hour) first, so duplicate/overlapping rows for the
    # SAME unit are not summed twice.
    per_unit_hourly = expanded.groupby(["unit", "timestamp"], as_index=False)["unavailable_mw"].mean()

    # Then sum over units -> total unavailable capacity per hour
    outage_mw = per_unit_hourly.groupby("timestamp", as_index=False)["unavailable_mw"].sum()
    outage_mw = outage_mw.rename(columns={"unavailable_mw": "outage_mw"})

    df = pd.merge(df, outage_mw, on="timestamp", how="left")
    missing_before = df["outage_mw"].isna().sum()
    # Hours with no recorded (or not yet visible) outage -> 0.
    df["outage_mw"] = df["outage_mw"].fillna(0.0)
    print(f"    outage_mw: {len(outages)} rows (planned+forced, lagged), "
          f"{missing_before} hours without outages filled with 0.")

    out_path = _stage_path("06_with_outages")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 7 — Hydro reservoir level (ENTSO-E, weekly)
# ---------------------------------------------------------------------------
def add_hydro_reservoir(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hydro reservoir filling level (ENTSO-E [16.1.D], "Water Reservoirs and
    Hydro Storage Plants") and its weekly change.

    Leakage: ENTSO-E reports a WEEKLY value, which only exists once the week
    is over. Using "this week's level" for hours inside the week would use
    information from after the forecast. Every hour therefore gets the level
    of the PREVIOUS, completed week (shift(1) before expanding to hours).
    """
    print("[7/8] Adding hydro reservoir levels...")
    df = df.copy()

    hydro_dir = CONFIG["hydro_dir"]
    hydro_files = list(hydro_dir.rglob("*.csv")) if hydro_dir and hydro_dir.exists() else []

    if not hydro_files:
        print("    Hydro reservoir: no files found, skipped.")
        out_path = _stage_path("07_with_hydro")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} rows)")
        return df

    raw = pd.concat([pd.read_csv(f) for f in hydro_files], ignore_index=True)
    raw["week_start"] = pd.to_datetime(raw["Time Interval"].str.split(" - ").str[0], dayfirst=True)
    raw["week_end"] = pd.to_datetime(raw["Time Interval"].str.split(" - ").str[1], dayfirst=True)
    raw = raw.drop_duplicates(subset=["week_start"]).sort_values("week_start").reset_index(drop=True)
    raw["Energy (MWh)"] = pd.to_numeric(raw["Energy (MWh)"], errors="coerce")
    raw = raw.dropna(subset=["Energy (MWh)"])

    # Use the PREVIOUS row's value (a completed week) for THIS row's dates.
    raw["energy_lagged"] = raw["Energy (MWh)"].shift(1)
    # The change is computed on the already-lagged values, so it is safe too.
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

    # The first ~1-2 weeks of the dataset (before any completed previous
    # week exists) are filled with the first available value -- a small
    # assumption limited to the start of the data, which the models never
    # train on (their windows start later).
    missing_before = df["hydro_reservoir_mwh"].isna().sum()
    df["hydro_reservoir_mwh"] = df["hydro_reservoir_mwh"].bfill().ffill()
    df["hydro_reservoir_change"] = df["hydro_reservoir_change"].bfill().ffill()

    print(f"    Hydro reservoir: {len(hydro_files)} files, {missing_before} initial hours "
          f"filled with the first available value.")

    out_path = _stage_path("07_with_hydro")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
    return df


# ---------------------------------------------------------------------------
# Stage 8 — Forecast (not the realized value) of net_out
# ---------------------------------------------------------------------------
def add_net_out_forecast(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds `net_out_forecast`: a FORECAST of net exports, built only from
    information known at day-ahead forecast time (~12:00 on D-1). Unlike the
    realized `net_out` from stage 5 (a leak if used directly), this is safe
    as a model feature.

    Why a forecast rather than nothing: an "oracle" test with the realized
    values showed a small, inconsistent gain; a forecasting submodel, which
    keeps only the predictable (seasonal/structural) part of the flows, gave
    a better main-model MAE in 5 of 6 test periods tried.

    Blocked walk-forward: this is the only feature that is itself a model
    output. Fitting it once on the full history would leak future
    information into earlier rows, so the data is split into
    `net_out_forecast_n_blocks` consecutive blocks; each block is predicted
    by a model trained on all EARLIER blocks only. The first block (no
    history) gets no forecast and falls back to 0.

    Inputs: calendar, load/RES forecasts, gas and carbon prices, and lagged
    realized flows. The flow lags are at least 48 h: realized flows are
    published after delivery, so at ~12:00 on D-1 the flows for the afternoon
    and evening of D-1 are not yet known, and a 24 h lag would leak them for
    hours 12-23 of day D.
    """
    print("[8/8] Forecasting net_out (blocked walk-forward, safe as a feature)...")
    df = df.copy()

    if "net_out" not in df.columns or df["net_out"].isna().all():
        print("    net_out_forecast: no net_out in the dataset, skipped.")
        df["net_out_forecast"] = 0.0
        out_path = _stage_path("08_with_net_out_forecast")
        df.to_csv(out_path, index=False)
        print(f"    -> {out_path}  ({len(df)} rows)")
        return df

    # Lagged/rolling realized flows, at least 48 h old (see docstring).
    df["net_out_lag48h"] = df["net_out"].shift(48)
    df["net_out_lag72h"] = df["net_out"].shift(72)
    df["net_out_lag168h"] = df["net_out"].shift(168)
    df["net_out_rollmean7d"] = df["net_out"].shift(48).rolling(24 * 7).mean()

    forecast_features = [
        "hour", "month", "day_of_week", "is_weekend",
        "load_forecast_mw", "res_forecast_mw", "NGAS_Price", "carbon_price_eur",
        "net_out_lag48h", "net_out_lag72h", "net_out_lag168h", "net_out_rollmean7d",
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

    # The helper lag/rolling columns are not needed in the final dataset
    df = df.drop(columns=["net_out_lag48h", "net_out_lag72h", "net_out_lag168h", "net_out_rollmean7d"])

    print(f"    net_out_forecast: {n_blocks} blocks, {n_warmup} warm-up rows (no forecast, -> 0), "
          f"{n_missing} rows filled with 0 in total.")

    out_path = _stage_path("08_with_net_out_forecast")
    df.to_csv(out_path, index=False)
    print(f"    -> {out_path}  ({len(df)} rows)")
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
    print(f"\nDone. Final dataset: {final_path}  ({len(df)} rows, {len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    build_dataset()