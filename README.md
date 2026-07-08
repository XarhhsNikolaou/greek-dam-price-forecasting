GREEK DAY-AHEAD MARKET (DAM) ELECTRICITY PRICE FORECASTING

=============================================================



STATUS: work in progress. This README documents where the project stands

today, including known limitations, honestly and without polishing over

the rough edges.





1\. BACKGROUND — A FALSE START

\-------------------------------

This project actually began earlier, as a first attempt that was quickly

abandoned. That first version started making price predictions before a

proper, complete data pipeline existed, and the codebase for building the

dataset had grown chaotic — many small, overlapping scripts, each doing

one ad-hoc transformation, with no clear single path from raw data to a

usable dataset. Rather than try to patch that structure, the data pipeline

was rebuilt from scratch with a clear, reproducible, single-script design

(see PIPELINE section below). This restart is mentioned here deliberately,

as part of an honest account of the project's actual history rather than

presenting it as if it were clean from day one.





2\. MOTIVATION

\--------------

The interest in this topic came from a workshop at the National Technical

University of Athens (NTUA) on electricity market price forecasting. The

dataset used in that workshop — historical Greek DAM prices together with

load and RES (renewables) forecasts — became the starting point (the

"base dataset") for this project, extended significantly since then with

additional data sources (see CORE MODELING LOGIC below). The goal is to

improve the simple model presented in the workshop to reach an average daily

MAE of 14 or under (the presenter stated that is a threshold that actual

companies want to achieve).



Known limitation: the original workshop dataset itself has three gaps

where source data was unavailable:

&#x20; - 26 March 2023 (1 day)

&#x20; - 28 September to 1 October 2024 (4 days)

&#x20; - 16 to 29 January 2025 (14 days)

No alternative source was found that could fill these gaps to a

satisfactory standard, so they remain as missing periods in the dataset.

This is treated as an accepted, documented limitation rather than a bug to

silently work around.





3\. CORE MODELING LOGIC

\------------------------

The starting principle is simple: the day-ahead market clearing price is

ultimately set by supply and demand. The feature set was built around

that idea — trying to include the factors that actually drive supply and

demand for Greek electricity, added to the dataset in roughly this order

as the project progressed:



&#x20; 1. Load forecast and RES (renewables) generation forecast — the demand

&#x20;    side and the largest, most variable supply component.

&#x20; 2. Fuel prices (natural gas) and carbon (EUA) prices — cost drivers for

&#x20;    conventional (thermal) generation, which sets the marginal price in

&#x20;    many hours.

&#x20; 3. Calendar features (hour of day, month, day of week, weekend flag) —

&#x20;    proxies for predictable demand patterns (daily/weekly seasonality).

&#x20; 4. Cross-border physical flows (imports/exports with neighboring

&#x20;    countries) — additional supply/demand balance information.

&#x20; 5. Planned (scheduled) unit outages — reductions in available thermal

&#x20;    generation capacity known in advance.



Two features were deliberately EXCLUDED from the actual model input,

despite being present in the built dataset, because they would not

realistically be available at the time a day-ahead prediction is made:

&#x20; - Cross-border flows (net\_out): the values in the underlying source are

&#x20;   REALIZED (actual) flows, only known after the fact — not forecasts.

&#x20;   Using them as-is would leak future information into the model.

&#x20; - Unplanned ("forced") unit outages: these are failures that, by

&#x20;   definition, are not known ahead of time. Only PLANNED outages

&#x20;   (scheduled maintenance, announced in advance) are used as a feature.

Both are kept in the dataset itself for completeness and possible future

use (e.g., once a way to forecast them is developed), just not fed to the

model today.



On top of these, standard time-series features were added for the model

itself: lagged prices (24h, 25h, 48h, 168h/1-week) and rolling price

averages (1-day, 7-day) — all computed using only information available

up to (not including) the hour being predicted, to avoid leakage.





4\. MODEL AND RESULTS

\----------------------

Model: XGBoost regression, trained on a rolling window of historical data,

predicting the DAM clearing price one day ahead.



Evaluation approach: rather than reporting MAE from a single arbitrary

day (which turned out to vary a lot — from around 4 to 60 depending on

which day was tested), the model is evaluated with a full rolling

evaluation: a separate prediction for every day of a full calendar year,

retraining on a rolling window each time, then aggregating MAE by month,

by season, and overall for the year.



Current Results:

\--- Predictions for 2025 ---

Average MAE: 16.38

Standard Deviation: 8.09

Min/Max: 4.19 / 58.78

Days Under with MAE under 14 (our goal): 46.3%





5\. REPOSITORY STRUCTURE (PLANNED)

\------------------------------------

&#x20; data/

&#x20;   raw/            - not committed; original downloaded files

&#x20;   processed/       - not committed; intermediate/final datasets

&#x20; src/

&#x20;   data\_pipeline/   - build\_dataset.py and related scripts

&#x20;   features/        - feature engineering

&#x20;   models/          - training and rolling evaluation

&#x20; experiments/       - earlier model iterations, kept for transparency

&#x20; requirements.txt

&#x20; .gitignore

&#x20; README.md





6\. DATA SOURCES

\-----------------

All data used is publicly available:

&#x20; - DAM prices: Hellenic Energy Exchange (EnEx/HEnEx), EL-DAM\_Results

&#x20;   publications

&#x20; - Load \& RES forecasts: ADMIE (Greek TSO) and ENTSO-E Transparency

&#x20;   Platform (load forecast only — confirmed to match ADMIE's own timing)

&#x20; - Cross-border physical flows: ENTSO-E Transparency Platform

&#x20; - Unit unavailability (outages): ENTSO-E Transparency Platform

&#x20; - Natural gas prices: EnEx Natural Gas prices publications

&#x20; - Carbon (EUA) futures prices: Investing.com historical data



A technical note on time zones, for anyone extending this project: the

DAM prices, ADMIE load/RES data, and ENTSO-E cross-border flow data all

appear to use CET/CEST (confirmed via how each source handles the March

daylight-saving transition, and in the ENTSO-E case, an explicit "(CET)"/

"(CEST)" label in the raw file itself) rather than true Greek local time

(EET/EEST, one hour ahead of CET). Since all sources are consistently in

the same convention, this does not affect model correctness — it only

means the "hour" label in the dataset is not literally true Greek wall-

clock time.





7\. KNOWN ISSUES / LIMITATIONS

\---------------------------------

&#x20; - Three data gaps in the base dataset (see MOTIVATION section).

&#x20; - A separate, unrelated single-hour quirk on three specific calendar

&#x20;   days (the DST transition day each year in 2024/2025/2026, \~3 hours

&#x20;   total across the whole dataset) where the base dataset appears to

&#x20;   use a different hour-skipping convention than expected; negligible

&#x20;   in size, noted here for completeness.

&#x20; - Cross-border flows and unplanned outages are excluded from the

&#x20;   current model (see CORE MODELING LOGIC) due to data-leakage concerns;

&#x20;   kept in the dataset for future use.





8\. ROADMAP

\------------

&#x20; - Forecast (rather than exclude) cross-border flows, so they can be

&#x20;   reintroduced as a legitimate day-ahead feature.

&#x20; - Explore ensembling / model combination once a stable, well-evaluated

&#x20;   baseline is established.

&#x20; - Error analysis: where and why the model underperforms (e.g. price

&#x20;   spikes).





LICENSE

\--------

Code will be made available under the MIT License. Data sources are

public; see DATA SOURCES above for attribution.

