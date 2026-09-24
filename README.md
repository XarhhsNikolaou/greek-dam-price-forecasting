# Greek Day-Ahead Electricity Price Forecasting

Hourly forecasts of the Greek day-ahead market (DAM) clearing price, made the
day before delivery, using XGBoost on market fundamentals and past prices.

**Result on a full held-out year (July 2025 – June 2026, 8,736 hours):
MAE 15.74 €/MWh, RMSE 24.25 €/MWh — 31% lower error than the
"same hour yesterday" baseline (22.70) and 44% lower than
"same hour last week" (28.07).**

The model combination was chosen on a separate selection year (July 2024 –
June 2025) and frozen in Git before the test year was scored, so the test
result involves no selection on test data.

## How the forecast is made

The Greek DAM clears around midday on day *D−1* for all 24 hours of day *D*.
Every feature below is known at that point.

The final forecast combines two model families, both retrained every day on a
rolling 18-month window with median (quantile 0.5) XGBoost regression:

| Hours | Model | Idea |
|---|---|---|
| 00:00–03:00 | **Early-hour chain** (one model per hour) | The 23:00 price of *D−1* is already published at forecast time and is the strongest predictor of 00:00. Each hour's prediction then feeds the next hour's model. |
| 04:00–23:00 | **General + peak-hour model** | One model for all hours, with a specialised model for the peak hours 10–12 and 18–20 that also sees how spiky recent prices were. |

### Features

| Group | Features |
|---|---|
| Demand and renewables (day-ahead forecasts) | load forecast, RES forecast, load minus RES, total net load (load + forecast net exports − RES) |
| Fuel and carbon costs | natural gas price, EUA carbon price |
| Calendar | hour, month, day of week, weekend flag |
| Hydro | reservoir level and weekly change (previous completed week) |
| Cross-border flows | walk-forward **forecast** of net exports |
| Past prices | same hour 1, 2 and 7 days earlier, 25 h earlier, 1-day and 7-day rolling means, weekly max/min |
| Peak-hour model only | skewness and kurtosis of prices over *D−1* and the previous week |
| Early-hour chain only | previous 1–2 hours' price (published *D−1* prices or the chain's own predictions), planned outages delayed 36 h |

### Leakage decisions

- **Realized cross-border flows are excluded.** ENTSO-E's physical flows are
  metered and published after the fact. A walk-forward forecast of net exports
  (each block predicted only from earlier blocks) is used instead.
- **Only planned outages are used, delayed 36 hours**, so no outage that starts
  after the forecast time is visible to any hour of day *D*.
- **Model selection never touches the test year** (next section).

## Evaluation protocol

1. **Candidates.** Four model families were run over July 2024 – June 2026 with
   the same daily walk-forward retraining (`run_candidates.py`): two variants of
   the general + peak-hour model (A: original features, B: adds total net load
   and the peak-hour distribution features) and two early-hour chains
   (C and D, on each feature set).
2. **Selection year, July 2024 – June 2025** (`select_rule.py`). The rule was
   chosen here only:
   - A and B tied (MAE 20.36 each); B was kept for the whole year. A seasonal
     switch was allowed only if one model won at least 60% of a season's days;
     neither did (54% and 52%), so no switch.
   - The early-hour chain beat the general model at 00:00–03:00 and lost from
     04:00, so it covers hours 0–3. Chain C edged out D.
3. **Rule frozen** in `results/selection_rule.json` and committed before
   step 4.
4. **Test year, July 2025 – June 2026** (`evaluate_test.py`), scored once.

The selection year had far more volatile prices (July 2024 standard deviation
110 €/MWh versus 50 in July 2025), which is why its errors are higher.

## Results (test year)

| | MAE (€/MWh) |
|---|---|
| **Final model** | **15.74** |
| Same hour yesterday | 22.70 |
| Same hour last week | 28.07 |

| Part of the day | Hours | MAE |
|---|---|---|
| Early-hour chain | 00–03 | 8.96 |
| General + peak model | 04–23 | 17.09 |

By month, errors range from 12.1 (August 2025) to 19.8 (October 2025).

## Project history

- **v1** built the data pipeline (`build_dataset_complete.py`, eight documented
  stages from the base dataset to the final one) and the first rolling
  models.
- **v2** added total net load, the peak-hour distribution features and the
  early-hour chain, and reported MAE 15.20. A later review found two problems
  with that number: v2 used realized cross-border flows (leakage), and the
  final model mix had been chosen on the same year it was evaluated on.
- **Current version** fixes both: forecast flows replace realized ones, and
  the combination is selected on a separate year and frozen before testing.
  The honest test-year MAE is 15.74, only 0.54 above the flawed figure.

The v1 and v2 scripts are kept under `experiments/` for reference.

## Data

The base dataset (DAM prices, load and RES forecasts, 2023–2025) comes from an
NTUA workshop on electricity price forecasting, extended with:

| Source | Data |
|---|---|
| HEnEx (Hellenic Energy Exchange) | 2026 DAM prices, natural gas prices |
| ADMIE / IPTO | 2026 day-ahead load and RES forecasts |
| ENTSO-E Transparency Platform | cross-border physical flows, unit unavailability, hydro reservoir levels |
| Investing.com | EUA carbon futures prices |

Known gaps with no satisfactory replacement source: 26 March 2023,
28 September – 1 October 2024, and 16–29 January 2025. Days in these gaps are
skipped in evaluation. The raw source files are not redistributed here.

## Repository layout

```
build_dataset_complete.py   data pipeline: raw files -> Dataset_Creation/processed/final_dataset.csv
                            (8 stages, each saved as its own CSV for inspection)
run_candidates.py           daily walk-forward run of one candidate model family
select_rule.py              chooses the combination rule on the selection year only
evaluate_test.py            applies the frozen rule to the test year
results/
  selection_rule.json       the frozen rule (committed before the test year was scored)
  test_year_summary.txt     output of evaluate_test.py
experiments/
  v1/                       first rolling models, spike / extreme-value (GPD) experiments,
                            classifier and hour-bucket attempts, early outlier handling
  v2/                       v2 feature set and the first early-hour chain scripts
```

The `experiments/` scripts are kept as a record of what was tried. They are
not needed to reproduce the results, and v2's `rolling_hybrid_v2.py` and
`__hour*_v2.py` still use realized flows (the leak described above).

## Reproducing

```bash
pip install -r requirements.txt

# 1. build the dataset from the raw files (Raw Data/ is not included in the repo)
python build_dataset_complete.py

# 2. run the four candidates (independent -- can run in parallel terminals)
python run_candidates.py --candidate A
python run_candidates.py --candidate B
python run_candidates.py --candidate C
python run_candidates.py --candidate D

# 3. choose the rule (selection year only), then score it on the test year
python select_rule.py
python evaluate_test.py
```

Each candidate takes roughly 30-60 minutes on a laptop (daily retraining over
two years). The dataset build takes a few minutes. Rebuilding
`net_out_forecast` with a different XGBoost version gives slightly different
values, so rerun results can differ from the committed ones in the decimals.

## Limitations

- **One selection year.** The combination rule rests on a single year, which
  gives one sample of each season. That is why seasonal switching required a
  day-level win rate, not just a lower average.
- **Price spikes.** Extreme hours (above 400 €/MWh) are not predictable from
  these features, and errors concentrate there.
- **Simulated forecast time.** The inputs are day-ahead forecasts and lagged
  values as archived, not captured live at 12:00 on *D−1*; small publication
  delays in the source data are not modelled.
