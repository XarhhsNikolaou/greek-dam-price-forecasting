"""
evaluate_test.py
================
Applies the frozen rule from results/selection_rule.json to the test year
(2025-07-01 -> 2026-06-30) and reports the final numbers, next to naive
baselines. Nothing here is tuned: run it once, after committing the rule.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
TEST_START, TEST_END = "2025-07-01", "2026-06-30 23:00"


def load(c: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / f"predictions_{c}.csv", parse_dates=["timestamp"]).set_index("timestamp")


if __name__ == "__main__":
    rule = json.loads((RESULTS_DIR / "selection_rule.json").read_text())
    A, B = load("A"), load("B")
    idx = A.index.intersection(B.index)
    idx = idx[(idx >= TEST_START) & (idx <= TEST_END)]
    actual = A.loc[idx, "actual"]

    month_to_model = {m: rule["general_by_season"][s] for s, ms in rule["seasons"].items() for m in ms}
    use_a = np.asarray(idx.month.map(month_to_model) == "A")
    final = pd.Series(np.where(use_a, A.loc[idx, "predicted"], B.loc[idx, "predicted"]), index=idx)
    source = pd.Series(np.where(use_a, "A", "B"), index=idx)

    if rule["chain_family"]:
        ch = load(rule["chain_family"])
        ch = ch[ch["hour"] <= rule["chain_last_hour"]]
        common = ch.index.intersection(idx)
        final.loc[common] = ch.loc[common, "predicted"]
        source.loc[common] = rule["chain_family"]

    err = final - actual
    print("=" * 60)
    print(f"TEST YEAR {TEST_START} -> {TEST_END[:10]}  ({len(idx)} hours)")
    print("=" * 60)
    print(f"Final model:  MAE = {err.abs().mean():.2f}  RMSE = {np.sqrt((err ** 2).mean()):.2f}  EUR/MWh")

    # Naive baselines on the same hours: same hour yesterday / last week.
    for name, lag in (("same hour D-1", "24h"), ("same hour D-7", "168h")):
        full = pd.concat([A["actual"], B["actual"]]).groupby(level=0).first().sort_index()
        naive = full.shift(freq=lag).reindex(idx)
        ok = naive.notna()
        print(f"Naive {name}: MAE = {(naive[ok] - actual[ok]).abs().mean():.2f}  "
              f"(model on the same {ok.sum()} hours: {err[ok].abs().mean():.2f})")

    print("\nBy source:")
    for s, e in err.groupby(source):
        print(f"  {s}: n={len(e):5d}  MAE={e.abs().mean():.2f}")
    print("\nBy month:")
    for m, e in err.groupby(idx.to_period("M")):
        print(f"  {m}: MAE={e.abs().mean():.2f}")

    out = pd.DataFrame({"actual": actual, "predicted": final, "source": source})
    out.to_csv(RESULTS_DIR / "test_year_final_predictions.csv")
    print(f"\nSaved results/test_year_final_predictions.csv")
