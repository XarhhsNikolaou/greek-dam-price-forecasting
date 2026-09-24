"""
select_rule.py
==============
Decides how to combine the candidates, using ONLY the selection year
(2024-07-01 -> 2025-06-30). The test year is never read here.

Rule structure:
  1. General model (A or B), optionally switched by season. A season switch is
     only accepted if the season's winner beats the full-year winner on at
     least MIN_DAILY_WIN_RATE of that season's days -- one year gives a single
     sample of each season, so a small average gap alone is not enough.
  2. Early-hour chain (C or D, one family -- each chain feeds on its own
     earlier hours). Starting from 00:00, the chain is used for consecutive
     hours while it beats the selected general model; the first hour where it
     doesn't ends the chain.

Writes results/selection_rule.json. Commit that file BEFORE running
evaluate_test.py: the commit timestamp shows the rule was fixed before the
test year was looked at.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
SEL_START, SEL_END = "2024-07-01", "2025-06-30 23:00"
SEASONS = {"summer": [6, 7, 8], "rest": [1, 2, 3, 4, 5, 9, 10, 11, 12]}
MIN_DAILY_WIN_RATE = 0.60


def load(c: str) -> pd.Series:
    df = pd.read_csv(RESULTS_DIR / f"predictions_{c}.csv", parse_dates=["timestamp"])
    df = df[df["timestamp"].between(SEL_START, SEL_END)]
    return df.set_index("timestamp")


def mae(pred: pd.Series, actual: pd.Series) -> float:
    return float((pred - actual).abs().mean())


if __name__ == "__main__":
    A, B = load("A"), load("B")
    common = A.index.intersection(B.index)
    A, B = A.loc[common], B.loc[common]
    actual = A["actual"]

    print("=" * 60)
    print(f"SELECTION YEAR {SEL_START[:10]} -> {SEL_END[:10]}  ({len(common)} hours)")
    print("=" * 60)
    year_winner = "A" if mae(A.predicted, actual) <= mae(B.predicted, actual) else "B"
    print(f"General models, full year:  A={mae(A.predicted, actual):.2f}  B={mae(B.predicted, actual):.2f}"
          f"  -> {year_winner}")

    err = pd.DataFrame({"A": (A.predicted - actual).abs(), "B": (B.predicted - actual).abs()})
    daily = err.groupby(err.index.normalize()).mean()
    season_choice = {}
    for name, months in SEASONS.items():
        d = daily[daily.index.month.isin(months)]
        s_winner = "A" if d["A"].mean() <= d["B"].mean() else "B"
        win_rate = float((d[s_winner] < d["B" if s_winner == "A" else "A"]).mean())
        accept = s_winner == year_winner or win_rate >= MIN_DAILY_WIN_RATE
        season_choice[name] = s_winner if accept else year_winner
        print(f"  {name:7s}: A={d['A'].mean():.2f}  B={d['B'].mean():.2f}  winner={s_winner} "
              f"(wins {win_rate:.0%} of days) -> use {season_choice[name]}")

    month_to_model = {m: season_choice[s] for s, ms in SEASONS.items() for m in ms}
    general = pd.Series(np.where(common.month.map(month_to_model) == "A", A.predicted, B.predicted), index=common)

    print("\nEarly-hour chains vs the selected general model (MAE per hour):")
    best = None
    for fam in ["C", "D"]:
        ch = load(fam)
        cutoff, gain = -1, 0.0
        for h in sorted(ch["hour"].unique()):
            rows = ch[ch["hour"] == h]
            idx = rows.index.intersection(general.index)
            m_chain, m_gen = mae(rows.loc[idx, "predicted"], actual.loc[idx]), mae(general.loc[idx], actual.loc[idx])
            better = m_chain < m_gen
            print(f"  {fam} {h:02d}:00  chain={m_chain:.2f}  general={m_gen:.2f}  {'better' if better else 'worse'}")
            if better and cutoff == h - 1:
                cutoff, gain = int(h), gain + (m_gen - m_chain) * len(idx)
        print(f"  -> {fam}: chain used for hours 00..{cutoff:02d}, total gain {gain:.1f} EUR/MWh-hours")
        if cutoff >= 0 and (best is None or gain > best[2]):
            best = (fam, cutoff, gain)

    rule = {
        "general_by_season": season_choice,
        "seasons": SEASONS,
        "chain_family": best[0] if best else None,
        "chain_last_hour": best[1] if best else -1,
    }
    (RESULTS_DIR / "selection_rule.json").write_text(json.dumps(rule, indent=2))
    print(f"\nRule written to results/selection_rule.json:\n{json.dumps(rule, indent=2)}")
    print("\nCommit this file before running evaluate_test.py.")
