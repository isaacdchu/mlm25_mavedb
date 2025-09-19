"""
We ran this baseline through kaggle on 9/18/2025 and recieved a score of 8.28148, which 
surpasses the OLS baseline score of 8.64961.


This program checks the scoreset and alt_long values, and returns the mean score from
the test data.

This can serve as a good simple baseline as we work on more complecated ML models in the future.
"""

# baseline_scoreset_altmean.py
from pathlib import Path
import pandas as pd
import csv  # for quoting option

ROOT  = Path.cwd()
TRAIN = ROOT / "data" / "train" / "raw_train.csv"
TEST  = ROOT / "data" / "test" / "raw_test.csv"
OUT   = ROOT / "preds" / "baseline_scoreset_altmean.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

# --- load ---
train = pd.read_csv(TRAIN)
test  = pd.read_csv(TEST)

# --- group means ---
mean_by_set_alt = (
    train.groupby(["scoreset", "alt_short"], dropna=False)["score"]
         .mean()
         .rename("mean_set_alt")
         .reset_index()
)
mean_by_set = train.groupby("scoreset", dropna=False)["score"].mean().rename("mean_set")
mean_by_alt = train.groupby("alt_short", dropna=False)["score"].mean().rename("mean_alt")
global_mean = float(train["score"].mean())

# --- merge (preserves test order) ---
pred = test[["accession", "scoreset", "alt_short"]].merge(
    mean_by_set_alt, on=["scoreset", "alt_short"], how="left", sort=False
)
pred = pred.merge(mean_by_set, on="scoreset", how="left", sort=False)
pred = pred.merge(mean_by_alt, on="alt_short", how="left", sort=False)

pred["score"] = (
    pred["mean_set_alt"]
        .fillna(pred["mean_set"])
        .fillna(pred["mean_alt"])
        .fillna(global_mean)
)

# --- write submission: accession,score ---
submission = pred[["accession", "score"]].copy()
submission["accession"] = submission["accession"].astype(str)

submission.to_csv(
    OUT,
    index=False,
    float_format="%.16f",   # plenty of precision
    lineterminator="\n",
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL,
)

print(f"Wrote {len(submission):,} rows to {OUT}")
