import os
import pandas as pd
import numpy as np

sample_submission_path = os.path.abspath("data/sample_submission.csv")
df = pd.read_csv(sample_submission_path, dtype={"accession": str}, usecols=["accession"])
rng = np.random.default_rng(seed=0)
df["score"] = rng.normal(loc=0.731701, scale=4.619144, size=len(df))
df.to_csv("data/random_submission.csv", index=False)
