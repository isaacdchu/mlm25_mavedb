import pandas as pd

df = pd.read_csv("data/random_submission.csv")
print(df.shape)
print(tuple(df.columns) == ("accession", "score"))