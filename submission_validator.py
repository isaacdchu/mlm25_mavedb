import sys
from pathlib import Path
import pandas as pd

def validate(submission: str) -> tuple[bool, str]:
    success: bool = True
    message: str = ""

    # Verify file path
    try:
        submission_path: Path = Path(submission).resolve()
    except Exception as e:
        success = False
        message += f"Error resolving submission path: {e}\n"
        return (success, message)
    
    # Verify file type
    if not submission_path.exists():
        success = False
        message += f"Submission path does not exist: {submission_path}\n"
    elif not submission_path.is_file():
        success = False
        message += f"Submission path is not a file: {submission_path}\n"
    elif submission_path.suffix != ".csv":
        success = False
        message += f"Submission path is not a CSV file: {submission_path}\n"
    if not success:
        return (success, message)
    
    # Verify file content
    try:
        df = pd.read_csv(
            submission_path,
            delimiter=",",
            header=0,
            on_bad_lines="error"
        )
        correct_shape = (118421, 2)
        if df.shape != correct_shape:
            success = False
            message += f"Submission file has incorrect shape: {df.shape}, expected {correct_shape}\n"
        correct_columns = ("accession", "score")
        if tuple(df.columns) != correct_columns:
            success = False
            message += f"Submission file has incorrect columns: {tuple(df.columns)}, expected {correct_columns}\n"
        if not all(isinstance(score, (float, int)) for score in df["score"]):
            success = False
            message += "Submission file 'score' column contains non-numeric values\n"
        if any(df["score"].isna()):
            success = False
            message += "Submission file 'score' column contains NaN values\n"
        accessions = pd.read_csv(
            Path("data/sample_submission.csv").resolve(),
            usecols=["accession"],
            dtype={"accession": str}
        )
        # Check order of accessions
        if any(accessions["accession"].iloc[i] != df["accession"].iloc[i] for i in range(len(df))):
            success = False
            message += "Submission file 'accession' column contains invalid/out-of-order accession(s)\n"
    except Exception as e:
        success = False
        message += f"Error reading CSV file: {e}\n"
        return (success, message)

    return (success, message)

def main(*args) -> None:
    print("----------------------------------------------------------------------")
    for submission in args:
        print(f"\tValidating submission:\t{submission}")
        is_valid, msg = validate(submission)
        if msg:
            print(msg, end="")
        if not is_valid:
            print(f"\tValidation failed:\t{submission}")
        else:
            print(f"\tValidation succeeded:\t{submission}")
        print("----------------------------------------------------------------------")
    exit(0)

if __name__ == "__main__":
    main(*sys.argv[1:])