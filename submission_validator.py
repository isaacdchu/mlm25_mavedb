'''
This script validates whether a submission file meets the required format and content standards.
It checks for:
1. Valid file path
2. CSV file type
3. Correct shape (118421, 2)
4. Correct columns ("accession", "score")
5. Numeric values in "score" column
6. No NaN values in "score" column
7. Valid and in-order accessions in "accession" column as per "data/sample_submission.csv"
'''

import sys
from pathlib import Path
import pandas as pd

def validate(submission: str) -> tuple[bool, str]:
    """
    Validates a submission file.
    The submission file must meet the aforementioned criteria to be considered valid.
    If any of the criteria are not met, returns False and a message indicating the issues.
    Args:
        submission (str): Path to the submission file.
    Returns:
        tuple[bool, str]: A tuple containing a boolean indicating success or failure,
                            and a message string with details about the validation.
    """
    success: bool = True
    message: str = ""

    # Verify file path
    try:
        submission_path: Path = Path(submission).resolve()
    except OSError as e:
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
            message += f"Submission file has incorrect shape: \
            {df.shape}, expected {correct_shape}\n"
        correct_columns = ("accession", "score")
        if tuple(df.columns) != correct_columns:
            success = False
            message += f"Submission file has incorrect columns: \
                {tuple(df.columns)}, expected {correct_columns}\n"
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
            message += "Submission file 'accession' column contains \
                invalid/out-of-order accession(s)\n"
    except (FileNotFoundError, pd.errors.ParserError, ValueError) as e:
        success = False
        message += f"Error reading CSV file: {e}\n"
        return (success, message)

    return (success, message)

def main(*args) -> None:
    """ 
    Main function for validating submission files.
    Multiple submission files can be validated in one run through command line arguments.
    Args:
        *args: Variable length argument list of submission file paths.
    Returns:
        None
    """
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
    sys.exit(0)

if __name__ == "__main__":
    main(*sys.argv[1:])
