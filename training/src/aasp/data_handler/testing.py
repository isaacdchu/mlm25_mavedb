import sys
from data_handler import DataHandler

def main(*args) -> None:
    call_path: str = "/".join(args[0].split("/")[:-1])
    print(f"Called from {call_path}")
    DataHandler(f"{call_path}/config.yaml", "../data")  # Initialize the DataHandler with config
    return

if __name__ == "__main__":
    main(*sys.argv)
