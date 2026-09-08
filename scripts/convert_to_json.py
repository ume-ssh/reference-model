import csv
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

root_dir = Path().cwd()
data_dir = root_dir / "data"


def main(file_name: str):
    df = pd.read_csv(data_dir / f"{file_name}.csv", encoding="cp932")
    df.to_json(data_dir / f"{file_name}.json", orient="records")

    print("Convert done...")


if __name__ == "__main__":
    main("202601_delay_7days_rcscore")
