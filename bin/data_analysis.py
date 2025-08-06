import argparse
import os
from utils import mkpath
import pandas as pd

"""
Class count analysis
--------------------
This script is used to analyze the class counts of the dataset, as found in the class_counts.csv file.
"""
DATA_ROOT = "datasets"
CLASS_COUNTS_FILENAME = "class_counts.csv"

def main(dataset_name: str):
    print(os.getcwd())
    rel_path = mkpath(DATA_ROOT, dataset_name, "data", f"{dataset_name}_{CLASS_COUNTS_FILENAME}")
    full_path = mkpath(os.getcwd(), rel_path)

    if not os.path.exists(full_path):
        print(f"Class count file for dataset '{dataset_name}' does not exist at: {full_path}")
        exit(1)

    df = pd.read_csv(full_path, names=['class', 'count'])
    print(df.describe())

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("dataset", type=str)
    dataset_name = args.parse_args().dataset

    main(dataset_name)
