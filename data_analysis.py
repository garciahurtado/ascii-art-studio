import os

import pandas as pd


def main():
    print(os.getcwd())
    df = pd.read_csv("lib/datasets/ascii_amstrad_cpc/data/class_counts.csv", names=['class', 'count'])
    print(df.describe())


if __name__ == "__main__":
    main()
