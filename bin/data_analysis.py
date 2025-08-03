import os

import pandas as pd

"""
Class count analysis
--------------------
This script is used to analyze the class counts of the dataset, as found in the class_counts.csv file.
"""
def main():
    print(os.getcwd())
    df = pd.read_csv("lib/datasets/ascii_amstrad_cpc/data/class_counts.csv", names=['class', 'count'])
    print(df.describe())


if __name__ == "__main__":
    main()
