import pandas as pd
import matplotlib.pyplot as plt
import sys


def load(path: str) -> pd.DataFrame:
    """Load a dataset"""
    try:
        assert path.endswith(".csv"), "not a csv file"
        df = pd.read_csv(path)
        print(f"Loading dataset of dimensions {df.shape}")
        return df
    except AssertionError as e:
        print("AssertionError:", e)
        exit()
    except FileNotFoundError:
        print("FileNotFoundError:", "no such file or directory")
        exit()
