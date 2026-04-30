import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys


def load(path: str) -> pd.DataFrame:
    """Load a dataset"""
    try:
        assert path.endswith(".csv"), "not a csv file"
        df = pd.read_csv(path)
        # print(f"Loading dataset of dimensions {df.shape}")
        return df
    except AssertionError as e:
        print("AssertionError:", e)
        exit()
    except FileNotFoundError:
        print("FileNotFoundError:", "no such file or directory")
        exit()


try:
	assert len(sys.argv) == 2, "Usage : python3 linear_regression.py <dataset.csv>"
	df = load(sys.argv[1])
	# print(df)
	mileages = np.array(df["km"]).astype(float)
	price = np.array(df["price"]).astype(float)
    
	thetas = np.polyfit(mileages, price, 1)
	print("thetas0 :", thetas[1])
	print("thetas1 :", thetas[0])
except AssertionError as e:
	print(e)
except KeyboardInterrupt as e:
	pass
