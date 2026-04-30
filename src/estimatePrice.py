from load import load
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# TODO: Error management
def estimatePrice(mileage, theta0=0, theta1=0):
    """Estimate the Car price
    
    Parameters : mileage: float, theta0: float, theta1: float
    
    Return : Car price estimation: float"""
    return theta0 + (theta1 * mileage)

def thetas_tofloat():
    """Convert thetas from str to float"""
    try:
        if not os.path.exists("thetas.txt"):
            raise FileNotFoundError("Process linear regression before to get <thetas.txt>")
        file = open("thetas.txt", 'r')
        theta0 = file.readline()
        pos = theta0.find(":")
        theta0 = float(theta0[pos + 1:].strip())

        theta1 = file.readline()
        pos = theta1.find(":")
        theta1 = float(theta1[pos + 1:].strip())

        if theta0 == 0 and theta1 == 0:
            raise ValueError("thetas are 0")

        print("-" * 50)
        print("Thetas from Linear regression")
        print("theta0 :", theta0)
        print("theta1 :", theta1)
        print("-" * 50)

        return theta0, theta1
    except FileNotFoundError as e:
        print("FileNotFoundError :", e)
        exit()
    except ValueError as e:
        print("ValueError :", e)
        exit()

def plot_datas(mileage, mileages, price, theta0, theta1):
    """Plot linear regression graph and estimate price
    
    Parameters : mileage: float, mileages: ndarray, price: ndarray, theta0: float, theta1: float
    """
    x_values = np.linspace(0, 300000, 100)
    y_pred = (theta1 * x_values) + theta0
    x = mileage
    y = estimatePrice(mileage, theta0, theta1)
    plt.title("Linear regression")
    plt.plot(x_values, y_pred, color='red')
    plt.scatter(x, y, color='g', zorder=3)
    plt.plot([x, x], [0, y], linestyle='--', color=(0, 1, 0, 0.8))
    plt.plot([0, x], [y, y], linestyle='--', color=(0, 1, 0, 0.8))
    plt.scatter(mileages, price, color='b')
    plt.xlabel("mileage")
    plt.xlim(left=0, right=260000)
    plt.ylim(bottom=0)
    plt.ylabel("Price")
    plt.show()

def predict_price(theta0, theta1):
    """Predict the car price from thetas got after linear resgression
    
    Parameters : theta0: float, theta1: float
    
    Retrun : mileage: float"""
    try:
        mileage = float(input("Enter a mileage between 0 and 250 000 : "))
        assert mileage >= 0 and mileage <= 350000, "Enter mileage between 0 and 350000"
        
        print(f"For a mileage of \033[1;32m{mileage:.2f} km\033[0m, the price is ", end="")
        print(f"\033[1;32m{estimatePrice(mileage, theta0, theta1):.2f} Euros\033[1;0m.")

        return mileage
    except ValueError as e:
        print("ValueError :", e)
        exit()
    except AssertionError as e:
        print("AssertionError :", e)
        exit()

def main():
    try:
        assert len(sys.argv) == 2, "Usage : python3 linear_regression.py <dataset.csv>"
        df = load(sys.argv[1])
        print(df)
        mileages = np.array(df["km"]).astype(float)
        price = np.array(df["price"]).astype(float)

        theta0, theta1 = thetas_tofloat()

        mileage = predict_price(theta0, theta1)
        plot_datas(mileage, mileages, price, theta0, theta1)
    except AssertionError as e:
        print(e)
    except KeyboardInterrupt as e:
        pass


if __name__ == "__main__":
    main()
