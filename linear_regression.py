from load import load
import matplotlib.pyplot as plt
import numpy as np
import sys


def cost(x, y, theta0, theta1):
    m = len(y)
    return np.sum((y - ((theta1 * x) + theta0)) ** 2) / m

def estimatePrice(mileage, theta0=0, theta1=0):
    return theta0 + (theta1 * mileage)

def normalize(array):
    normalized_data = (array - np.mean(array)) / np.std(array)
    return np.asarray(normalized_data)

def linear_regression(learning_rate, nb_iter):
    df = load(sys.argv[1])
    print(df)
    mileage = np.array(df["km"]).astype(float)
    print("-" * 50)
    print("mileage:\n", mileage)

    price = np.array(df["price"]).astype(float)
    print("-" * 50)
    print("price:\n", price)

    x_normalized = normalize(mileage)
    # y_normalized = normalize(df["price"].astype(float))
    print("-" * 50)
    print("x_normalized:\n", x_normalized)
    # print("-" * 50)
    # print("y_normalized:\n", y_normalized)

    costs = []
    alpha = learning_rate / len(mileage)
    theta0, theta1 = 0, 0
    for i in range(nb_iter):
        tmp0 = []
        tmp1 = []
        for j in range(len(mileage)):
            tmp0.append(estimatePrice(x_normalized[j], theta0, theta1) - price[j])
            tmp1.append(tmp0[j] * x_normalized[j])
        nptmp0 = np.array(tmp0)
        # print("-" * 50)
        # print("nptmp0:\n", nptmp0)
        nptmp1 = np.array(tmp1)
        # print("-" * 50)
        # print("nptmp1:\n", nptmp1)

        tmp_theta0 = theta0 - alpha * np.sum(nptmp0)
        tmp_theta1 = theta1 - alpha * np.sum(nptmp1)
        theta0 = tmp_theta0
        theta1 = tmp_theta1
        # plt.plot(i, theta0, 'o', color='b')
        # plt.plot(i, theta1, 'o', color='r')
        # costs.append(cost(x_normalized, y_normalized, theta0, theta1))
        # x_values = np.linspace(-2, 3, 100)
        x_values = np.linspace(-2, 3, 100)
        # print("-" * 50)
        # print("x_values:\n", x_values)
        y_pred = (theta1 * x_values) + theta0
        # print("-" * 50)
        # print("Y_pred:\n", y_pred)
        plt.plot(x_values, y_pred, color=(0,1,0,0.5), zorder=2)
        plt.draw()

    print("theta0:", theta0)
    print("theta1:", theta1)
    plt.plot(x_values, y_pred, color='red')

    # print("COST:\n", costs)
    # x_values = np.linspace(-2, 3, 100)
    # print("-" * 50)
    # print("x_values:\n", x_values)
    # y_pred = (theta1 * x_values) + theta0
    # print("-" * 50)
    # print("Y_pred:\n", y_pred)
    # plt.plot(x_values, y_pred, color='red', zorder=1)

    plt.scatter(x_normalized, price, color='b', zorder=3)
    plt.xlabel("mileage")
    plt.ylabel("price")
    plt.show()

def main():
    linear_regression(0.1, 100)

if __name__ == "__main__":
    main()