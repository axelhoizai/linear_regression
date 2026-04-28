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
    normalized_data = [float((x - np.mean(array)) / np.std(array)) for x in array]
    return np.asarray(normalized_data)

def linear_regression(learning_rate, nb_iter):
    df = load(sys.argv[1])
    print(df)
    mileage = df["km"]
    price = df["price"]

    x_normalized = normalize(df["km"].astype(float))
    y_normalized = normalize(df["price"].astype(float))
    # print("-" * 50)
    # print("x_normalized:\n", x_normalized)
    # print("-" * 50)
    # print("y_normalized:\n", y_normalized)

    costs = []
    alpha = learning_rate / len(mileage)
    theta0, theta1 = 0, 0
    for i in range(nb_iter):
        tmp0 = []
        tmp1 = []
        for i in range(len(mileage)):
            tmp0.append(estimatePrice(x_normalized[i], theta0, theta1) - y_normalized[i])
            tmp1.append(tmp0[i] * x_normalized[i])
        tmp_theta0 = theta0 - alpha * np.sum(tmp0)
        tmp_theta1 = theta1 - alpha * np.sum(tmp1)
        theta0 = tmp_theta0
        theta1 = tmp_theta1
        # costs.append(cost(x_normalized, y_normalized, theta0, theta1))

    print("theta0:", theta0)
    print("theta1:", theta1)
    # print("COST:\n", costs)
    y_pred = (theta1 * mileage) + theta0
    print("Y_pred:\n", y_pred)


    plt.plot([min(mileage), max(mileage)], [min(y_pred), max(y_pred)], color='red')
    plt.scatter(mileage, price)
    # plt.xlabel("mileage")
    # plt.ylabel("price")
    plt.show()

def main():
    linear_regression(0.0001, 100)

if __name__ == "__main__":
    main()