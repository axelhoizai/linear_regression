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
    # Get data from file
    df = load(sys.argv[1])
    print(df)
    mileage = np.array(df["km"]).astype(float)
    price = np.array(df["price"]).astype(float)

    # Normalize mileage
    x_normalized = normalize(mileage)

    #Init graphs
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    loss = axs[0]
    linear = axs[1]

    costs = []
    alpha = learning_rate / len(mileage)
    theta0, theta1 = 0, 0
    for i in range(nb_iter):
        tmp0 = []
        tmp1 = []
        # Apply gradient descent
        for j in range(len(mileage)):
            tmp0.append(estimatePrice(x_normalized[j], theta0, theta1) - price[j])
            tmp1.append(tmp0[j] * x_normalized[j])
        nptmp0 = np.array(tmp0)
        nptmp1 = np.array(tmp1)
        tmp_theta0 = theta0 - alpha * np.sum(nptmp0)
        tmp_theta1 = theta1 - alpha * np.sum(nptmp1)
        theta0 = tmp_theta0
        theta1 = tmp_theta1

        # Apply loss function
        costs.append(cost(x_normalized, price, theta0, theta1))

        # Display each adjusted affine curves
        x_values = np.linspace(-2, 3, 100)
        y_pred = (theta1 * x_values) + theta0
        linear.plot(x_values, y_pred, color=(0,1,0,0.5), zorder=2)
        plt.draw()

    # Display loss for each iteration
    loss.set_title("Loss Function")
    loss.set_ylabel("Loss")
    loss.set_xlabel("Nb of iterations")
    loss.plot(np.arange(nb_iter), costs, color='b')

    # Display final affine line and data
    linear.set_title("Linear regression")
    linear.scatter(x_normalized, price, color='b', zorder=3)
    linear.plot(x_values, y_pred, color='red')
    linear.set_xlabel("Normalized mileage")
    linear.set_ylabel("Price")
    plt.show()

def main():
    linear_regression(0.1, 100)

if __name__ == "__main__":
    main()