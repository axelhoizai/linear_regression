from load import load
from estimatePrice import estimatePrice
import matplotlib.pyplot as plt
import numpy as np
import sys


def ft_loss(x, y, theta0, theta1):
    """Measure the performance of the model
    
    Parameters : x: ndarray, y: ndarray, theta0: float, theta1: float
    
    return : loss func result: float"""
    m = len(y)
    return np.sum((y - ((theta1 * x) + theta0)) ** 2) / m

def normalize(array):
    """Normalize data
    
    Parameters : array: ndarray
    
    Return : normalized data array: ndarray"""
    normalized_data = (array - np.mean(array)) / np.std(array)
    return np.asarray(normalized_data)

def gradient_descent(mileage, price, theta0, theta1, learning_rate):
    """Apply gradient descent to learn from error until a model convergence
    
    Parameters: mileage: ndarray, price: ndarray, theta0: float, theta1: float, learning_rate: float
    
    Return : theta0: float, theta1: float"""
    alpha = learning_rate / len(mileage)
    x_normalized = normalize(mileage)

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

    return theta0, theta1

def linear_regression(mileage, price, learning_rate, nb_iter):
    """Apply linear regression using a gradient descent algorithm
    
    Parameters : mileage: ndarray, price: ndarray, learning_rate: float, nb_iter: int
    
    Return : Finals thetas -> theta0: float, theta1: float"""
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
        # Apply gradient descent
        theta0, theta1 = gradient_descent(mileage, price, theta0, theta1, learning_rate)

        # Apply loss function
        costs.append(ft_loss(x_normalized, price, theta0, theta1))

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

    return theta0, theta1

def main():
    # Get data from file
    df = load(sys.argv[1])
    print(df)
    mileage = np.array(df["km"]).astype(float)
    price = np.array(df["price"]).astype(float)

    theta0, theta1 = linear_regression(mileage, price, 0.1, 100)

    # De-normalized thetas
    theta0 = theta0 - (theta1 * np.mean(mileage / np.std(mileage)))
    theta1 = theta1 / np.std(mileage)

    with open("thetas.txt", 'w') as file:
        file.write(f"theta0:{theta0}\ntheta1:{theta1}")


if __name__ == "__main__":
    main()
