import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


def visualise(thetas, mileage_to_predect = None):
    data = pd.read_csv("data.csv")

    data = data.sort_values("km")

    x = data["km"]
    y = data["price"]
    std = np.std(x)
    x = (x - np.mean(x)) / np.std(x)
    predicted_y = thetas["theta0"] + thetas["theta1"] * x

    plt.scatter(x, y)
    plt.plot(x, predicted_y)
    if mileage_to_predect is not None:
        # Normalize the given mileage
        normalized_mileage = (mileage_to_predect - thetas["mean"]) / thetas["std"]

        # Predict its price
        predicted_price = thetas["theta0"] + thetas["theta1"] * normalized_mileage

        # Draw the point in red
        plt.scatter(
            normalized_mileage,
            predicted_price,
            color="red"
        )
    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.title("cars")
    plt.show()

def store_result(theta0, theta1, mean, std):
    data = {
        "theta0": theta0,
        "theta1": theta1,
        "mean" : mean,
        "std" : std
    }
    with open("thetas.json", "w") as file:
        json.dump(data, file)

def train(): #gradient_descent
    data = pd.read_csv("data.csv")

    data = data.sort_values("km")

    x = data["km"]
    y = data["price"]

    # model : estimatePrice  = theta0 ​ + theta0 ​* mileage 

    # you have to find theta0 and theta1

    


    theta0 = 0
    theta1 = 0
    learningRate = 0.01 # controls how strongly you modify theta0 and theta1 at each iteration

    m = len(x)
    iteration = 1000
    mean = np.mean(x)
    std = np.std(x)
    x = (x - np.mean(x)) / np.std(x)

    for iter in range(iteration):
        tmp_theta0 = 0
        tmp_theta1 = 0
        for i in range(len(x)):
            estimate_price = theta0 + theta1 * x[i]
            error = estimate_price - y[i] # y real price
            tmp_theta0 += learningRate * error * (1/m)
            tmp_theta1 += learningRate * error * x[i] * (1/m)
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
    store_result(theta0, theta1, mean, std)

if __name__ == "__main__":
    train()

