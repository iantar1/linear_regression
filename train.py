import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

JSON_FILE_PATH = "thetas.json"

def visualise(thetas, mileage_to_predict = None):
    data = pd.read_csv("data.csv")

    x = data["km"]
    y = data["price"]
    std = np.std(x)
    x = normalize(x)
    predicted_y = thetas["theta0"] + thetas["theta1"] * x

    plt.scatter(x, y)
    plt.plot(x, predicted_y)
    if mileage_to_predict is not None:
        # Normalize the given mileage
        normalized_mileage = (mileage_to_predict - thetas["mean"]) / thetas["std"]

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
    with open(JSON_FILE_PATH, "w") as file:
        json.dump(data, file)

def normalize(x):
    return (x - np.mean(x)) / np.std(x)

def gradient_descent(km, price, learning_rate=0.01, iterations=1000):
    # model : estimatePrice = theta0 + theta1 * mileage
    theta0 = 0
    theta1 = 0
    n_samples = len(km)

    for _ in range(iterations):
        tmp_theta0 = 0
        tmp_theta1 = 0
        for i in range(n_samples):
            estimate_price = theta0 + theta1 * km[i]
            error = estimate_price - price[i] # price is the real price
            tmp_theta0 += learning_rate * error * (1/n_samples)
            tmp_theta1 += learning_rate * error * km[i] * (1/n_samples)
        theta0 = theta0 - tmp_theta0
        theta1 = theta1 - tmp_theta1
    return theta0, theta1

def train():
    data = pd.read_csv("data.csv")

    data = data.sort_values("km")

    km = data["km"]
    price = data["price"]

    # you have to find theta0 and theta1

    mean = np.mean(km)
    std = np.std(km)

    theta0, theta1 = gradient_descent(normalize(km), price)
    store_result(theta0, theta1, mean, std)

if __name__ == "__main__":
    train()

