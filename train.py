import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 


def denormalize(t0_scaled, t1_scaled, mean, std):
    theta1 = t1_scaled / std
    theta0 = t0_scaled - (t1_scaled * mean / std)
    return theta0, theta1


def train():
    data = pd.read_csv("data.csv")

    data = data.sort_values("km")

    x = data["km"]
    y = data["price"]




    # model : estimatePrice  = theta0 ​ + theta0 ​* mileage 


    # you have to find theta0 and theta1



    plt.xlabel("Kilometers")
    plt.ylabel("Price")
    plt.title("cars")


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
    theta0, theta1 = denormalize(theta0, theta1, mean, std)
    return theta0, theta1

def normalise():
    pass

def getMeanStd():
    data = pd.read_csv("data.csv")

    data = data.sort_values("km")

    x = data["km"]
    mean = np.mean(x)
    std = np.std(x)
    return mean, std

def visialse(theta0, theta1):
    data = pd.read_csv("data.csv")

    data = data.sort_values("km")

    x = data["km"]
    predicted_y = theta0 + theta1 * x

    xx = (61789 - mean) / std
    print(theta0 + theta1 * xx)
    plt.scatter(x, y)
    # print(theta0)
    # print(theta1)
    # print(predicted_y)
    plt.plot(x, predicted_y)


    plt.show()