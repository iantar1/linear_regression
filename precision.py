import pandas as pd
from predict import estimate_price


def main():
    data = pd.read_csv("data.csv")
    prices = data["price"]
    kms = data["km"]
    total_error = 0
    for i in range(len(prices)):
        error = abs((prices[i] - estimate_price(kms[i]))/prices[i])
        total_error += error
    precision = (total_error / len(prices)) * 100
    print(f"{round(precision, 2)} %")


if __name__ == "__main__":
    main()