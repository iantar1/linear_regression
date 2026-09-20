from train import *

def estimate_price(mileage, theta0 = 0, theta1 = 0):
    mean, std = getMeanStd()
    mileage_normalized = (mileage - mean) / std
    return theta0 + (theta1 * mileage_normalized)



def main():
    mileage = int(input("enter a mileage:"))
    price_before_training = estimate_price(mileage)
    print(f"the estimated price before taining is : {price_before_training}")
    theta0, theta1 = train()
    price_after_training = estimate_price(mileage, theta0, theta1)
    print(f"the estimated price after taining is : {price_after_training}")

    #caluclate theta1 and theta2


if __name__ == "__main__":
    main()
        