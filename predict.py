from train import train, visualise, load_thetas, JSON_FILE_PATH
import json

def load_thetas():
    try:
        with open(JSON_FILE_PATH, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        return None



def estimate_price(mileage):
    data = load_thetas()
    if data is None:
        return 0
    mileage_normalized = (mileage - data["mean"]) / data["std"]
    return data["theta0"] + (data["theta1"] * mileage_normalized)




def main():


    mileage = int(input("enter a mileage:"))
    estimated_price = estimate_price(mileage)
    print(f"the estimated price : {estimated_price}")
    data =  load_thetas()
    if data is None:
        return
    visualise(data, mileage)
    #caluclate theta1 and theta2


if __name__ == "__main__":
    main()
        