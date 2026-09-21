# 🚗 Linear Regression: predict a car's price from its mileage

A tiny machine-learning project, built from scratch (no `sklearn`).
It learns a straight line that links **mileage** to **price**, then uses that line to guess the price of any car.

```
price ≈ θ₀ + θ₁ × mileage
```

---

## 🧠 The math, step by step

### 1. The model: a straight line

We assume the price depends linearly on the mileage:

$$\text{estimatePrice}(km) = \theta_0 + \theta_1 \cdot km$$

- **θ₀** (theta zero) is the *intercept*: the price of a car with 0 km.
- **θ₁** (theta one) is the *slope*: how much the price changes for each extra km (it will be negative: more km, cheaper car).

Training means finding the best values of θ₀ and θ₁.

### 2. How wrong are we? The error

For one car *i*, the error is the gap between our guess and the real price:

$$\text{error}_i = (\theta_0 + \theta_1 \cdot km_i) - price_i$$

To score the *whole* line we use the **Mean Squared Error** (squared so that positive and negative errors don't cancel out):

$$J(\theta_0, \theta_1) = \frac{1}{2m} \sum_{i=1}^{m} \big(\theta_0 + \theta_1 \cdot km_i - price_i\big)^2$$

Small `J` = good line. Our goal is to make `J` as small as possible.

### 3. Gradient descent: walking downhill

Imagine `J` as a valley. We start somewhere (θ₀ = 0, θ₁ = 0) and take small steps **downhill** until we reach the bottom.
The slope of the valley at our position is the **gradient** (the partial derivatives of `J`):

$$\frac{\partial J}{\partial \theta_0} = \frac{1}{m}\sum_{i=1}^{m} \text{error}_i
\qquad
\frac{\partial J}{\partial \theta_1} = \frac{1}{m}\sum_{i=1}^{m} \text{error}_i \cdot km_i$$

Then, on each iteration, we update **both thetas at the same time**:

$$\theta_0 \leftarrow \theta_0 - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m} \text{error}_i$$

$$\theta_1 \leftarrow \theta_1 - \alpha \cdot \frac{1}{m}\sum_{i=1}^{m} \text{error}_i \cdot km_i$$

- **α** (alpha) is the *learning rate*: the size of each step. Here `α = 0.01`.
  Too big → we overshoot and bounce around. Too small → it takes forever.
- We repeat this `1000` times.

> 💡 "At the same time" matters: the code sums up `tmp_theta0` and `tmp_theta1` first, and only then updates both, so that θ₁'s step doesn't use an already-changed θ₀.

### 4. Normalization: making the valley round

Mileage is huge (up to ~250 000) while the intercept is tiny in comparison. With raw values the valley is a super narrow canyon and gradient descent crawls or explodes.
So we **standardize** the mileage before training (z-score):

$$km_{norm} = \frac{km - \mu}{\sigma}$$

where **μ** is the mean of the mileages and **σ** their standard deviation.
The values now sit around 0, mostly between −2 and 2, and gradient descent converges quickly.

We save `μ` and `σ` next to the thetas, because **any mileage we predict later must be normalized with the same μ and σ**.

### 5. Prediction

To estimate the price of a car with mileage `km`:

1. Normalize it: `km_norm = (km − μ) / σ`
2. Apply the line: `price = θ₀ + θ₁ · km_norm`

If you prefer the line in real (un-normalized) units, it is equivalent to:

$$\theta_1^{real} = \frac{\theta_1}{\sigma}
\qquad
\theta_0^{real} = \theta_0 - \theta_1 \cdot \frac{\mu}{\sigma}$$

### 6. How good is the line?

`precision.py` computes the **mean absolute percentage error** over the dataset:

$$\text{error \%} = \frac{100}{m} \sum_{i=1}^{m} \left| \frac{price_i - \text{estimatePrice}(km_i)}{price_i} \right|$$

On average, how many percent are we off from the real price? **Lower is better.**

---

## 📁 Files

| File | What it does |
|------|--------------|
| `data.csv` | The dataset: `km,price` for 24 cars |
| `train.py` | Normalizes the data, runs gradient descent, saves the result in `thetas.json` |
| `predict.py` | Asks for a mileage, prints the estimated price and plots it as a red dot on the line |
| `precision.py` | Prints the average error of the model on the dataset |
| `thetas.json` | Trained parameters: `theta0`, `theta1`, `mean`, `std` |

## ▶️ How to run

```bash
# 1. install dependencies (creates a .venv)
bash install_depen.sh
source .venv/bin/activate

# 2. train the model  ->  writes thetas.json
python train.py

# 3. ask for a price
python predict.py

# 4. measure the error
python precision.py
```

> Without a trained `thetas.json`, `predict.py` returns `0`, so run `train.py` first.

## 📊 Result

With the current `thetas.json` (normalized units):

```
θ₀ ≈ 6331.56      θ₁ ≈ -1105.97      μ ≈ 101066      σ ≈ 51565
```

In real units that is roughly **price ≈ 8499 − 0.0214 × km**: a fresh car starts around 8 500 and loses about **2.1 cents per km** driven.
