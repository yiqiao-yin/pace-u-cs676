# Lecture Notes: Sampling, Bootstrap, and Cross Validation

## What is Sampling?

**Sampling** refers to the process of selecting a subset of data from a larger dataset. The goal is to make inferences about the entire dataset using the selected subset.

### Types of Sampling:
1. **Random Sampling**: Each data point has an equal chance of being selected.
2. **Stratified Sampling**: Ensures proportional representation of subgroups within the dataset.
3. **Systematic Sampling**: Selects data points at regular intervals from the dataset.

### Why Sampling?
- Reduces computational cost.
- Enables analysis when the full dataset is unavailable.
- Helps in testing hypotheses without processing all data.

---

## What is Bootstrap?

**Bootstrap** is a statistical technique that involves:
1. Randomly sampling from the data with replacement.
2. Generating multiple datasets (called bootstrap samples).
3. Calculating statistics (e.g., mean, variance) for each sample to estimate uncertainty.

### Why Use Bootstrap?
- It estimates the variability of a statistic without requiring additional data.
- Useful for small datasets where traditional sampling may fail.

**Python Example:**
```python
import numpy as np

# Original dataset
data = [5, 10, 15, 20, 25]

# Generate bootstrap samples
bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
print("Bootstrap Sample:", bootstrap_sample)
```

---

## Why are Sampling and Bootstrap Helpful for Machine Learning?

### Motivation:
1. **Model Generalization**: Understand how well a model performs on unseen data.
2. **Variance Estimation**: Assess the stability of model predictions.
3. **Robustness**: Improve model reliability by accounting for data variability.

Sampling and bootstrap allow us to:
- Train models on subsets to prevent overfitting.
- Measure model performance under different scenarios.

---

## What is Cross Validation?

**Cross Validation (CV)** is a method to evaluate the performance of a model by splitting the data into training and validation sets multiple times.

### Why Do We Need Cross Validation?
- Prevents overfitting.
- Provides a more reliable estimate of model performance.
- Helps in hyperparameter tuning.

### Common Cross Validation Techniques:
1. **K-Fold Cross Validation**:
   - Split the dataset into `k` subsets (folds).
   - Train the model on `k-1` folds and validate on the remaining fold.
   - Repeat for all folds and average the results.

2. **Leave-One-Out Cross Validation (LOOCV)**:
   - Use one sample as the validation set and the rest as training data.

3. **Stratified K-Fold**: Ensures proportional representation of classes in each fold.

---

### How to Perform Cross Validation Using Packages:

#### Using Scikit-Learn:
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate a synthetic dataset
X, y = make_regression(n_samples=100, n_features=1, noise=0.1)

# Initialize model
model = LinearRegression()

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=kf)
print("Cross Validation Scores:", scores)
print("Mean Score:", scores.mean())
```

#### Using TensorFlow:
```python
import tensorflow as tf
from sklearn.model_selection import KFold
import numpy as np

# Generate a synthetic dataset
X = np.random.rand(100, 1)
y = 3 * X[:, 0] + np.random.randn(100) * 0.1

# Define a simple model
def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1, input_shape=(1,))
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# Perform K-Fold Cross Validation
kf = KFold(n_splits=5)
scores = []

for train_index, val_index in kf.split(X):
    model = build_model()
    model.fit(X[train_index], y[train_index], epochs=10, verbose=0)
    score = model.evaluate(X[val_index], y[val_index], verbose=0)
    scores.append(score)

print("Cross Validation Losses:", scores)
print("Mean Loss:", np.mean(scores))
```

---

### Cross Validation From Scratch:
```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 1)
y = 3 * X[:, 0] + np.random.randn(100) * 0.1

# Number of folds
k = 5
fold_size = len(X) // k

mse_scores = []

for i in range(k):
    # Split data into training and validation sets
    val_start = i * fold_size
    val_end = val_start + fold_size

    X_val = X[val_start:val_end]
    y_val = y[val_start:val_end]

    X_train = np.concatenate([X[:val_start], X[val_end:]], axis=0)
    y_train = np.concatenate([y[:val_start], y[val_end:]], axis=0)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Validate model
    y_pred = model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mse_scores.append(mse)

print("Cross Validation MSEs:", mse_scores)
print("Mean MSE:", np.mean(mse_scores))
```

---

## Measuring Variance and Error in Cross Validation

### Variance:
Variance reflects how much the model’s predictions vary across different validation sets. This can be calculated as the standard deviation of the scores.

**Python Example:**
```python
import numpy as np

scores = [0.85, 0.88, 0.84, 0.87, 0.86]
variance = np.var(scores)
print("Variance:", variance)
```

### Error:
Error represents the average difference between the predicted values and the actual values.

**Python Example:**
```python
from sklearn.metrics import mean_squared_error

y_true = [1.5, 2.0, 1.8]
y_pred = [1.4, 2.1, 1.7]
error = mean_squared_error(y_true, y_pred)
print("Mean Squared Error:", error)
```

