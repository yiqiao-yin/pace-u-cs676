# 02: Basics in Statistical Learning

## What is Statistical Machine Learning?

Statistical Machine Learning combines the fields of statistics and machine learning to extract meaningful patterns and predictions from data. It involves methods that use data samples to build models capable of inference, decision-making, and forecasting.

### Key Features:
- Focuses on understanding relationships between variables in data.
- Balances prediction accuracy with interpretability.
- Relies on probabilistic models and statistical tests to evaluate results.

In essence, it provides a theoretical foundation to support machine learning algorithms, enabling us to analyze data and derive robust conclusions.

---

## Motivation for Using Statistical Concepts in Machine Learning

Integrating statistical principles enhances machine learning by:

1. **Improving Understanding**:
   - Statistical tools help explain *why* a machine learning model behaves in a particular way.

2. **Model Validation**:
   - Statistical metrics ensure models generalize well to unseen data.

3. **Interpretability**:
   - Many statistical models, such as linear regression, provide interpretable relationships between input and output variables.

4. **Robustness**:
   - Statistical insights prevent overfitting and optimize model performance on diverse datasets.

### Benefits:
- Helps identify the right features and data distributions.
- Provides clarity and transparency in decision-making.

---

## Main Concepts and Definitions

Statistical learning revolves around several core ideas. Let’s explore these concepts in the context of a common supervised learning model—**linear regression**.

### 1. **Supervised Learning**:
   In supervised learning, we learn a mapping from inputs (features) to outputs (labels) using labeled data.

```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Example: Simple Linear Regression
import numpy as np
import pandas as pd

# Simulated dataset
np.random.seed(0)
x = np.random.rand(100, 1)  # Feature
y = 3 * x.squeeze() + 2 + np.random.randn(100) * 0.1  # Linear relationship with noise

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predictions
y_pred = model.predict(x_test)

# Performance
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")
```

### 2. **Training vs. Testing**:
   - **Training Data**: Used to fit the model.
   - **Testing Data**: Used to evaluate the model’s generalizability.

### 3. **Linear Regression**:
   - A fundamental statistical learning method.
   - Models the relationship between features \( X \) and target \( Y \) using a linear equation.
   - Example:
     - Predicting housing prices based on square footage and location.

---

## Examples of Statistical Learning Applications

1. **Finance**:
   - Predicting stock prices using historical data.
   - Example: Using time-series models like ARIMA.

2. **Healthcare**:
   - Diagnosing diseases based on patient symptoms and test results.
   - Example: Logistic regression for binary classification of disease presence.

3. **Marketing**:
   - Understanding customer behavior for targeted advertising.
   - Example: Clustering customers using statistical methods like K-Means.

4. **Manufacturing**:
   - Predicting equipment failure based on sensor data.
   - Example: Linear regression for trend analysis and anomaly detection.

---

Statistical learning provides the foundation to develop interpretable and reliable machine learning models. It bridges the gap between theory and practice, ensuring models are both predictive and meaningful.