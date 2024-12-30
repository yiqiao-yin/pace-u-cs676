# Lecture Notes: Model Selection and Related Concepts

Back to [home](../README.md)

## 1. What is Model Selection? Why is it Important?
Model selection refers to the process of choosing the best statistical model from a set of candidate models. It is crucial for:

- **Predictive Performance**: Ensures accurate predictions on unseen data.
- **Interpretability**: Identifies important predictors, making results actionable.
- **Overfitting Avoidance**: Balances complexity and performance to prevent overfitting or underfitting.
- **Resource Efficiency**: Reduces computational costs and improves scalability.
- **Generalization**: Ensures the model performs well on new, unseen data.

---

## 2. What is \( p \choose 2 \)?
When we have \( p \) features and want to choose 2, \( p \choose 2 \) represents the number of unique pairs of features that can be formed. For example, with 10 features, there are 45 possible pairs. This combinatorial complexity increases rapidly as \( p \) grows, impacting computational feasibility when analyzing interactions or selecting features in high-dimensional datasets.

---

## 3. Subset Selection and Error Measurement
### How to Measure Errors:
- **Residual Sum of Squares (RSS)**: Measures the total discrepancy between observed and predicted values. Lower RSS indicates a better fit.
- **Akaike Information Criterion (AIC)**: Balances goodness-of-fit with model complexity. Lower AIC values indicate better models.
- **Bayesian Information Criterion (BIC)**: Similar to AIC but imposes a stronger penalty for complexity, favoring simpler models.

### Key Differences:
- **AIC**: Optimizes prediction accuracy, tolerating slightly more complexity.
- **BIC**: More conservative, leading to simpler models.
- **RSS**: A raw error measure, unsuitable for comparing models with differing complexities.

---

## 4. What is Forward Stepwise Selection?
Forward stepwise selection starts with no predictors and adds the most significant predictor iteratively until no further improvement is observed.

### Advantages:
- Simplifies modeling for large datasets.
- Computationally efficient compared to testing all subsets.

### Disadvantages:
- May miss optimal predictor combinations.

### Python Example:
```python
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression

X, y = ... # Replace with data
model = LinearRegression()
forward_selector = SequentialFeatureSelector(model, direction='forward', n_features_to_select=5)
forward_selector.fit(X, y)
print("Selected features:", forward_selector.get_support())
```

---

## 5. What is Backward Stepwise Selection?
Backward stepwise selection starts with all predictors and removes the least significant one iteratively until further removal reduces performance.

### Advantages:
- Considers all predictors initially, reducing missed significant predictors.

### Disadvantages:
- Computationally intensive for large datasets.

### Python Example:
```python
backward_selector = SequentialFeatureSelector(model, direction='backward', n_features_to_select=5)
backward_selector.fit(X, y)
print("Selected features:", backward_selector.get_support())
```

---

## 6. Cp Statistic
**Cp** evaluates models by penalizing RSS with a term proportional to the number of predictors. It balances fit quality and model complexity, discouraging overfitting.

---

## 7. Adjusted \( R^2 \)
- **Definition**: Adjusted \( R^2 \) accounts for the number of predictors, increasing only if new predictors improve the model beyond chance levels.
- **Comparison with \( R^2 \)**:
  - \( R^2 \): Measures explained variance but always increases with added predictors.
  - Adjusted \( R^2 \): Penalizes unnecessary predictors, ensuring reliable model comparison.

---

## 8. Ridge and Lasso Regression
- **Ridge Regression**:
  - Adds an L2 penalty to coefficients, shrinking them but retaining all predictors.
  - Ideal for multicollinearity.

- **Lasso Regression**:
  - Adds an L1 penalty, encouraging sparsity by shrinking some coefficients to zero.
  - Suitable for feature selection.

### Python Example:
```python
from sklearn.linear_model import Ridge, Lasso

ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("Ridge coefficients:", ridge.coef_)

lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("Lasso coefficients:", lasso.coef_)
```

---

## 9. What is ElasticNet?
ElasticNet combines Ridge and Lasso penalties, balancing coefficient stability and sparsity. It is useful for correlated predictors.

### Python Example:
```python
from sklearn.linear_model import ElasticNet

elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net.fit(X, y)
print("ElasticNet coefficients:", elastic_net.coef_)
```

---

### Summary
Model selection and regularization techniques like Ridge, Lasso, and ElasticNet enable interpretable and predictive models. Forward and backward stepwise selection streamline feature selection, while metrics like AIC, BIC, Cp, and adjusted \( R^2 \) guide model evaluation effectively.

