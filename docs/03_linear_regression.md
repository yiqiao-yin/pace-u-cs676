# Linear Regression

Linear regression is one of the most fundamental and widely used techniques in machine learning and statistics. It's a method to model the relationship between a dependent variable and one or more independent variables. The goal is to fit a linear equation to the observed data so that we can predict new, unseen data.

## Why Linear Regression?

Linear regression is often the first model tried when performing regression tasks. Here are some reasons why it’s popular:

1. **Simplicity**: Linear regression is easy to understand, implement, and interpret. It provides a straightforward way to predict outcomes based on a linear relationship.
  
2. **Efficiency**: It’s computationally efficient compared to more complex models, especially when dealing with large datasets.

3. **Interpretability**: The coefficients (betas) in the linear regression model offer direct insight into the importance of each feature in predicting the target.

4. **Benchmarking**: Linear regression serves as a good baseline model. If more sophisticated models don’t outperform it significantly, there might be no need to use more complex methods.

## Assumptions for Linear Regression

Linear regression relies on several key assumptions for the model to be valid. These are:

1. **Linearity**: The relationship between the dependent variable and the independent variables is linear. That is, the change in the dependent variable is proportional to the change in the independent variable(s).
   
2. **Independence**: The residuals (errors) are independent. In simpler terms, the error for one observation should not influence the error of another observation.

3. **Homoscedasticity**: The residuals should have constant variance at all levels of the independent variable(s). This means the spread of residuals should not change as the value of the independent variables increases.

4. **Normality of Errors**: The residuals should be normally distributed for the purposes of inference (e.g., hypothesis testing).

5. **No multicollinearity**: The independent variables should not be highly correlated with each other. If they are, the coefficients become unstable and unreliable.

## The Linear Regression Formula

The linear regression model is represented by the following equation:

$$
y = \beta_0 + \beta_1x + \epsilon
$$

Where:
- $y$ is the dependent variable (output),
- $x$ is the independent variable (input),
- $\beta_0$ is the intercept (the value of $y$ when $x = 0$),
- $\beta_1$ is the slope (the change in $y$ for a one-unit change in $x$),
- $\epsilon$ is the error term (residuals, which capture the difference between the observed and predicted values).

For multiple independent variables, the equation extends to:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \dots + \beta_nx_n + \epsilon
$$

Where $x_1, x_2, \dots, x_n$ are the independent variables.

## Closed Form Solution for Beta Coefficients

The closed-form solution for the coefficients of linear regression can be derived using the least squares method. This method minimizes the sum of squared residuals. The objective is to find the coefficients ($\beta_0, \beta_1$) that minimize the following loss function:

$$
L(\beta_0, \beta_1) = \sum_{i=1}^m (y_i - (\beta_0 + \beta_1 x_i))^2
$$

Where $m$ is the number of data points, $y_i$ is the observed value, and $x_i$ is the corresponding independent variable.

Taking the derivative of the loss function with respect to $\beta_0$ and $\beta_1$, we find the following closed-form expressions for $\beta_0$ and $\beta_1$:

### Beta_1 (Slope):

$$
\beta_1 = \frac{\sum_{i=1}^m (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^m (x_i - \bar{x})^2}
$$

### Beta_0 (Intercept):

$$
\beta_0 = \bar{y} - \beta_1 \bar{x}
$$

Where $\bar{x}$ and $\bar{y}$ represent the mean values of the independent and dependent variables, respectively.

## How Do We Fit the Model?

Fitting a linear regression model involves finding the best-fitting line (or hyperplane in higher dimensions) to the data. There are several ways to fit a model:

### 1. **Closed-form Solution (Normal Equation)**

We can directly compute the coefficients using the closed-form solution derived earlier:

$$
\beta = (X^T X)^{-1} X^T y
$$

Where:
- $X$ is the matrix of input features (with a column of ones for the intercept term),
- $y$ is the vector of target values.

### 2. **Dynamic Programming Approach**

In dynamic programming, we can break down the problem into smaller subproblems. In this context, the dynamic programming approach may involve solving the least squares problem iteratively, though it’s not commonly applied to linear regression. This approach can be useful when we have more complex models, such as neural networks, or when solving problems that involve decision-making processes.

### 3. **Gradient Descent**

Gradient descent is an iterative optimization algorithm used when the dataset is large or the closed-form solution is computationally expensive. We update the coefficients iteratively to minimize the loss function.

The update rule for gradient descent is as follows:

$$
\beta_j = \beta_j - \alpha \frac{1}{m} \sum_{i=1}^m (h_\beta(x_i) - y_i) x_{ij}
$$

Where:
- $\alpha$ is the learning rate,
- $h_\beta(x_i)$ is the predicted value $\beta_0 + \beta_1 x_i$,
- $x_{ij}$ is the feature for the $j$-th coefficient.

## Implementing Linear Regression

### Using Scikit-Learn

Scikit-learn provides a simple and efficient implementation of linear regression. Here’s how we can use it:

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# Toy data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([1, 2, 3, 4, 5])            # Dependent variable

# Create a linear regression model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Make predictions
predictions = model.predict(X)

print(f'Intercept: {model.intercept_}')
print(f'Coefficients: {model.coef_}')
```

### Coding it from Scratch

Now let’s implement linear regression from scratch using toy data:

```python
import numpy as np

# Toy data
X = np.array([[1], [2], [3], [4], [5]])  # Independent variable
y = np.array([1, 2, 3, 4, 5])            # Dependent variable

# Adding bias (intercept) term
X_bias = np.c_[np.ones(X.shape[0]), X]

# Calculate coefficients using the closed-form solution
beta = np.linalg.inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)

print(f'Intercept: {beta[0]}')
print(f'Coefficient: {beta[1]}')
```

## Pros and Cons of Linear Regression

### Pros:
- **Simplicity**: Linear regression is easy to implement and understand.
- **Interpretability**: The model is easy to interpret because the coefficients represent the effect of each feature on the target variable.
- **Efficiency**: It’s computationally inexpensive, making it a good choice for large datasets.

### Cons:
- **Assumptions**: Linear regression relies on assumptions that may not always hold in real-world data (e.g., linearity, homoscedasticity).
- **Outliers**: Linear regression is sensitive to outliers, which can heavily influence the model.
- **Limited to linear relationships**: It cannot capture complex, non-linear relationships between variables unless you transform the features.

In summary, linear regression is a powerful and easy-to-use tool for regression tasks, provided that the assumptions hold. However, it's important to carefully evaluate its limitations and assumptions in real-world scenarios.