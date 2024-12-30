# Supervised Learning Overview

Back to [home](../README.md)

## 📚 What is Supervised Learning?
Supervised Learning is a type of machine learning approach that involves training a model on a labeled dataset. In this context, the dataset consists of features (X) and a corresponding response (Y). The goal is to learn a mapping from the input features to the output response, so that the model can make predictions on new, unseen data.

- **Features (X)**: These are the input variables or predictors in the dataset.
- **Response (Y)**: This is the output variable or the target that the model is trying to predict.

## Linear Regression Model
A linear regression model is a fundamental approach in statistics used to model the relationship between a dependent variable (Y) and one or more independent variables (X). The formula for a simple linear regression model with one predictor is:

$$ Y = \beta_0 + \beta_1X + \epsilon $$

Where:
- $ Y $ is the response variable.
- $ X $ is the predictor variable.
- $ \beta_0 $ is the y-intercept.
- $ \beta_1 $ is the slope of the line.
- $ \epsilon $ is the error term.

## Least Square Loss Function
The least squares loss function is used to estimate the coefficients of the linear regression model. It calculates the sum of the squared differences between the observed responses in the dataset and the responses predicted by the linear approximation. The formula is:

$$ L(\beta) = \sum_{i=1}^{n} (y_i - ( \beta_0 + \beta_1x_i ))^2 $$

Where:
- $ L(\beta) $ is the loss function.
- $ y_i $ is the actual response for the ith observation.
- $ \beta_0 + \beta_1x_i $ is the predicted response for the ith observation.
- $ n $ is the number of observations.

## Main Assumptions of OLS Model
The Ordinary Least Squares (OLS) model relies on several key assumptions:
1. Linearity: The relationship between $X$ and $Y$ is linear.
2. Independence: The observations are independent of each other.
3. Homoscedasticity: The variance of the error terms is constant.
4. Normality: The error terms are normally distributed.

## Training the Model
There are two main approaches to training a linear regression model: likelihood maximization and gradient descent.

### Likelihood Maximization
This approach involves solving the model using the log-likelihood function. The log-likelihood is maximized to find the parameter estimates that make the observed data most probable. The formula for the log-likelihood in linear regression is:

$$ \log L(\beta) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^{n} (y_i - ( \beta_0 + \beta_1x_i ))^2 $$

### Gradient Descent
Gradient descent is an optimization algorithm used to minimize the loss function. It iteratively adjusts the parameters to find the best values that minimize the cost function. The update rule in gradient descent is:

$$ \beta := \beta - \alpha \frac{\partial}{\partial \beta} L(\beta) $$

Where $\alpha$ is the learning rate.

## Solve for the Best Solution based on the Least Square Error
To find the best solution for the coefficients $\beta_0$ and $\beta_1$ in a linear regression model based on Ordinary Least Squares (OLS) assumptions, we minimize the least squares loss function. The least squares loss function, $L(\beta)$, is given by:

$$ L(\beta) = \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1x_i))^2 $$

To minimize this function with respect to $\beta_0$ and $\beta_1$, we take the partial derivatives of $L(\beta)$ with respect to $\beta_0$ and $\beta_1$, and set them equal to zero.

### Partial Derivative with Respect to $\beta_0$

$$ \frac{\partial L(\beta)}{\partial \beta_0} = -2 \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1x_i)) = 0 $$

### Partial Derivative with Respect to $\beta_1$

$$ \frac{\partial L(\beta)}{\partial \beta_1} = -2 \sum_{i=1}^{n} (y_i - (\beta_0 + \beta_1x_i))x_i = 0 $$

Solving these equations simultaneously gives us the values of $\beta_0$ and $\beta_1$ that minimize the loss function. 

Let's denote:

- $ \bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i $ as the mean of $X$.
- $ \bar{y} = \frac{1}{n}\sum_{i=1}^{n}y_i $ as the mean of $Y$.

Then, solving the system of equations, we obtain the formulas for $\beta_1$ and $\beta_0$:

### Solution for $\beta_1$

$$ \beta_1 = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2} $$

### Solution for $\beta_0$

$$ \beta_0 = \bar{y} - \beta_1\bar{x} $$

These equations give us the best linear unbiased estimators for $\beta_0$ and $\beta_1$ under the OLS assumptions. The slope $\beta_1$ tells us how much the response variable $Y$ changes for a one-unit change in the predictor variable $X$, and the intercept $\beta_0$ gives the value of $Y$ when $X$ is zero.

## Example Using sklearn and Numpy

Here is a simple Python code snippet using `sklearn` and `numpy` to illustrate the linear regression model:

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generating synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Training the model
model = LinearRegression()
model.fit(X, y)

# Making predictions
X_new = np.array([[0], [2]])
y_predict = model.predict(X_new)

# Plotting
plt.scatter(X, y)
plt.plot(X_new, y_predict, color='red')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Example')
plt.show()
```

This code generates a simple linear dataset and fits a linear regression model using `sklearn`. It also visualizes the fitted line. 

### Use Tensorflow Library

To replicate a linear regression model similar to `sklearn`'s `LinearRegression` using TensorFlow, you can create a simple neural network with no hidden layers and a single unit in the output layer. This setup mimics the behavior of a linear regression model. Below is a step-by-step guide and the corresponding code to achieve this:

#### Step 1: Import TensorFlow
First, import the TensorFlow library. If you haven't already installed TensorFlow, you can do so using `pip install tensorflow`.

```python
import tensorflow as tf
```

#### Step 2: Define the Model
Define a neural network model using TensorFlow's `Sequential` API. Since it's a linear regression, the model will have no hidden layers and just one unit in the output layer.

```python
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],))
])
```

Here, `Dense(1)` creates a single unit (neuron) which is equivalent to the output of linear regression. `input_shape` is set to match the number of features in your input dataset `X`.

#### Step 3: Compile the Model
Compile the model with Stochastic Gradient Descent (SGD) as the optimizer and Mean Squared Error (MSE) as the loss function. These choices are typical for a linear regression model.

```python
model.compile(optimizer='sgd', loss='mean_squared_error')
```

#### Step 4: Train the Model
Train the model with your features `X` and labels `y`. You will need to specify the number of epochs, which determines how many times the model will iterate over the entire dataset.

```python
model.fit(X, y, epochs=number_of_epochs)
```

Replace `number_of_epochs` with the desired number of iterations (e.g., 100, 200, etc.)

#### Full Example Code

```python
import tensorflow as tf

# Assuming X and y are defined and preprocessed

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X.shape[1],))
])

# Compile the model
model.compile(optimizer='sgd', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100)  # Replace 100 with your chosen number of epochs
```

This code will create and train a TensorFlow model that performs linear regression, analogous to using `LinearRegression` from `sklearn`. Ensure that your input data `X` and labels `y` are correctly preprocessed and available for training.

# Tensorflow Implementation on Colab

```python
# import
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# get data
train_path = '/content/sample_data/california_housing_train.csv'
test_path = '/content/sample_data/california_housing_test.csv'
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# define X and y
X_train = train.iloc[:, train.columns!='median_house_value']
y_train = train['median_house_value']
X_test = test.iloc[:, test.columns!='median_house_value']
y_test = test['median_house_value']

# import
import tensorflow as tf

# define a neural network model
model = tf.keras.models.Sequential(name='this_model')
model.add(tf.keras.layers.Dense(1, input_shape=[8]))
model.summary()

# compile
model.compile(optimizer='rmsprop', loss='mae')

# fit
model.fit(X_train, y_train, validation_split=0.2, epochs=10)

# save
model.save('tmp_model_yin.h5')

# predict on test set
y_test_pred_ = model.predict(X_test)

# how accurate?
# step 1: take the difference
y_test_pred_.reshape((-1)) - np.asarray(y_test)

# step 2: take the absolute value of the difference
np.abs(y_test_pred_.reshape((-1)) - np.asarray(y_test))

# sttep 3: take the average of the absolute difference
np.mean(np.abs(y_test_pred_.reshape((-1)) - np.asarray(y_test)))
```