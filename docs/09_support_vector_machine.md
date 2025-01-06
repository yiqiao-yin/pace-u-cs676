# Support Vector Machine

Back to [home](../README.md)

## Introduction to SVM

![graph](../pics/09_svm_01.gif)

Support Vector Machine (SVM) is a powerful supervised machine learning algorithm primarily used for classification tasks but can also be adapted for regression. The main objective of SVM is to find the optimal hyperplane that best separates data points of different classes in the feature space. This hyperplane, which maximizes the margin between two classes, helps the model make predictions on new or unseen data points.

SVMs are particularly effective in high-dimensional spaces and situations where the number of dimensions exceeds the number of data samples. They are also versatile, as they can implement various kernel functions to handle non-linear classification tasks effectively.

### Key Concepts:

- **Hyperplane**: In an n-dimensional space, a hyperplane is a flat affine subspace with (n-1) dimensions that divides the space into two parts.
  
- **Margin**: The distance between the hyperplane and the nearest data point from either class. SVM aims to maximize this margin.
  
- **Support Vectors**: Data points that lie closest to the decision boundary or hyperplane. These points are crucial as they define the position of the hyperplane.

## SVM for Classification

In classification tasks, SVM works by transforming input data into a higher-dimensional space where it becomes easier to separate classes linearly using a hyperplane. When dealing with data that isn't linearly separable, SVM employs kernel tricks to implicitly map inputs into high-dimensional feature spaces.

### Process:

1. **Select the Kernel Function**: Choose a kernel like linear, polynomial, radial basis function (RBF), or sigmoid, depending on the nature of your data.
   
2. **Training**: During training, SVM identifies support vectors and calculates the optimal hyperplane that maximizes the margin.
   
3. **Prediction**: For a given test point, the SVM determines its position relative to the hyperplane and predicts its class based on which side of the hyperplane it falls.

### Example Code:

```python
from sklearn import svm
from sklearn.datasets import make_classification

# Generate synthetic binary classification dataset
X, y = make_classification(n_samples=100, n_features=2, random_state=42)

# Create an SVM classifier with a linear kernel
classifier = svm.SVC(kernel='linear')

# Train the model
classifier.fit(X, y)

# Predict a new sample
sample = [[0.5, 0.5]]
prediction = classifier.predict(sample)
print("Predicted class:", prediction)
```

## SVM for Regression

Although initially designed for classification, SVM can be adapted for regression tasks using Support Vector Regression (SVR). Unlike traditional regression methods, SVR seeks to fit the error within a certain threshold, known as epsilon.

### Characteristics of SVR:

- SVR attempts to find a function that deviates from the actual observed outputs by a value no greater than ε (epsilon), while simultaneously being as flat as possible.

- The capacity of the model is controlled by parameters like `C` (penalty parameter) and `epsilon`, which determine the trade-off between the flatness of the regression line and the amount tolerated by deviations.

### Example Code:

```python
from sklearn.svm import SVR
import numpy as np

# Generate synthetic regression dataset
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])

# Fit regression model
regressor = SVR(kernel='rbf', C=1e3, gamma=0.1)
regressor.fit(X, y)

# Make prediction
sample = [[1.5]]
predicted_value = regressor.predict(sample)
print("Predicted value:", predicted_value)
```
## SVM from Scratch

write 400-600 word .md file regards the following:

## SVM from Scratch

Support Vector Machines (SVM) are a powerful set of supervised learning algorithms primarily used for classification tasks. SVMs aim to find the hyperplane that best separates data into different classes in an n-dimensional space. Creating an SVM from scratch is an excellent way to understand how the algorithm works under the hood. Below, we'll explore how to implement a linear SVM classifier without using third-party packages like scikit-learn.

### Understanding the Problem

Given:
- A matrix $X$ with shape $m \times n$, where $m$ represents the number of samples and $n$ represents the number of features.
- A target vector $Y$ with length $m$, containing labels (typically -1 or 1).

The goal is to construct a linear SVM model that finds the optimal hyperplane separating the classes in the feature space.

## Coding a Simple SVM

To begin, we need to define our objective: maximize the margin between data points of different classes. This involves minimizing:

$$\frac{1}{2} ||w||^2$$

subject to the constraints:

$$y_i(w \cdot x_i + b) \geq 1$$

Where:
- $w$ is the weight vector orthogonal to the deciding hyperplane.
- $b$ is the bias term.
- $x_i$ is a feature vector.

### Implementation Steps

1. **Initialize Parameters**: Start with a random weight vector $w$ and bias $b$.
2. **Iterate through Samples**: Use gradient descent to update $w$ and $b$.
3. **Hinge Loss Function**: Compute the hinge loss for error calculation and update parameters accordingly.

Here's how you can implement a simple linear SVM:

```python
import numpy as np

class SimpleSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
    
    def fit(self, X, y):
        m, n = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n)
        self.b = 0

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    dw = 2 * self.lambda_param * self.w
                    db = 0
                else:
                    dw = 2 * self.lambda_param * self.w - np.dot(x_i, y_[idx])
                    db = y_[idx]
                
                self.w -= self.learning_rate * dw
                self.b -= self.learning_rate * db
    
    def predict(self, X):
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# Example usage:
# Define a dataset
X = np.array([[1, 2], [2, 3], [3, 3]])
Y = np.array([-1, -1, 1])

# Create and train SVM
svm = SimpleSVM()
svm.fit(X, Y)

# Predict output
predictions = svm.predict(X)
print("Predicted labels:", predictions)
```

### Key Points

- **Weight Update Rule**: The update for $w$ and $b$ depends on whether the sample satisfies the margin constraint. Gradient descent helps tweak these parameters iteratively.
- **Regularization Term ($\lambda$)**: Helps prevent overfitting by penalizing large weights.
- **Learning Rate**: Determines the size of steps taken towards the minimum loss.

This simplified version of SVM implements a basic linear classifier capable of partitioning linearly separable data. While it lacks optimizations and comprehensive feature support found in library-based SVM implementations, writing it from scratch provides valuable insight into the mechanics of SVMs. For more complex datasets, kernel tricks and support for non-linear decision boundaries would be necessary.

## Conclusion

Support Vector Machines are a versatile tool for both classification and regression. Their strength lies in their ability to handle high-dimensional data and separate classes using non-linear boundaries through kernel functions. While they can be computationally expensive, especially with large datasets, their accuracy and efficiency in complex spaces make them a valuable asset in the machine learning toolkit.