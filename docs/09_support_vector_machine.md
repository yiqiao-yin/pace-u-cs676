# Support Vector Machine

Back to [home](../README.md)

## Introduction to SVM

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

## Conclusion

Support Vector Machines are a versatile tool for both classification and regression. Their strength lies in their ability to handle high-dimensional data and separate classes using non-linear boundaries through kernel functions. While they can be computationally expensive, especially with large datasets, their accuracy and efficiency in complex spaces make them a valuable asset in the machine learning toolkit.