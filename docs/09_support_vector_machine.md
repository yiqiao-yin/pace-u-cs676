---
sidebar_position: 9
title: "Support Vector Machine"
sidebar_label: "09. Support Vector Machine"
---

# Support Vector Machine

Back to [home](https://github.com/yiqiao-yin/pace-u-cs676)

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

## A Real-World Example: Radar Returns from the Ionosphere

Everything above uses synthetic data, which is useful for seeing the mechanics and
useless for seeing whether any of it works. This section runs the same algorithm on a
real dataset from the **UCI Machine Learning Repository**.

**Ionosphere** is one of the standard benchmarks in the SVM literature: 351 radar
readings, 34 numeric features each, and a binary label for whether the return showed
structure in the ionosphere (`g` for good) or not (`b` for bad). It is a good fit for
this page for one specific reason — **it is not linearly separable**, so you can watch
the kernel trick actually earn something rather than take it on faith.

```bash
pip install ucimlrepo scikit-learn
```

### Linear kernel versus RBF

```python
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iono = fetch_ucirepo(id=52)                 # Ionosphere
X = iono.data.features.to_numpy(dtype=float)
y = (iono.data.targets.to_numpy().ravel() == "g").astype(int)   # 'g' = good return

print("rows:", X.shape[0], " features:", X.shape[1],
      " good:", int(y.sum()), " bad:", int((1 - y).sum()))

cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, kernel in [("linear kernel", "linear"), ("RBF kernel", "rbf")]:
    model = make_pipeline(StandardScaler(), SVC(kernel=kernel))
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(f"{name:14} accuracy = {scores.mean():.4f}  (+/- {scores.std():.4f})")

# rows: 351  features: 34  good: 225  bad: 126
# linear kernel  accuracy = 0.9002  (+/- 0.0462)
# RBF kernel     accuracy = 0.9401  (+/- 0.0247)
```

**The RBF kernel wins by four points**, 0.9401 against 0.9002. That gap is the entire
argument for kernels on this page made concrete: a straight hyperplane in the original 34
dimensions cannot cleanly separate these two classes, and the RBF kernel finds a boundary
that can. Notice also that the RBF result has *half* the fold-to-fold spread (±0.0247
against ±0.0462), so it is both more accurate and more stable here.

Three details in that code worth copying into your own work:

- **`StratifiedKFold`** rather than plain `KFold`, because the classes are 225 to 126
  rather than balanced. See [Session 05](./sampling_and_bootstrap) for why that
  matters.
- **`make_pipeline(StandardScaler(), SVC())`** rather than scaling the data up front. The
  scaler is fitted inside each fold, on the training rows only. Scaling the whole array
  before cross validation lets the validation rows influence the mean and standard
  deviation used to transform them, which is a quiet form of leakage.
- **`== "g"`** converts the string labels to 0/1. Real datasets rarely hand you integers.

### Why SVMs need their features scaled

This is the most common way an SVM silently underperforms, and it is worth seeing rather
than being told. An SVM maximises a *distance* to the decision boundary, so a feature
measured in large numbers dominates that distance regardless of how informative it is.

```python
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iono = fetch_ucirepo(id=52)
X = iono.data.features.to_numpy(dtype=float)
y = (iono.data.targets.to_numpy().ravel() == "g").astype(int)
cv = StratifiedKFold(5, shuffle=True, random_state=42)

X_odd_units = X.copy()
X_odd_units[:, 2] *= 1000        # pretend one column was recorded in different units

for label, data, model in [
    ("scaled,   original units", X,           make_pipeline(StandardScaler(), SVC())),
    ("scaled,   one col x1000",  X_odd_units, make_pipeline(StandardScaler(), SVC())),
    ("UNSCALED, original units", X,           SVC()),
    ("UNSCALED, one col x1000",  X_odd_units, SVC()),
]:
    scores = cross_val_score(model, data, y, cv=cv, scoring="accuracy")
    print(f"{label:26} accuracy = {scores.mean():.4f}")

# scaled,   original units   accuracy = 0.9401
# scaled,   one col x1000    accuracy = 0.9401
# UNSCALED, original units   accuracy = 0.9373
# UNSCALED, one col x1000    accuracy = 0.8062
```

Read the four rows together. With the scaler in the pipeline, rescaling a column changes
**nothing** — 0.9401 either way. Without it, the same change costs **thirteen points**,
dropping to 0.8062, because one column now accounts for almost all of every distance
calculation.

There is an honest wrinkle here worth noticing: the unscaled model on the *original*
units does fine (0.9373). That is luck specific to this dataset — the Ionosphere features
are already normalised to roughly −1 to 1, so there was nothing for a scaler to fix. That
is exactly why the demonstration needs the artificial ×1000. On a dataset where one column
is a dollar amount and another is a ratio, the unscaled column is the real column, and
there is no warning that anything went wrong.

**Always put `StandardScaler` in a pipeline with an SVM.** It costs one line and removes
a failure mode you cannot see.

### Tuning C and gamma

An RBF SVM has two knobs, and they interact:

- **`C`** controls how much the model is penalised for misclassifying training points. Low
  `C` accepts mistakes for a smoother boundary; high `C` bends the boundary to fit them.
- **`gamma`** controls how far the influence of a single training point reaches. Low
  `gamma` gives a broad, smooth boundary; high `gamma` lets individual points carve out
  their own small regions, which overfits.

Searching both at once with cross validation:

```python
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

iono = fetch_ucirepo(id=52)
X = iono.data.features.to_numpy(dtype=float)
y = (iono.data.targets.to_numpy().ravel() == "g").astype(int)

pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
grid = {"svc__C": [0.1, 1, 10, 100], "svc__gamma": ["scale", 0.01, 0.1, 1]}

search = GridSearchCV(pipe, grid,
                      cv=StratifiedKFold(5, shuffle=True, random_state=42),
                      scoring="accuracy")
search.fit(X, y)

print("best C     :", search.best_params_["svc__C"])
print("best gamma :", search.best_params_["svc__gamma"])
print(f"best CV accuracy: {search.best_score_:.4f}")

# best C     : 100
# best gamma : 0.1
# best CV accuracy: 0.9515
```

Tuning moved accuracy from the default **0.9401** to **0.9515** — about one extra correct
prediction in every ninety. Worth having, and worth keeping in proportion: the jump from
a linear kernel to RBF was four points, and tuning on top of that bought one. **Pick the
right kernel first, then tune.**

One caution carried over from [Session 05](./sampling_and_bootstrap): that 0.9515 is
the score of the configuration that *won* this search, chosen using these same folds, so
it is slightly optimistic. If you need an honest estimate of how this model will perform
on new radar readings, wrap the whole search in an outer cross-validation loop or hold out
a test set the search never touches.

### Things to try

1. Swap in a different UCI dataset — `fetch_ucirepo(id=17)` is Breast Cancer Wisconsin,
   `id=109` is Wine. Does RBF still beat linear?
2. Add `kernel="poly"` with `degree=2` and `degree=3` to the comparison.
3. Replace `scoring="accuracy"` with `scoring="roc_auc"`. With 225 good against 126 bad,
   accuracy is flattering.
4. Print `search.best_estimator_[-1].n_support_` to see how many of the 351 rows ended up
   as support vectors. That number *is* the model — every other row could be deleted
   without changing the boundary.

## Conclusion

Support Vector Machines are a versatile tool for both classification and regression. Their strength lies in their ability to handle high-dimensional data and separate classes using non-linear boundaries through kernel functions. While they can be computationally expensive, especially with large datasets, their accuracy and efficiency in complex spaces make them a valuable asset in the machine learning toolkit.