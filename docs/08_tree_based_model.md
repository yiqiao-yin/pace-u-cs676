# Decision Tree Algorithm Lecture Notes

Back to [home](../README.md)

## What is the Decision Tree Algorithm?

- **Overview**: A decision tree is a supervised learning algorithm used for both classification and regression tasks. It works by splitting the dataset into branches based on certain conditions, making predictions at each branch's end.
- **Structure**: The tree consists of nodes, where each internal node represents a decision on an attribute, and each leaf node represents the outcome. This hierarchical structure makes decision trees interpretable and easy to visualize.
- **Applications**: Decision trees are widely used in industries like healthcare, finance, and marketing for their ability to handle both categorical and numerical data.

## How Does the Model Make Splits in the Predictor Space?

- **Splitting Criterion**: The algorithm splits the predictor space by selecting features that maximize information gain or reduce impurity the most. For classification, metrics like Gini impurity or entropy are used, while regression often uses the Residual Sum of Squares (RSS).
- **Binary Splitting**: Each split divides the data into two groups, creating branches in the tree. The process repeats recursively, forming deeper levels of the tree.
- **Stopping Criteria**: The splitting process halts when all data points in a node belong to the same class, or a specified minimum number of samples in a node is reached.

### Explanation:

- **`gini`:** This function calculates the Gini impurity for a given set of labels.
  
- **`best_split`:** This method searches for the best feature index and threshold to split the dataset that results in the lowest Gini impurity.

- **`build_tree`:** This recursive function builds the decision tree by partitioning the data at each step of recursion based on the best splits found.

- **`train`:** This method fits the decision tree model to the training data.

- **`predict_sample`:** Given a single data point, this method follows the tree structure to predict the label.

- **`predict`:** This function takes in multiple data points and outputs their predicted labels. 

This classifier is a simplistic version of a decision tree and may not handle all edge cases or provide the performance optimizations seen in more advanced libraries like scikit-learn.

## Using RSS as the Main Loss Function to Fit a Decision Tree

- **Residual Sum of Squares (RSS)**: For regression tasks, the decision tree minimizes the RSS at each split. RSS measures the discrepancy between actual and predicted values by summing the squared differences.
- **Fitting the Model**: The model starts with the entire dataset and calculates the potential RSS reduction for each feature and split point. It selects the split that yields the highest RSS reduction.
- **Recursive Process**: This process is repeated recursively, creating splits until the stopping criteria are met, like reaching a maximum depth or a minimum number of samples in a node.

```python
# Example: Using RSS in regression
from sklearn.tree import DecisionTreeRegressor

# Sample data
X = [[1], [2], [3], [4], [5]]
y = [2.3, 2.1, 3.8, 4.5, 5.0]

# Fit the decision tree
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Predicting
predictions = regressor.predict([[3.5]])
print(predictions)
```

## Python Package for Building Decision Tree Classifier and Regressor

- **sklearn.tree.DecisionTreeClassifier**: This class in the `scikit-learn` library is used to build decision tree classifiers. It supports criteria like Gini impurity and entropy for classification tasks.

```python
from sklearn.tree import DecisionTreeClassifier

# Sample data
X = [[0, 0], [1, 1]]
y = [0, 1]

# Initialize and fit the classifier
classifier = DecisionTreeClassifier()
classifier.fit(X, y)

# Making a prediction
print(classifier.predict([[2, 2]]))
```

- **sklearn.tree.DecisionTreeRegressor**: This class is used for regression tasks and uses RSS as the default loss function.

```python
from sklearn.tree import DecisionTreeRegressor

# Sample data
X = [[1], [2], [3]]
y = [1.5, 2.5, 3.5]

# Initialize and fit the regressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Making a prediction
print(regressor.predict([[2.5]]))
```

- **Advantages of sklearn**: The `sklearn` library offers a simple and efficient API for building decision trees. It also provides utilities for tuning hyperparameters like `max_depth` and `min_samples_split` to optimize performance.

## Summary

- Decision trees are powerful, interpretable algorithms for classification and regression tasks.
- The splitting process in decision trees is based on maximizing information gain or minimizing impurity.
- For regression tasks, RSS is used as the primary loss function to decide splits and fit the model.
- The `sklearn.tree` module in Python provides easy-to-use classes for building decision tree classifiers and regressors.

