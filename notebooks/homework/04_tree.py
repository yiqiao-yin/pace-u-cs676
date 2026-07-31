"""
04 — Decision Tree from scratch.

Grow a classification tree by exhaustively searching for the split that lowers
Gini impurity the most, then recursing on each side.

    python 04_tree.py
    python 04_tree.py --plot
    python 04_tree.py --report

The SimpleDecisionTreeClassifier below is the class from the course notes
(docs/08_tree_based_model.md), used as-is. What this homework adds is toy data
generated with numpy, a train/test split so the accuracy means something, and a
printed picture of the tree that got grown.

A tree is not fitted by gradient descent. There is no loss surface to roll down —
instead you search, greedily, one split at a time. That difference is the point
of this exercise.
"""

import argparse
import numpy as np

SEED = 676
N_PER_CLASS = 120
TEST_FRACTION = 0.30


# ===========================================================================
# The classifier from the course notes — docs/08_tree_based_model.md
# ===========================================================================
class SimpleDecisionTreeClassifier:
    def __init__(self):
        self.tree = None

    def gini(self, y):
        """Calculate the Gini Impurity for a list of labels."""
        prob = np.bincount(y) / len(y)
        return 1 - np.sum(prob ** 2)

    def best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        best_gini = float('inf')
        best_idx = None
        best_threshold = None

        n_samples, n_features = X.shape

        # ┌─ YOUR TASK ─────────────────────────────────────────────────────────
        # │ Write the exhaustive search for the best split.
        # │
        # │ For every feature index `idx` in range(n_features):
        # │   for every candidate `threshold` in np.unique(X[:, idx]):
        # │     1. left_indices  = X[:, idx] <= threshold
        # │        right_indices = X[:, idx] >  threshold
        # │     2. skip the candidate if either side came out empty
        # │     3. weighted_gini = (n_left * gini(left) + n_right * gini(right)) / n_samples
        # │     4. keep idx and threshold if weighted_gini beats best_gini
        # │
        # │ Notes:
        # │   - use self.gini(...) — it is already written for you
        # │   - weight each side by its SIZE. An unweighted average would call a
        # │     2-row pure leaf as good as a 200-row pure leaf.
        # │   - this is O(features x thresholds x rows). Real implementations sort
        # │     once per feature and sweep; yours does not have to.
        # └──────────────────────────────────────────────────────────────────────
        # YOUR CODE HERE — the exhaustive split search
        # Delete the raise below once you have written it.
        raise NotImplementedError(
            "Homework: write the exhaustive split search. "
            "See the YOUR TASK box just above for the steps."
        )

        return best_idx, best_threshold

    def build_tree(self, X, y):
        """Build the decision tree recursively."""
        if len(np.unique(y)) == 1:
            return {'label': y[0]}

        best_idx, best_threshold = self.best_split(X, y)
        if best_idx is None:
            return {'label': np.bincount(y).argmax()}

        left_indices = X[:, best_idx] <= best_threshold
        right_indices = X[:, best_idx] > best_threshold
        return {
            'feature_index': best_idx,
            'threshold': best_threshold,
            'left': self.build_tree(X[left_indices], y[left_indices]),
            'right': self.build_tree(X[right_indices], y[right_indices])
        }

    def train(self, X, y):
        """Fit the decision tree on the data."""
        self.tree = self.build_tree(X, y)

    def predict_sample(self, node, x):
        """Predict a single sample based on the built tree."""
        if 'label' in node:
            return node['label']

        if x[node['feature_index']] <= node['threshold']:
            return self.predict_sample(node['left'], x)
        else:
            return self.predict_sample(node['right'], x)

    def predict(self, X):
        """Predict class labels for samples in X."""
        return np.array([self.predict_sample(self.tree, x) for x in X])


# ---------------------------------------------------------------------------
# 1. Toy data
# ---------------------------------------------------------------------------
def make_data(rng):
    """
    Two classes that a few axis-aligned cuts can mostly separate.

    Class 1 sits in two separate clumps. That is deliberate: a single straight
    line cannot capture it, but a tree can, because it slices the space one axis
    at a time and does not care whether the result is connected.
    """
    class0 = rng.normal([0.0, 0.0], 0.9, size=(N_PER_CLASS, 2))
    half = N_PER_CLASS // 2
    class1 = np.vstack([
        rng.normal([3.2, 3.0], 0.8, size=(half, 2)),
        rng.normal([-3.0, 3.4], 0.8, size=(N_PER_CLASS - half, 2)),
    ])
    X = np.vstack([class0, class1])
    y = np.concatenate([np.zeros(N_PER_CLASS, dtype=int),
                        np.ones(N_PER_CLASS, dtype=int)])
    order = rng.permutation(len(y))
    return X[order], y[order]


def train_test_split(X, y, test_fraction, rng):
    """
    Hold out a slice for testing.

    A tree grown to purity will classify its own training data perfectly — its
    training accuracy is guaranteed to be 1.0 and tells you nothing. The held-out
    rows are the only honest measurement here.
    """
    n_test = int(round(len(y) * test_fraction))
    idx = rng.permutation(len(y))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], y[train_idx], X[test_idx], y[test_idx]


# ---------------------------------------------------------------------------
# 2. Inspecting the tree
# ---------------------------------------------------------------------------
def describe_tree(node, depth=0, prefix="root"):
    """Print the tree as indented text so you can read the rules it learned."""
    pad = "    " * depth
    if 'label' in node:
        print(f"{pad}{prefix}: predict class {int(node['label'])}")
        return
    print(f"{pad}{prefix}: is x{node['feature_index']} <= {node['threshold']:.3f} ?")
    describe_tree(node['left'], depth + 1, "yes")
    describe_tree(node['right'], depth + 1, "no ")


def tree_stats(node):
    """Return (n_leaves, max_depth) for a fitted tree."""
    if 'label' in node:
        return 1, 1
    l_leaves, l_depth = tree_stats(node['left'])
    r_leaves, r_depth = tree_stats(node['right'])
    return l_leaves + r_leaves, 1 + max(l_depth, r_depth)


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# 3. Reporting
# ---------------------------------------------------------------------------
def report(clf, X_tr, y_tr, X_te, y_te):
    train_acc = accuracy(y_tr, clf.predict(X_tr))
    test_acc = accuracy(y_te, clf.predict(X_te))
    n_leaves, depth = tree_stats(clf.tree)

    print("=" * 64)
    print("  DECISION TREE — greedy Gini splitting")
    print("=" * 64)
    print(f"\n  training rows : {len(y_tr)}")
    print(f"  test rows     : {len(y_te)}")
    print(f"  leaves        : {n_leaves}")
    print(f"  depth         : {depth}")

    print(f"\n  training accuracy : {train_acc:.4f}   <- grown to purity, so this is ~1.0")
    print(f"  test accuracy     : {test_acc:.4f}   <- the number that counts")

    baseline = max(np.mean(y_te), 1 - np.mean(y_te))
    print(f"  majority baseline : {baseline:.4f}")
    verdict = "PASS" if test_acc > baseline else "NO BETTER THAN GUESSING — check best_split"
    print(f"  [{verdict}]")

    print("\n  the tree it learned:\n")
    describe_tree(clf.tree)
    print("=" * 64)
    return train_acc, test_acc, n_leaves, depth, baseline


def save_plot(X_tr, y_tr, X_te, y_te, clf, path="04_tree_regions.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped — pip install matplotlib]")
        return None

    X = np.vstack([X_tr, X_te])
    pad = 0.7
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, 220),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, 220),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    zz = clf.predict(grid).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.contourf(xx, yy, zz, alpha=0.18, levels=[-0.5, 0.5, 1.5])
    ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], s=16, label="train c0")
    ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], s=16, label="train c1")
    ax.scatter(X_te[:, 0], X_te[:, 1], s=30, facecolors="none", edgecolors="black",
               linewidths=0.8, label="test")
    ax.set_xlabel("x0"); ax.set_ylabel("x1")
    ax.set_title("Decision regions — note the axis-aligned edges")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  [plot saved -> {path}]")
    return path


def save_report(train_acc, test_acc, n_leaves, depth, baseline, path="04_tree_report.md"):
    text = f"""# Homework 04 — Decision Tree

Grew a classification tree with the `SimpleDecisionTreeClassifier` from the
course notes on {2 * N_PER_CLASS} synthetic points in two dimensions, holding out
{int(TEST_FRACTION * 100)}% for testing.

## Results

| quantity | value |
| --- | --- |
| training accuracy | {train_acc:.4f} |
| test accuracy | {test_acc:.4f} |
| majority-class baseline | {baseline:.4f} |
| leaves | {n_leaves} |
| depth | {depth} |

## Reading the numbers

Training accuracy is {train_acc:.4f}. That is not a sign of a good model — this
tree splits until every leaf is pure, so it can always memorise its training set.
Any tree grown without a stopping rule will report roughly 1.0 here. The test
accuracy of {test_acc:.4f} is the only number that says anything about
generalisation.

The tree has {n_leaves} leaves at depth {depth}. Each leaf is a rectangle in
feature space, because every split compares one feature against one threshold.
That is why the decision regions have square corners, and why a tree needs a
staircase of splits to approximate a diagonal boundary that logistic regression
would capture with a single line.

## Something to try

Add a stopping rule — refuse to split a node with fewer than, say, five rows, or
cap the depth. Watch training accuracy fall and test accuracy hold steady or
improve. That gap closing is exactly what pruning is for.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [report saved -> {path}]")
    return path


def main():
    ap = argparse.ArgumentParser(description="Decision tree from scratch.")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    X, y = make_data(rng)
    X_tr, y_tr, X_te, y_te = train_test_split(X, y, TEST_FRACTION, rng)

    clf = SimpleDecisionTreeClassifier()
    clf.train(X_tr, y_tr)

    train_acc, test_acc, n_leaves, depth, baseline = report(clf, X_tr, y_tr, X_te, y_te)
    if args.plot:
        save_plot(X_tr, y_tr, X_te, y_te, clf)
    if args.report:
        save_report(train_acc, test_acc, n_leaves, depth, baseline)


if __name__ == "__main__":
    main()
