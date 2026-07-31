"""
02 — Logistic Regression from scratch.

Binary classification by gradient descent on the log-loss. Pure numpy.

    python 02_logreg.py
    python 02_logreg.py --plot
    python 02_logreg.py --report

This is homework 01 with two changes: the prediction is squashed through a
sigmoid, and the loss is log-loss instead of squared error. Everything else —
the loop, the step, the bookkeeping — is identical. That is the point. Notice
how little has to change to go from predicting a number to predicting a class.

THIS EXERCISE HAS TWO BLANKS, and the script will complain about the second one
first, because that is the one main() reaches soonest. Do them in this order:

    1. sigmoid()               — the squashing function
    2. fit_gradient_descent()  — the training loop that uses it

Writing the loop first works too; you will just hit the sigmoid error next.
"""

import argparse
import numpy as np

SEED = 676
N_PER_CLASS = 150
CLASS0_CENTER = np.array([-1.3, -0.7])
CLASS1_CENTER = np.array([1.4, 0.9])
SPREAD = 1.1

LEARNING_RATE = 0.15
N_ITERATIONS = 3000


# ---------------------------------------------------------------------------
# 1. Toy data
# ---------------------------------------------------------------------------
def make_data(rng):
    """
    Two overlapping Gaussian blobs, one per class.

    The blobs overlap on purpose. A dataset a straight line can separate
    perfectly teaches you nothing about what the loss is doing — with overlap
    you can watch the model trade one kind of mistake against another.
    """
    X0 = rng.normal(CLASS0_CENTER, SPREAD, size=(N_PER_CLASS, 2))
    X1 = rng.normal(CLASS1_CENTER, SPREAD, size=(N_PER_CLASS, 2))
    X = np.vstack([X0, X1])
    y = np.concatenate([np.zeros(N_PER_CLASS), np.ones(N_PER_CLASS)])

    order = rng.permutation(len(y))       # shuffle so class order carries no signal
    return X[order], y[order]


def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# 2. The pieces of the model
# ---------------------------------------------------------------------------
def sigmoid(z):
    """
    Map any real number to (0, 1) — the squashing function that turns a linear
    score into a probability.

    :param z: numpy array of any shape
    :return:  array of the same shape, every entry strictly between 0 and 1
    """
    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Implement the logistic sigmoid.
    # │
    # │ The definition is   sigma(z) = 1 / (1 + e^-z)
    # │
    # │ Write that and it will work — until z goes very negative, where e^-z
    # │ overflows, numpy warns, and you get nan. Try it: sigmoid(np.array([-800.0])).
    # │
    # │ The fix is to compute the two halves differently. For z < 0, multiply
    # │ top and bottom by e^z and convince yourself that
    # │
    # │       1 / (1 + e^-z)   ==   e^z / (1 + e^z)
    # │
    # │ These are the same function, but the second never exponentiates a large
    # │ positive number when z is negative. So: use the first form where z >= 0
    # │ and the second where z < 0, then stitch the two halves back together.
    # │
    # │ Handling this is worth doing properly. Overflow in the sigmoid is one of
    # │ the classic ways a working model starts producing nan halfway through
    # │ training, and it is invisible until it happens.
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — the sigmoid function
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write the sigmoid function. "
        "See the YOUR TASK box just above for the steps."
    )


def predict_proba(X_design, beta):
    """P(y = 1 | x) for every row."""
    return sigmoid(X_design @ beta)


def predict_label(X_design, beta, threshold=0.5):
    """Hard 0/1 prediction. The threshold is a decision, not a fact — try moving it."""
    return (predict_proba(X_design, beta) >= threshold).astype(int)


def log_loss(y, p, eps=1e-12):
    """
    Mean negative log-likelihood.

    Clipping keeps log(0) out of the arithmetic when the model becomes confident.
    """
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def gradient(X_design, y, beta):
    """
    Gradient of the log-loss.

        1/n * X.T @ (sigmoid(X beta) - y)

    Compare this with the gradient in homework 01. Same shape, no sigmoid
    derivative in sight — the log-loss is chosen precisely because it cancels.
    """
    n = X_design.shape[0]
    return X_design.T @ (predict_proba(X_design, beta) - y) / n


# ---------------------------------------------------------------------------
# 3. Fitting
# ---------------------------------------------------------------------------
def fit_gradient_descent(X_design, y, lr=LEARNING_RATE, n_iter=N_ITERATIONS):
    """
    Fit beta by batch gradient descent on the log-loss.

    :return: (beta, history) where history is the log-loss at each iteration.
    """
    beta = np.zeros(X_design.shape[1])
    history = []

    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Write the gradient descent loop.
    # │
    # │ Repeat `n_iter` times:
    # │   1. p = predict_proba(X_design, beta)
    # │   2. append log_loss(y, p) to `history`
    # │   3. grad = gradient(X_design, y, beta)
    # │   4. beta = beta - lr * grad
    # │
    # │ If this looks like your answer to homework 01, that is correct — only
    # │ the loss and the prediction changed. If your loss stops falling early,
    # │ the learning rate is too small; if it oscillates, too large.
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — the gradient descent loop
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write the gradient descent loop. "
        "See the YOUR TASK box just above for the steps."
    )

    return beta, history


# ---------------------------------------------------------------------------
# 4. Reporting
# ---------------------------------------------------------------------------
def confusion(y_true, y_pred):
    """Return (tn, fp, fn, tp) — the four ways a binary prediction can land."""
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return tn, fp, fn, tp


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion(y_true, y_pred)
    acc = (tp + tn) / len(y_true)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def report(beta, history, y, y_hat, p):
    m = metrics(y, y_hat)
    print("=" * 62)
    print("  LOGISTIC REGRESSION — gradient descent on the log-loss")
    print("=" * 62)
    print(f"\n  coefficients: intercept={beta[0]:+.4f}  x1={beta[1]:+.4f}  x2={beta[2]:+.4f}")
    print(f"  iterations   : {len(history)}")
    print(f"  log-loss     : {history[0]:.4f} -> {history[-1]:.4f}")

    print("\n  confusion matrix")
    print("                 pred 0   pred 1")
    print(f"      true 0  {m['tn']:>8}{m['fp']:>9}")
    print(f"      true 1  {m['fn']:>8}{m['tp']:>9}")

    print(f"\n  accuracy  {m['accuracy']:.4f}")
    print(f"  precision {m['precision']:.4f}")
    print(f"  recall    {m['recall']:.4f}")
    print(f"  f1        {m['f1']:.4f}")

    # A model that only ever guessed the majority class would score this:
    baseline = max(np.mean(y), 1 - np.mean(y))
    print(f"\n  majority-class baseline: {baseline:.4f}   <- you must beat this")
    verdict = "PASS" if m["accuracy"] > baseline else "NO BETTER THAN GUESSING — check the loop"
    print(f"  [{verdict}]")
    print("=" * 62)
    return m, baseline


def save_plot(X, y, beta, history, path="02_logreg_fit.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped — pip install matplotlib]")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    # Decision boundary: the line where beta0 + beta1*x1 + beta2*x2 = 0
    xs = np.linspace(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5, 200)
    if abs(beta[2]) > 1e-9:
        ax1.plot(xs, -(beta[0] + beta[1] * xs) / beta[2], color="black", lw=2,
                 label="decision boundary")
    ax1.scatter(X[y == 0, 0], X[y == 0, 1], s=16, alpha=0.6, label="class 0")
    ax1.scatter(X[y == 1, 0], X[y == 1, 1], s=16, alpha=0.6, label="class 1")
    ax1.set_xlabel("x1"); ax1.set_ylabel("x2"); ax1.set_title("Data and boundary"); ax1.legend()

    ax2.plot(history, lw=1.6)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("log-loss"); ax2.set_title("Loss curve")

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  [plot saved -> {path}]")
    return path


def save_report(beta, history, m, baseline, path="02_logreg_report.md"):
    text = f"""# Homework 02 — Logistic Regression

Fitted {2 * N_PER_CLASS} observations (two overlapping Gaussian blobs) by batch
gradient descent on the log-loss, learning rate {LEARNING_RATE},
{len(history)} iterations.

## Model

| term | coefficient |
| --- | --- |
| intercept | {beta[0]:+.4f} |
| x1 | {beta[1]:+.4f} |
| x2 | {beta[2]:+.4f} |

## Confusion matrix

| | predicted 0 | predicted 1 |
| --- | --- | --- |
| **actual 0** | {m['tn']} | {m['fp']} |
| **actual 1** | {m['fn']} | {m['tp']} |

## Metrics

| metric | value |
| --- | --- |
| accuracy | {m['accuracy']:.4f} |
| precision | {m['precision']:.4f} |
| recall | {m['recall']:.4f} |
| f1 | {m['f1']:.4f} |
| majority-class baseline | {baseline:.4f} |
| log-loss, first iteration | {history[0]:.4f} |
| log-loss, last iteration | {history[-1]:.4f} |

## Reading the numbers

Accuracy alone is not enough: always compare against the majority-class baseline
of {baseline:.4f}, which is what a model that ignores the features entirely would
score. The remaining errors sit where the two blobs overlap, and no straight line
can fix those — they are a property of the data, not of the fit.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [report saved -> {path}]")
    return path


def main():
    ap = argparse.ArgumentParser(description="Logistic regression from scratch.")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    X, y = make_data(rng)
    X_design = add_intercept(X)

    beta, history = fit_gradient_descent(X_design, y)
    p = predict_proba(X_design, beta)
    y_hat = predict_label(X_design, beta)

    m, baseline = report(beta, history, y, y_hat, p)
    if args.plot:
        save_plot(X, y, beta, history)
    if args.report:
        save_report(beta, history, m, baseline)


if __name__ == "__main__":
    main()
