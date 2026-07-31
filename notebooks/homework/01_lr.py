"""
01 — Linear Regression from scratch.

Fit y = b0 + b1*x1 + b2*x2 using batch gradient descent, written by hand.
No scikit-learn, no statsmodels — the point is to see every step.

    python 01_lr.py
    python 01_lr.py --plot     # save a PNG of fit + loss curve
    python 01_lr.py --report   # write a markdown lab report

Why gradient descent when a closed form exists? Because the closed form only
works for linear regression. The loop you write here is the same loop that
trains logistic regression (homework 02) and every neural network in the course.
We fit both ways and compare — that comparison is your correctness check.
"""

import argparse
import numpy as np

SEED = 676
N_SAMPLES = 200
TRUE_INTERCEPT = 3.0
TRUE_COEFS = np.array([2.5, -1.2])
NOISE_SD = 1.5

LEARNING_RATE = 0.05
N_ITERATIONS = 2000


# ---------------------------------------------------------------------------
# 1. Toy data
# ---------------------------------------------------------------------------
def make_data(rng):
    """
    Build a small regression dataset with a known answer.

    Because we choose the true coefficients ourselves, we can check whether the
    fitting loop recovered them. Real data never gives you that luxury, which is
    exactly why toy data is the right place to learn the algorithm.
    """
    X = rng.normal(loc=0.0, scale=1.0, size=(N_SAMPLES, 2))
    noise = rng.normal(loc=0.0, scale=NOISE_SD, size=N_SAMPLES)
    y = TRUE_INTERCEPT + X @ TRUE_COEFS + noise
    return X, y


def add_intercept(X):
    """Prepend a column of ones so the intercept is just another coefficient."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# 2. The pieces of the model
# ---------------------------------------------------------------------------
def predict(X_design, beta):
    """yhat = X @ beta. One matrix multiply is the whole forward pass."""
    return X_design @ beta


def mse(y_true, y_pred):
    """Mean squared error — the loss we are minimising."""
    return float(np.mean((y_true - y_pred) ** 2))


def gradient(X_design, y, beta):
    """
    Gradient of the MSE with respect to beta.

    d/dbeta  mean((y - X beta)^2)  =  -2/n * X.T @ (y - X beta)

    Work that derivative out on paper once. Everything else in this course that
    "learns" is doing this same thing with a different loss.
    """
    n = X_design.shape[0]
    residual = y - predict(X_design, beta)
    return -2.0 / n * (X_design.T @ residual)


def normal_equation(X_design, y):
    """
    Closed-form least squares: beta = (X'X)^-1 X'y.

    This is the exact answer. We use it only to check the loop — pinv is used
    instead of inv so a singular X'X does not blow up.
    """
    return np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y


# ---------------------------------------------------------------------------
# 3. Fitting
# ---------------------------------------------------------------------------
def fit_gradient_descent(X_design, y, lr=LEARNING_RATE, n_iter=N_ITERATIONS):
    """
    Fit beta by batch gradient descent.

    :return: (beta, history) where history is the loss at each iteration.
    """
    beta = np.zeros(X_design.shape[1])
    history = []

    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Write the gradient descent loop.
    # │
    # │ Repeat `n_iter` times:
    # │   1. compute the current predictions
    # │   2. record the loss with mse(), appending to `history`
    # │   3. compute the gradient with gradient()
    # │   4. step downhill:  beta = beta - lr * grad
    # │
    # │ Two rules that catch most mistakes:
    # │   - update ALL coefficients at once; do not loop over them one at a time
    # │   - subtract the gradient, do not add it (you are minimising)
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
def r_squared(y_true, y_pred):
    """Fraction of variance explained. 1.0 is perfect, 0.0 is no better than the mean."""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot)


def report(beta_gd, beta_cf, history, y, y_pred):
    print("=" * 62)
    print("  LINEAR REGRESSION — gradient descent vs closed form")
    print("=" * 62)
    names = ["intercept", "x1 coef", "x2 coef"]
    truth = [TRUE_INTERCEPT, *TRUE_COEFS]
    print(f"\n  {'':<12}{'true':>10}{'gradient':>12}{'closed form':>14}{'gap':>9}")
    for i, name in enumerate(names):
        gap = abs(beta_gd[i] - beta_cf[i])
        print(f"  {name:<12}{truth[i]:>10.4f}{beta_gd[i]:>12.4f}{beta_cf[i]:>14.4f}{gap:>9.5f}")

    print(f"\n  iterations run     : {len(history)}")
    print(f"  loss, first iter   : {history[0]:.4f}")
    print(f"  loss, last iter    : {history[-1]:.4f}")
    print(f"  MSE  (final)       : {mse(y, y_pred):.4f}")
    print(f"  R^2  (final)       : {r_squared(y, y_pred):.4f}")
    print(f"  noise floor (sd^2) : {NOISE_SD ** 2:.4f}   <- MSE cannot beat this")

    gap = float(np.max(np.abs(beta_gd - beta_cf)))
    verdict = "PASS" if gap < 0.01 else "TOO FAR APART — check the loop"
    print(f"\n  max gap vs closed form: {gap:.6f}   [{verdict}]")
    print("=" * 62)
    return gap


def save_plot(X, y, beta, history, path="01_lr_fit.png"):
    """Optional. Skips cleanly when matplotlib is not installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped — pip install matplotlib]")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    order = np.argsort(X[:, 0])
    ax1.scatter(X[:, 0], y, s=16, alpha=0.55, label="observed")
    ax1.plot(X[order, 0], predict(add_intercept(X), beta)[order], lw=2, color="crimson",
             label="fitted")
    ax1.set_xlabel("x1"); ax1.set_ylabel("y"); ax1.set_title("Fit (vs x1)"); ax1.legend()

    ax2.plot(history, lw=1.6)
    ax2.set_xlabel("iteration"); ax2.set_ylabel("MSE"); ax2.set_yscale("log")
    ax2.set_title("Loss curve")

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  [plot saved -> {path}]")
    return path


def save_report(beta_gd, beta_cf, history, y, y_pred, gap, path="01_lr_report.md"):
    """Optional markdown lab report — the shape a written submission should take."""
    rows = "\n".join(
        f"| {n} | {t:.4f} | {beta_gd[i]:.4f} | {beta_cf[i]:.4f} |"
        for i, (n, t) in enumerate(zip(["intercept", "x1", "x2"], [TRUE_INTERCEPT, *TRUE_COEFS]))
    )
    text = f"""# Homework 01 — Linear Regression

Fitted {N_SAMPLES} observations by batch gradient descent
(learning rate {LEARNING_RATE}, {len(history)} iterations), then checked the
result against the closed-form least squares solution.

## Coefficients

| term | true | gradient descent | closed form |
| --- | --- | --- | --- |
{rows}

## Fit

| metric | value |
| --- | --- |
| MSE | {mse(y, y_pred):.4f} |
| R-squared | {r_squared(y, y_pred):.4f} |
| loss at first iteration | {history[0]:.4f} |
| loss at last iteration | {history[-1]:.4f} |
| max gap vs closed form | {gap:.6f} |

## Reading the numbers

The MSE settles near {NOISE_SD ** 2:.2f}, which is the variance of the noise we
added. No model can do better than that on this data — the remaining error is not
a modelling failure, it is the noise itself.

The gradient-descent and closed-form coefficients agree to {gap:.6f}, so the loop
is finding the same optimum the algebra does.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [report saved -> {path}]")
    return path


def main():
    ap = argparse.ArgumentParser(description="Linear regression from scratch.")
    ap.add_argument("--plot", action="store_true", help="save a PNG of the fit and loss curve")
    ap.add_argument("--report", action="store_true", help="write a markdown lab report")
    args = ap.parse_args()

    rng = np.random.default_rng(SEED)
    X, y = make_data(rng)
    X_design = add_intercept(X)

    beta_gd, history = fit_gradient_descent(X_design, y)
    beta_cf = normal_equation(X_design, y)
    y_pred = predict(X_design, beta_gd)

    gap = report(beta_gd, beta_cf, history, y, y_pred)
    if args.plot:
        save_plot(X, y, beta_gd, history)
    if args.report:
        save_report(beta_gd, beta_cf, history, y, y_pred, gap)


if __name__ == "__main__":
    main()
