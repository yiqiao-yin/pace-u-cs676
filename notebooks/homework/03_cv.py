"""
03 — K-Fold Cross Validation by hand.

Build an n x p dataset, split it into k folds, and rotate: each fold takes a
turn as the validation set while the other k-1 folds train the model.

    python 03_cv.py
    python 03_cv.py --folds 5
    python 03_cv.py --report

The model here is deliberately boring — closed-form least squares, solved in one
line — because the model is not the lesson. The lesson is the *rotation*: how you
get an honest estimate of out-of-sample error from a dataset you only have once.

The script also fits the full dataset and scores it on itself. Compare that
number with the cross-validated one. The gap between them is the whole reason
cross validation exists.

TWO BLANKS in this exercise. Suggested order:

    1. make_folds()     — shuffle and cut the indices
    2. cross_validate() — the rotation loop that uses them
"""

import argparse
import numpy as np

SEED = 676
N_SAMPLES = 240
N_FEATURES = 30         # p — deliberately wide relative to n
N_INFORMATIVE = 3       # only the first 3 features actually drive y;
                        # the other 27 are pure noise the model will still fit
NOISE_SD = 2.0
DEFAULT_FOLDS = 10


# ---------------------------------------------------------------------------
# 1. Toy data
# ---------------------------------------------------------------------------
def make_data(rng):
    """
    An n x p design matrix where only some columns matter.

    The last p - N_INFORMATIVE columns are pure noise. A model fitted on all of
    them will use them anyway — that is overfitting, and cross validation is how
    you catch it.
    """
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    true_beta = np.zeros(N_FEATURES)
    true_beta[:N_INFORMATIVE] = rng.uniform(1.0, 3.0, size=N_INFORMATIVE)
    y = X @ true_beta + rng.normal(scale=NOISE_SD, size=N_SAMPLES)
    return X, y, true_beta


def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


# ---------------------------------------------------------------------------
# 2. The model — one line, on purpose
# ---------------------------------------------------------------------------
def fit_ols(X_design, y):
    """Closed-form least squares. pinv keeps a singular X'X from exploding."""
    return np.linalg.pinv(X_design.T @ X_design) @ X_design.T @ y


def rmse(y_true, y_pred):
    """Root mean squared error — same units as y, which makes it readable."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ---------------------------------------------------------------------------
# 3. Fold construction
# ---------------------------------------------------------------------------
def make_folds(n, k, rng):
    """
    Split indices 0..n-1 into k roughly equal, shuffled folds.

    :param n:   number of rows in the dataset
    :param k:   number of folds
    :param rng: a numpy Generator, so the split is reproducible
    :return:    a list of k index arrays. Every index 0..n-1 appears in exactly
                one of them.
    """
    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Build the folds.
    # │
    # │ Two steps:
    # │   1. shuffle the indices 0..n-1     (rng.permutation(n) gives you these
    # │      already shuffled, in one call)
    # │   2. cut the shuffled sequence into k pieces
    # │
    # │ For step 2, np.array_split(arr, k) is what you want, NOT np.split.
    # │ np.split refuses when k does not divide n exactly; array_split handles it
    # │ by giving the first n % k folds one extra element. With n=240 and k=10
    # │ you would not notice the difference, but with k=7 np.split raises.
    # │
    # │ Why shuffle at all? Because real data arrives ordered — by date, by
    # │ customer, by whoever exported the file. Take contiguous blocks of a
    # │ time-sorted dataset and every fold trains on the past and validates on
    # │ the future, or worse, the reverse. The scores come out confidently wrong.
    # │
    # │ Sanity check your result: the k pieces should have lengths summing to n,
    # │ and np.sort(np.concatenate(folds)) should equal np.arange(n).
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — shuffle and split the indices
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write shuffle and split the indices. "
        "See the YOUR TASK box just above for the steps."
    )


# ---------------------------------------------------------------------------
# 4. The cross-validation loop
# ---------------------------------------------------------------------------
def cross_validate(X, y, k, rng):
    """
    Run k-fold cross validation and return the per-fold validation RMSEs.

    :return: (val_scores, train_scores) — two lists of length k.
    """
    folds = make_folds(len(y), k, rng)
    val_scores, train_scores = [], []

    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Write the rotation loop.
    # │
    # │ For each i in range(k):
    # │   1. val_idx   = folds[i]
    # │   2. train_idx = every index from the OTHER folds, concatenated
    # │                  (np.concatenate over folds[j] for j != i)
    # │   3. fit on the training rows only:
    # │         beta = fit_ols(add_intercept(X[train_idx]), y[train_idx])
    # │   4. score on BOTH:
    # │         append rmse(y[val_idx],   add_intercept(X[val_idx])   @ beta) to val_scores
    # │         append rmse(y[train_idx], add_intercept(X[train_idx]) @ beta) to train_scores
    # │
    # │ The one rule that matters: the validation rows must never appear in the
    # │ data you fit on. If they do, your score is a measure of memory, not of
    # │ prediction — and it will look suspiciously good.
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — the k-fold rotation loop
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write the k-fold rotation loop. "
        "See the YOUR TASK box just above for the steps."
    )

    return val_scores, train_scores


# ---------------------------------------------------------------------------
# 5. Reporting
# ---------------------------------------------------------------------------
def report(val_scores, train_scores, k, resub_rmse, folds_sizes):
    print("=" * 66)
    print(f"  {k}-FOLD CROSS VALIDATION — ordinary least squares")
    print("=" * 66)
    print(f"\n  {'fold':>6}{'n_val':>8}{'train RMSE':>14}{'val RMSE':>12}{'gap':>10}")
    for i, (tr, va) in enumerate(zip(train_scores, val_scores), start=1):
        print(f"  {i:>6}{folds_sizes[i-1]:>8}{tr:>14.4f}{va:>12.4f}{va - tr:>10.4f}")

    va = np.array(val_scores)
    tr = np.array(train_scores)
    print(f"\n  validation RMSE : {va.mean():.4f}  +/- {va.std(ddof=1):.4f}  (sd across folds)")
    print(f"  training   RMSE : {tr.mean():.4f}")
    print(f"  range           : {va.min():.4f} to {va.max():.4f}")

    print(f"\n  resubstitution RMSE (fit and score on ALL data): {resub_rmse:.4f}")
    print(f"  cross-validated RMSE                           : {va.mean():.4f}")
    print(f"  optimism (how much the naive number flatters)  : {va.mean() - resub_rmse:.4f}")
    print(f"\n  noise floor (sd of the noise we added)         : {NOISE_SD:.4f}")
    print("=" * 66)
    return va, tr


def save_report(val_scores, train_scores, k, resub_rmse, path="03_cv_report.md"):
    va = np.array(val_scores)
    rows = "\n".join(
        f"| {i} | {tr:.4f} | {v:.4f} |"
        for i, (tr, v) in enumerate(zip(train_scores, val_scores), start=1)
    )
    text = f"""# Homework 03 — K-Fold Cross Validation

{k}-fold cross validation over {N_SAMPLES} observations and {N_FEATURES} features,
of which only {N_INFORMATIVE} actually drive the response. Model: ordinary least
squares, solved in closed form.

## Per-fold results

| fold | train RMSE | validation RMSE |
| --- | --- | --- |
{rows}

## Summary

| quantity | value |
| --- | --- |
| mean validation RMSE | {va.mean():.4f} |
| sd across folds | {va.std(ddof=1):.4f} |
| mean training RMSE | {np.mean(train_scores):.4f} |
| resubstitution RMSE (all data) | {resub_rmse:.4f} |
| optimism | {va.mean() - resub_rmse:.4f} |
| noise sd used to generate y | {NOISE_SD:.4f} |

## Reading the numbers

Training RMSE is lower than validation RMSE in essentially every fold. That gap
is the model fitting noise it has already seen — including the
{N_FEATURES - N_INFORMATIVE} features that are pure noise by construction.

The resubstitution RMSE, computed by fitting and scoring on the same rows, is the
most optimistic number available and the one most often reported by mistake. The
cross-validated figure is higher and is the honest one.

The spread across folds ({va.std(ddof=1):.4f}) is worth as much attention as the
mean. A single train/test split would have handed you one draw from that spread
and no way to know how lucky it was.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [report saved -> {path}]")
    return path


def main():
    ap = argparse.ArgumentParser(description="K-fold cross validation by hand.")
    ap.add_argument("--folds", type=int, default=DEFAULT_FOLDS, help=f"k (default {DEFAULT_FOLDS})")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.folds < 2 or args.folds > N_SAMPLES:
        raise SystemExit(f"--folds must be between 2 and {N_SAMPLES}")

    rng = np.random.default_rng(SEED)
    X, y, _ = make_data(rng)

    val_scores, train_scores = cross_validate(X, y, args.folds, rng)

    # The naive number, for contrast: fit on everything, score on everything.
    beta_all = fit_ols(add_intercept(X), y)
    resub = rmse(y, add_intercept(X) @ beta_all)

    # Fold sizes depend only on n and k, not on the shuffle, so derive them
    # directly rather than re-drawing folds that would not match the ones used.
    sizes = [len(f) for f in np.array_split(np.arange(len(y)), args.folds)]
    report(val_scores, train_scores, args.folds, resub, sizes)

    if args.report:
        save_report(val_scores, train_scores, args.folds, resub)


if __name__ == "__main__":
    main()
