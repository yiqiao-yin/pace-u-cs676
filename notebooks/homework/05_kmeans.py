"""
05 — K-Means Clustering from scratch.

Lloyd's algorithm in pure numpy: assign every point to its nearest centroid,
move each centroid to the mean of its points, repeat until nothing moves.

    uv run 05_kmeans.py
    uv run 05_kmeans.py --k 4
    uv run 05_kmeans.py --plot --report

The first four homeworks were supervised — every row came with an answer. This
one has no labels at all. Nothing tells the algorithm it is right, so "converged"
has to mean something you define yourself: the assignments stopped changing.

We do generate the data from three known blobs, but the algorithm never sees
those. They exist only so you can check afterwards whether it found them.

THREE BLANKS in this exercise — it is the last one, and you build the whole
algorithm. Suggested order, bottom up:

    1. assign_clusters()   — the ASSIGN step
    2. update_centroids()  — the UPDATE step
    3. fit_kmeans()        — the loop that alternates them until nothing moves
"""

import argparse
import numpy as np

SEED = 676
N_PER_BLOB = 90
TRUE_CENTERS = np.array([[-3.0, -2.0], [0.5, 3.2], [3.6, -1.4]])
BLOB_SPREAD = 0.85

DEFAULT_K = 3
MAX_ITERATIONS = 100


# ---------------------------------------------------------------------------
# 1. Toy data
# ---------------------------------------------------------------------------
def make_data(rng):
    """
    Three Gaussian blobs. Returns the points and the true blob ids.

    The true ids are for scoring only. Passing them to the algorithm would be
    cheating, and would also miss the point — in a real clustering problem they
    do not exist.
    """
    chunks, labels = [], []
    for i, c in enumerate(TRUE_CENTERS):
        chunks.append(rng.normal(c, BLOB_SPREAD, size=(N_PER_BLOB, 2)))
        labels.append(np.full(N_PER_BLOB, i))
    X = np.vstack(chunks)
    true_ids = np.concatenate(labels)
    order = rng.permutation(len(true_ids))
    return X[order], true_ids[order]


# ---------------------------------------------------------------------------
# 2. The pieces of the algorithm
# ---------------------------------------------------------------------------
def init_centroids(X, k, rng):
    """
    Pick k distinct rows at random as starting centroids.

    This is the plain initialisation, and it is genuinely fragile — a bad draw
    can leave two centroids inside one blob and none in another. k-means++ exists
    to fix that. Run this script a few times with different seeds to see it.
    """
    idx = rng.choice(len(X), size=k, replace=False)
    return X[idx].copy()


def assign_clusters(X, centroids):
    """
    Label each point with the index of its nearest centroid — the ASSIGN step.

    :param X:         (n, d) array of points
    :param centroids: (k, d) array of centroids
    :return:          (n,) integer array; entry i is the centroid index closest
                      to point i
    """
    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ For every point, find which centroid is nearest.
    # │
    # │ A plain double loop over points and centroids is perfectly correct, and
    # │ if that is where you start, good. But try to do it without looping over
    # │ points, because the vectorised version is the one worth learning:
    # │
    # │   X[:, None, :]        has shape (n, 1, d)
    # │   centroids[None,:,:]  has shape (1, k, d)
    # │   subtracting them broadcasts to (n, k, d)   <- every point-centroid pair
    # │   square, then sum over the last axis        -> (n, k) of distances
    # │   np.argmin(..., axis=1)                     -> (n,) nearest index
    # │
    # │ Note you never need np.sqrt. The square root is monotonic, so whichever
    # │ centroid is nearest by squared distance is nearest by distance too — and
    # │ skipping it saves an operation on every pair.
    # │
    # │ Check the shape of what you return. Getting (k,) or (n, k) back instead
    # │ of (n,) is the usual mistake, and it fails confusingly further downstream.
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — the assign step
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write the assign step. "
        "See the YOUR TASK box just above for the steps."
    )


def update_centroids(X, labels, k, centroids):
    """
    Move each centroid to the mean of its assigned points — the UPDATE step.

    :param X:         (n, d) array of points
    :param labels:    (n,) cluster index per point, from assign_clusters
    :param k:         number of clusters
    :param centroids: (k, d) current centroids, needed for the empty-cluster case
    :return:          (k, d) array of new centroids
    """
    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Recompute each centroid as the average of the points assigned to it.
    # │
    # │ For each cluster j in range(k):
    # │   - select its members with a boolean mask:  X[labels == j]
    # │   - the new centroid is their mean along axis=0
    # │
    # │ THE CASE THAT WILL BITE YOU: a cluster can end up with no points at all.
    # │ np.mean of an empty array is nan with a RuntimeWarning, and one nan
    # │ centroid poisons every distance computed against it, so on the next pass
    # │ nothing is assigned to it and it stays broken forever.
    # │
    # │ Decide what should happen instead. Leaving that centroid exactly where it
    # │ was is the simplest defensible choice, which is why `centroids` is passed
    # │ in. Do NOT drop the cluster — that silently changes k, and the caller is
    # │ still expecting k rows back.
    # │
    # │ A loop over k is fine here. k is small, and clarity beats cleverness.
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — the update step
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write the update step. "
        "See the YOUR TASK box just above for the steps."
    )


def inertia(X, labels, centroids):
    """
    Total within-cluster sum of squared distances — the quantity k-means lowers.

    It falls monotonically with k, so it cannot be used to choose k on its own.
    That is what the elbow plot is for.
    """
    return float(np.sum((X - centroids[labels]) ** 2))


# ---------------------------------------------------------------------------
# 3. Fitting
# ---------------------------------------------------------------------------
def fit_kmeans(X, k, rng, max_iter=MAX_ITERATIONS):
    """
    Run Lloyd's algorithm to convergence.

    :return: (centroids, labels, history) where history is the inertia per pass.
    """
    centroids = init_centroids(X, k, rng)
    labels = np.full(len(X), -1)
    history = []

    # ┌─ YOUR TASK ─────────────────────────────────────────────────────────────
    # │ Write the k-means loop.
    # │
    # │ Repeat up to `max_iter` times:
    # │   1. new_labels = assign_clusters(X, centroids)
    # │   2. if new_labels is identical to `labels`, nothing will change again —
    # │      set labels = new_labels, record the inertia, and break
    # │   3. labels = new_labels
    # │   4. centroids = update_centroids(X, labels, k, centroids)
    # │   5. append inertia(X, labels, centroids) to `history`
    # │
    # │ Two things worth noticing:
    # │   - the stopping rule is "the assignments stopped changing", not "the
    # │     inertia is small". There is no target value to reach.
    # │   - assign, THEN update. Doing it the other way round moves centroids
    # │     using last round's assignments and the loop will wander.
    # └──────────────────────────────────────────────────────────────────────────
    # YOUR CODE HERE — the assign / update loop
    # Delete the raise below once you have written it.
    raise NotImplementedError(
        "Homework: write the assign / update loop. "
        "See the YOUR TASK box just above for the steps."
    )

    return centroids, labels, history


# ---------------------------------------------------------------------------
# A WARNING ABOUT THIS ALGORITHM — read this once your loop works
#
# Homeworks 01 and 02 had one right answer and always found it. Run gradient
# descent on that data a hundred times and you get the same coefficients every
# time. K-means is not like that, and the difference is worth understanding.
#
# Try a few seeds:
#
#     uv run 05_kmeans.py --seed 42
#     uv run 05_kmeans.py --seed 31
#
# Most seeds give purity 1.0000. Some do not. Over 500 seeds, 385 came out
# perfect and 61 collapsed badly — so roughly one run in eight fails, which you
# would rarely notice trying three or four by hand. Seed 31 is one of them:
#
#       cluster   size    centroid x   centroid y
#             0     37       -2.5282      -1.1127
#             1    177        2.1948       1.0198
#             2     56       -3.0758      -2.5940
#
#       purity vs the true blobs: 0.6630
#       inertia decreased every pass: yes
#
# Read the sizes. 37 + 56 = 93, and both of those centroids sit on top of the
# blob near (-3, -2): two clusters split ONE real blob down the middle. Cluster
# 1 then holds 177 points — it swallowed the other two blobs whole and parked
# its centre at (2.19, 1.02), empty space where no blob exists.
#
# Now read the last line again: inertia decreased every pass. The algorithm did
# nothing wrong. It went downhill the whole way and stopped when no single point
# could improve by switching clusters. That is a LOCAL OPTIMUM — the bottom of
# the wrong valley. Escaping would mean moving many points at once, and the
# algorithm only ever asks "would this one point be better off elsewhere?" The
# answer is no, so it is stuck, correctly obeying its own rules.
#
# More iterations cannot save it. It has converged. It converged to the wrong
# thing, because init_centroids happened to draw two starting points out of one
# blob and none out of another.
#
# THE USEFUL PART: you can detect this without knowing the true blobs.
#
#     purity 1.00       385 seeds     inertia  322 ..  464
#     purity 0.80-0.99   54 seeds     inertia  347 ..  437
#     purity under 0.80  61 seeds     inertia 1535 .. 2310
#
# A collapsed run has inertia three to five times higher than a good one, and
# inertia needs no labels — it is just the sum of squared distances, which you
# can always compute. So the fix is to run k-means several times from different
# random starts and keep the run with the LOWEST inertia. Best-of-10 restarts
# takes the success rate from 385/500 to about 89/100.
#
# This is not a toy workaround. It is what real implementations do:
# scikit-learn's KMeans runs 10 initialisations by default (n_init=10) for
# exactly this reason.
#
# Two honest limits. Inertia catches collapses, not near misses — the 0.80-0.99
# runs sit inside the good range, so a couple of boundary points in the wrong
# cluster are invisible to it. And restarts are not a guarantee: 11 in 100 still
# missed with ten tries.
#
# The lesson to carry out of this course: some algorithms find the one answer
# every time, and some find a different answer depending on where they started.
# Knowing which kind you are holding is most of what it means to understand a
# method. Nothing in the output of a single bad run says "I failed" — purity
# said it here only because we generated the data and kept the labels to score
# against. On real data you would not have that. You would have inertia, several
# restarts, and your own judgement.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 4. Scoring against the blobs we generated
# ---------------------------------------------------------------------------
def purity(true_ids, labels, k):
    """
    Fraction of points whose cluster's majority true-blob matches their own.

    Cluster ids are arbitrary — the algorithm has no idea our blob 0 is its
    cluster 2 — so accuracy cannot be computed directly. Purity sidesteps that
    by asking, for each cluster, which blob dominates it.
    """
    correct = 0
    for j in range(k):
        members = true_ids[labels == j]
        if len(members):
            correct += np.bincount(members).max()
    return float(correct / len(true_ids))


def report(X, centroids, labels, history, true_ids, k):
    print("=" * 64)
    print(f"  K-MEANS — Lloyd's algorithm, k = {k}")
    print("=" * 64)
    print(f"\n  points        : {len(X)}")
    print(f"  passes run    : {len(history)}  (stopped when assignments settled)")
    print(f"  inertia       : {history[0]:.2f} -> {history[-1]:.2f}")

    print(f"\n  {'cluster':>9}{'size':>7}{'centroid x':>14}{'centroid y':>13}")
    for j in range(k):
        size = int(np.sum(labels == j))
        print(f"  {j:>9}{size:>7}{centroids[j, 0]:>14.4f}{centroids[j, 1]:>13.4f}")

    print("\n  true blob centres used to generate the data:")
    for i, c in enumerate(TRUE_CENTERS):
        print(f"      blob {i}: ({c[0]:+.2f}, {c[1]:+.2f})")

    p = purity(true_ids, labels, k)
    print(f"\n  purity vs the true blobs: {p:.4f}")
    if k == len(TRUE_CENTERS):
        verdict = "PASS" if p > 0.85 else "LOW — likely a bad initialisation, try another seed"
        print(f"  [{verdict}]")
    else:
        print(f"  (k differs from the {len(TRUE_CENTERS)} blobs used, so purity is only indicative)")

    # Inertia must never increase; if it does, assign and update are out of order.
    increases = [i for i in range(1, len(history)) if history[i] > history[i - 1] + 1e-9]
    print(f"  inertia decreased every pass: {'yes' if not increases else 'NO — check loop order'}")
    print("=" * 64)
    return p


def save_plot(X, centroids, labels, history, k, path="05_kmeans_clusters.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot skipped — pip install matplotlib]")
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))
    for j in range(k):
        pts = X[labels == j]
        ax1.scatter(pts[:, 0], pts[:, 1], s=16, alpha=0.6, label=f"cluster {j}")
    ax1.scatter(centroids[:, 0], centroids[:, 1], marker="X", s=200,
                edgecolors="black", linewidths=1.2, c="yellow", label="centroids")
    ax1.scatter(TRUE_CENTERS[:, 0], TRUE_CENTERS[:, 1], marker="+", s=170,
                c="black", linewidths=1.6, label="true centres")
    ax1.set_xlabel("x"); ax1.set_ylabel("y"); ax1.set_title("Clusters found")
    ax1.legend(fontsize=8)

    ax2.plot(range(1, len(history) + 1), history, marker="o", lw=1.6)
    ax2.set_xlabel("pass"); ax2.set_ylabel("inertia")
    ax2.set_title("Inertia — must never go up")

    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)
    print(f"  [plot saved -> {path}]")
    return path


def save_report(centroids, labels, history, k, p, path="05_kmeans_report.md"):
    rows = "\n".join(
        f"| {j} | {int(np.sum(labels == j))} | ({centroids[j,0]:+.3f}, {centroids[j,1]:+.3f}) |"
        for j in range(k)
    )
    text = f"""# Homework 05 — K-Means Clustering

Lloyd's algorithm on {len(labels)} unlabelled points, k = {k}, run until the
assignments stopped changing ({len(history)} passes).

## Clusters found

| cluster | size | centroid |
| --- | --- | --- |
{rows}

## Convergence

| quantity | value |
| --- | --- |
| inertia, first pass | {history[0]:.2f} |
| inertia, final pass | {history[-1]:.2f} |
| passes to convergence | {len(history)} |
| purity vs the true blobs | {p:.4f} |

## Reading the numbers

Inertia falls on every pass and then stops. That is guaranteed: both steps of the
algorithm can only lower it, so it decreases monotonically until the assignments
repeat. Inertia rising is not a tuning problem — it means assign and update ran in
the wrong order.

Purity of {p:.4f} compares the clusters against the blobs the data was generated
from. The algorithm never saw those, and it never sees anything that tells it it
is right — which is what makes this unsupervised.

## Two things to try

Change the seed and re-run. Random initialisation sometimes drops two centroids
into one blob and none into another, and the result is visibly wrong even though
the code is correct. That failure is what k-means++ was invented to avoid.

Then sweep k from 1 to 8 and plot the final inertia against k. It falls every
time, because more clusters always fit better — which is precisely why you cannot
pick k by minimising it.
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  [report saved -> {path}]")
    return path


def main():
    ap = argparse.ArgumentParser(description="K-means clustering from scratch.")
    ap.add_argument("--k", type=int, default=DEFAULT_K, help=f"number of clusters (default {DEFAULT_K})")
    ap.add_argument("--seed", type=int, default=SEED, help="change to see initialisation sensitivity")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if args.k < 1:
        raise SystemExit("--k must be at least 1")

    rng = np.random.default_rng(args.seed)
    X, true_ids = make_data(rng)

    centroids, labels, history = fit_kmeans(X, args.k, rng)

    p = report(X, centroids, labels, history, true_ids, args.k)
    if args.plot:
        save_plot(X, centroids, labels, history, args.k)
    if args.report:
        save_report(centroids, labels, history, args.k, p)


if __name__ == "__main__":
    main()
