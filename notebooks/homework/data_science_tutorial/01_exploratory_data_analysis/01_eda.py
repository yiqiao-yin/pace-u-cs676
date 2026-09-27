"""
Exploratory data analysis: Breast Cancer Wisconsin (Diagnostic).

    cd notebooks/homework
    uv sync
    uv run data_science_tutorial/01_exploratory_data_analysis/01_eda.py

Everything this script prints is meant to be read, not skimmed. The point of an
EDA pass is not to produce plots; it is to arrive at a short list of defensible
statements about the data -- how many rows, how many classes, which features
actually separate them, which features are redundant, and where a model is
likely to cheat. The plots exist to support those statements.

The dataset is UCI id=17, 569 fine-needle-aspirate images of breast masses.
Each image was segmented into cell nuclei, ten shape/texture properties were
measured per nucleus, and each property was then summarised three ways across
the nuclei in that image: the mean, the standard error, and the "worst" (the
mean of the three largest values). Ten properties x three summaries = the 30
columns. The label is the diagnosis: M = malignant, B = benign.

That structure is the single most important fact about this dataset, and it is
invisible in the raw column names (`radius1`, `radius2`, `radius3`). Almost
every surprise below -- the block pattern in the correlation matrix, the
near-duplicate features, the fact that a two-component projection nearly
separates the classes -- follows from it.

Figures are written next to this file as PNGs. Nothing here uses scikit-learn:
the standardisation, the correlations, and the PCA are all a few lines of numpy,
which is the point.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files, never try to open a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

HERE = Path(__file__).resolve().parent
CACHE = HERE / "breast_cancer_wisconsin.csv"

# The three suffixes UCI uses for the three summaries of each measurement.
SUFFIX_MEANING = {"1": "mean", "2": "SE", "3": "worst"}

# The ten underlying nuclear measurements, in the order UCI lists them.
PROPERTIES = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]

# What each of the ten actually measures. UCI ships no per-variable
# descriptions, so these come from the paper the dataset accompanies.
PROPERTY_NOTES = {
    "radius": "mean distance from the nucleus centre to its boundary",
    "texture": "standard deviation of the grey-scale values inside the nucleus",
    "perimeter": "length of the traced nuclear boundary",
    "area": "number of pixels inside the boundary",
    "smoothness": "local variation in radius length along the boundary",
    "compactness": "perimeter^2 / area - 1.0; high when the outline is ragged",
    "concavity": "severity of the inward dents in the boundary",
    "concave_points": "how many inward dents there are (not how deep)",
    "symmetry": "difference between the two halves across the longest chord",
    "fractal_dimension": "coastline approximation - 1; boundary roughness across scales",
}

# Colours used for the two classes everywhere in this script. Benign is the
# cool colour and malignant the warm one, consistently, so that any figure can
# be read without hunting for the legend.
C_BENIGN = "#3b6ea5"
C_MALIGNANT = "#c1442e"

PLOT_STYLE = {
    # The course site is dark and puts a white plate behind every image, so
    # figures are drawn on white to match. See tools/figures/README.md.
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def rule(title: str) -> None:
    """Print a section header wide enough to find when scrolling back."""
    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)


def wrap(text: str, indent: str = "  ") -> None:
    """Print a paragraph at a readable width. EDA output is prose too."""
    print(textwrap.fill(text, width=78, initial_indent=indent, subsequent_indent=indent))


def pretty(column: str) -> str:
    """`concave_points3` -> `concave points (worst)`."""
    base, suffix = column[:-1], column[-1]
    return f"{base.replace('_', ' ')} ({SUFFIX_MEANING[suffix]})"


def save(fig: plt.Figure, name: str) -> None:
    path = HERE / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(HERE.parent)}")


def load() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, dict]:
    """Fetch the dataset, caching the merged frame next to this script.

    The cache is not an optimisation -- it is so the script still runs on a
    laptop with no network, which is how half of a class will meet it.
    """
    if CACHE.exists():
        print(f"  reading cached copy: {CACHE.name}")
        frame = pd.read_csv(CACHE)
        X = frame.drop(columns=["Diagnosis"])
        y = frame["Diagnosis"]
        # `variables` is only needed for the printed metadata, so when we are
        # offline we reconstruct the part of it we actually use.
        variables = pd.DataFrame(
            {"name": X.columns, "role": "Feature", "type": "Continuous"}
        )
        return X, y, variables, {"name": "Breast Cancer Wisconsin (Diagnostic)"}

    print("  fetching UCI id=17 over the network ...")
    ds = fetch_ucirepo(id=17)
    X = ds.data.features
    y = ds.data.targets.iloc[:, 0]  # single-column frame -> Series
    pd.concat([X, y.rename("Diagnosis")], axis=1).to_csv(CACHE, index=False)
    print(f"  cached to {CACHE.name} for offline reruns")
    return X, y, ds.variables, ds.metadata


def standardise(X: pd.DataFrame) -> np.ndarray:
    """Z-score each column. Written out because it is three lines of numpy."""
    A = X.to_numpy(dtype=float)
    return (A - A.mean(axis=0)) / A.std(axis=0)


# ---------------------------------------------------------------------------
# 1. shape, dtypes, missingness
# ---------------------------------------------------------------------------


def describe_shape(X: pd.DataFrame, y: pd.Series, metadata: dict) -> None:
    rule("1. What did we just load?")
    wrap(f"Dataset: {metadata.get('name', 'unknown')}")
    print()
    print(f"  rows (patients/images) : {len(X)}")
    print(f"  feature columns        : {X.shape[1]}")
    print(f"  target                 : Diagnosis, {y.nunique()} classes")
    print(f"  dtypes                 : {sorted({str(d) for d in X.dtypes})}")
    print(f"  missing cells          : {int(X.isna().sum().sum())}")
    print(f"  duplicate rows         : {int(X.duplicated().sum())}")
    print()
    wrap(
        "569 rows and 30 columns is a small, wide, clean table: no missing "
        "values, no duplicates, every column continuous. That combination is "
        "rare in practice and it changes what the rest of the analysis has to "
        "worry about. There is no imputation to do and no leakage from an ID "
        "column (UCI already split the ID off into its own field, which we do "
        "not load). What 569-by-30 does mean is that a flexible model can "
        "memorise this table easily, so every number we report later has to "
        "come from held-out data or cross-validation -- see 03_cv.py."
    )


def explain_features(X: pd.DataFrame, variables: pd.DataFrame) -> None:
    rule("2. What are the 30 features, really?")
    wrap(
        "The column names hide the structure. `radius1`, `radius2`, `radius3` "
        "are not three different measurements -- they are one measurement "
        "(nuclear radius) summarised three ways across the nuclei in the "
        "image: the mean, the standard error of that mean, and the average of "
        "the three largest values, which the authors call 'worst'. So the 30 "
        "columns are a 10 x 3 grid:"
    )
    print()
    print(f"  {'measurement':<20}{'what it captures'}")
    print(f"  {'-' * 20}{'-' * 54}")
    for prop in PROPERTIES:
        print(f"  {prop.replace('_', ' '):<20}{PROPERTY_NOTES[prop]}")
    print()
    wrap(
        "x three summaries: 1 = mean, 2 = standard error, 3 = worst. "
        "Reading the names this way immediately predicts two things we will "
        "confirm below. First, radius, perimeter and area are three ways of "
        "measuring the same size, so they must be almost perfectly "
        "correlated. Second, the 'worst' columns should be the most "
        "informative: a mass is malignant if some nuclei look bad, not if the "
        "average nucleus does, and averaging over a whole image washes that "
        "out."
    )
    print()
    roles = variables["role"].value_counts().to_dict() if "role" in variables else {}
    if roles:
        print(f"  UCI variable roles: {roles}")

    print()
    print("  Scale varies by orders of magnitude across columns:")
    print()
    summary = X.describe().T[["mean", "std", "min", "max"]]
    widest = summary.reindex(summary["std"].sort_values(ascending=False).index)
    print(f"  {'column':<26}{'mean':>12}{'std':>12}{'min':>12}{'max':>12}")
    for name in list(widest.index[:3]) + ["..."] + list(widest.index[-3:]):
        if name == "...":
            print(f"  {'...':<26}")
            continue
        r = widest.loc[name]
        print(
            f"  {pretty(name):<26}{r['mean']:>12.4g}{r['std']:>12.4g}"
            f"{r['min']:>12.4g}{r['max']:>12.4g}"
        )
    print()
    wrap(
        "`area (worst)` has a standard deviation near 570; "
        "`fractal dimension (SE)` near 0.003 -- five orders of magnitude "
        "apart. Nothing is wrong, they are different physical units. But it "
        "means any method that measures distance or sums squared coefficients "
        "-- k-means, k-NN, PCA, ridge/lasso, gradient descent on a logistic "
        "loss -- will be dominated by `area` unless the columns are "
        "standardised first. Every figure below that combines columns "
        "standardises them."
    )


def describe_target(y: pd.Series) -> None:
    rule("3. The target, and the baseline you have to beat")
    counts = y.value_counts()
    total = len(y)
    for label, n in counts.items():
        word = "malignant" if label == "M" else "benign"
        print(f"  {label} ({word:<9}) : {n:>4}  ({n / total:6.1%})")
    majority = counts.max() / total
    print()
    wrap(
        f"Mildly imbalanced, {majority:.1%} / {1 - majority:.1%}. Two "
        "consequences. First, always-predict-benign already scores "
        f"{majority:.1%} accuracy, so an accuracy below that is worse than a "
        "constant, and an accuracy of 90% is a smaller achievement than it "
        "sounds. That is the baseline the homework scripts compare against. "
        "Second, accuracy is the wrong headline metric here anyway: the two "
        "errors are not equally bad. Calling a malignant mass benign sends a "
        "patient home; calling a benign mass malignant sends them for a "
        "biopsy they did not need. Recall on M, and the precision/recall "
        "trade-off behind it, is what a clinician would ask about -- see "
        "docs/12."
    )


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def fig_class_balance(y: pd.Series) -> None:
    counts = y.value_counts().reindex(["B", "M"])
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    bars = ax.bar(
        ["Benign (B)", "Malignant (M)"],
        counts.to_numpy(),
        color=[C_BENIGN, C_MALIGNANT],
        width=0.6,
    )
    for bar, n in zip(bars, counts.to_numpy()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            n + 6,
            f"{n}  ({n / len(y):.1%})",
            ha="center",
            fontsize=9,
        )
    ax.axhline(
        counts.max(),
        color="grey",
        ls="--",
        lw=1,
        label=f"majority-class baseline = {counts.max() / len(y):.1%}",
    )
    ax.set_ylim(0, counts.max() * 1.22)
    ax.set_ylabel("images")
    ax.set_title("Class balance, and the accuracy a constant predictor gets")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(axis="x", visible=False)
    save(fig, "eda_01_class_balance.png")


def fig_distributions(X: pd.DataFrame, y: pd.Series) -> None:
    """Per-class histograms for the ten `mean` columns.

    The first thing to look at, because it answers 'is there any signal at
    all?' one feature at a time, with no modelling assumptions.
    """
    mask_m = (y == "M").to_numpy()
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    for ax, prop in zip(axes.ravel(), PROPERTIES):
        col = X[f"{prop}1"].to_numpy(dtype=float)
        bins = np.histogram_bin_edges(col, bins=30)
        ax.hist(col[~mask_m], bins=bins, color=C_BENIGN, alpha=0.65, label="benign")
        ax.hist(col[mask_m], bins=bins, color=C_MALIGNANT, alpha=0.65, label="malignant")
        ax.set_title(prop.replace("_", " "))
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(
        "Distribution of each mean measurement by diagnosis "
        "(overlap = how little that feature alone can tell you)",
        y=1.02,
    )
    fig.tight_layout()
    save(fig, "eda_02_distributions_by_class.png")


def fig_correlation(X: pd.DataFrame) -> pd.DataFrame:
    """The 30x30 correlation matrix, ordered to make its block structure visible."""
    order = [f"{p}{s}" for s in "123" for p in PROPERTIES]
    corr = X[order].corr()

    fig, ax = plt.subplots(figsize=(11, 9.5))
    im = ax.imshow(corr.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    labels = [pretty(c) for c in order]
    ax.set_xticks(range(len(order)), labels, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), labels, fontsize=7)
    for boundary in (9.5, 19.5):  # the mean | SE | worst blocks
        ax.axhline(boundary, color="black", lw=1.1)
        ax.axvline(boundary, color="black", lw=1.1)
    ax.grid(visible=False)
    fig.colorbar(im, ax=ax, shrink=0.72, label="Pearson r")
    ax.set_title(
        "Feature correlation, grouped as mean | SE | worst\n"
        "dark red blocks are features measuring the same thing twice",
        fontsize=10,
    )
    fig.tight_layout()
    save(fig, "eda_03_correlation_heatmap.png")
    return corr


def fig_target_correlation(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Point-biserial correlation of every feature with the label.

    With a 0/1 target, Pearson r is the point-biserial correlation, so this is
    a one-liner and a perfectly respectable first ranking of features.
    """
    target = (y == "M").astype(float)
    r = X.apply(lambda col: col.corr(target)).sort_values()

    fig, ax = plt.subplots(figsize=(7.5, 8.2))
    colors = [C_MALIGNANT if v > 0 else C_BENIGN for v in r]
    ax.barh([pretty(c) for c in r.index], r.to_numpy(), color=colors)
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("point-biserial correlation with malignancy")
    ax.set_title("Which single features separate the classes?", fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "eda_04_correlation_with_target.png")
    return r


def fig_pairplot(X: pd.DataFrame, y: pd.Series, top: list[str]) -> None:
    """Scatter matrix of the four strongest features, coloured by class."""
    mask_m = (y == "M").to_numpy()
    n = len(top)
    fig, axes = plt.subplots(n, n, figsize=(10.5, 10))
    for i, ci in enumerate(top):
        for j, cj in enumerate(top):
            ax = axes[i, j]
            if i == j:
                col = X[ci].to_numpy(dtype=float)
                bins = np.histogram_bin_edges(col, bins=25)
                ax.hist(col[~mask_m], bins=bins, color=C_BENIGN, alpha=0.65)
                ax.hist(col[mask_m], bins=bins, color=C_MALIGNANT, alpha=0.65)
                ax.set_yticks([])
            else:
                ax.scatter(
                    X[cj][~mask_m], X[ci][~mask_m], s=7, c=C_BENIGN, alpha=0.55, lw=0
                )
                ax.scatter(
                    X[cj][mask_m], X[ci][mask_m], s=7, c=C_MALIGNANT, alpha=0.55, lw=0
                )
            if i == n - 1:
                ax.set_xlabel(pretty(cj), fontsize=8)
            else:
                ax.set_xticklabels([])
            if j == 0:
                ax.set_ylabel(pretty(ci), fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
    fig.suptitle(
        "The four strongest features against each other "
        "(blue benign, red malignant)",
        y=0.995,
    )
    fig.tight_layout()
    save(fig, "eda_05_pairplot_top_features.png")


def fig_standardised_boxplots(X: pd.DataFrame, y: pd.Series, order: list[str]) -> None:
    """Z-scored per-class boxplots: one picture of all 30 features at once.

    Standardising is what makes the comparison legible -- on raw units, `area`
    would be the only visible box.
    """
    Z = pd.DataFrame(standardise(X), columns=X.columns)
    mask_m = (y == "M").to_numpy()
    fig, ax = plt.subplots(figsize=(13, 6))
    positions = np.arange(len(order))
    for data, offset, color, label in (
        (Z[~mask_m], -0.19, C_BENIGN, "benign"),
        (Z[mask_m], 0.19, C_MALIGNANT, "malignant"),
    ):
        bp = ax.boxplot(
            [data[c].to_numpy() for c in order],
            positions=positions + offset,
            widths=0.32,
            patch_artist=True,
            showfliers=False,
        )
        for box in bp["boxes"]:
            box.set(facecolor=color, alpha=0.75, linewidth=0.7)
        for part in ("whiskers", "caps", "medians"):
            for line in bp[part]:
                line.set(color="black", linewidth=0.7)
        bp["boxes"][0].set_label(label)
    ax.set_xticks(positions, [pretty(c) for c in order], rotation=90, fontsize=7)
    ax.axhline(0, color="grey", lw=0.8, ls="--")
    ax.set_ylabel("standardised value (z-score)")
    ax.set_title(
        "All 30 features on one scale, split by diagnosis "
        "(sorted by separation; gap between the pair of boxes = signal)",
        fontsize=10,
    )
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "eda_06_standardised_boxplots.png")


def fig_pca(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """PCA by SVD on the standardised matrix -- about five lines of numpy.

    Two jobs here: show how much of the 30-dimensional variance is really only
    a few dimensions (because the features are so redundant), and show whether
    the classes are separable before any classifier is fitted.
    """
    Z = standardise(X)
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    explained = S**2 / np.sum(S**2)
    scores = U * S  # projection of each row onto the components

    # A component's sign is arbitrary -- SVD is free to return v or -v. Pin it
    # so a rerun gives the same picture, and orient PC1 so that it increases
    # with nuclear size, which is what the caption claims it means.
    size = X["area1"].to_numpy(dtype=float)
    if np.corrcoef(scores[:, 0], size)[0, 1] < 0:
        scores[:, 0] *= -1
        Vt[0] *= -1
    if scores[:, 1].sum() < 0:
        scores[:, 1] *= -1
        Vt[1] *= -1

    mask_m = (y == "M").to_numpy()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))

    ax1.bar(
        np.arange(1, len(explained) + 1), explained * 100, color=C_BENIGN, alpha=0.8
    )
    ax1.plot(
        np.arange(1, len(explained) + 1),
        np.cumsum(explained) * 100,
        color=C_MALIGNANT,
        marker="o",
        ms=3.5,
        lw=1.2,
        label="cumulative",
    )
    ax1.axhline(95, color="grey", ls="--", lw=0.9)
    ax1.text(len(explained), 96, "95%", ha="right", fontsize=8, color="grey")
    ax1.set_xlabel("principal component")
    ax1.set_ylabel("variance explained (%)")
    ax1.set_title("30 correlated features are not 30 dimensions", fontsize=10)
    ax1.legend(fontsize=8, frameon=False)

    ax2.scatter(
        scores[~mask_m, 0], scores[~mask_m, 1], s=14, c=C_BENIGN, alpha=0.7, lw=0,
        label="benign",
    )
    ax2.scatter(
        scores[mask_m, 0], scores[mask_m, 1], s=14, c=C_MALIGNANT, alpha=0.7, lw=0,
        label="malignant",
    )
    ax2.set_xlabel(f"PC1 ({explained[0]:.1%} of variance)")
    ax2.set_ylabel(f"PC2 ({explained[1]:.1%})")
    ax2.set_title("The classes are nearly separable in two dimensions", fontsize=10)
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    save(fig, "eda_07_pca.png")
    return explained


# ---------------------------------------------------------------------------
# read the numbers back out
# ---------------------------------------------------------------------------


def report_correlation(corr: pd.DataFrame) -> None:
    rule("4. Redundancy: which features are the same feature?")
    tri = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    # `.stack()` keeps the NaN half of the matrix in pandas 3, and those NaNs
    # would show up in the pair count below, so drop them explicitly.
    pairs = tri.stack().dropna().sort_values(ascending=False)
    print("  Most correlated pairs:")
    print()
    for (a, b), r in pairs.head(8).items():
        print(f"    r = {r:.3f}   {pretty(a):<26} vs  {pretty(b)}")
    n_high = int((pairs.abs() > 0.9).sum())
    print()
    wrap(
        f"{n_high} of the {len(pairs)} feature pairs correlate above 0.9. The "
        "top of that list is exactly what the feature names predicted: "
        "radius, perimeter and area are one measurement of size recorded in "
        "three units, and r is above 0.99 -- they carry no independent "
        "information at all. Compactness, concavity and concave points form a "
        "second such cluster (boundary irregularity), and each measurement "
        "correlates with its own 'worst' version by construction."
    )
    print()
    wrap(
        "This matters differently for different models. A tree does not care: "
        "it picks one of the duplicates and splits on it (04_tree.py). Ridge, "
        "lasso and plain logistic regression care a great deal -- with "
        "near-collinear columns the individual coefficients become unstable "
        "and large with opposite signs, so you can read a coefficient as "
        "'importance' and be badly wrong, even while the predictions are "
        "fine. If you want interpretable coefficients here, drop to one "
        "column per cluster first."
    )


def report_target_correlation(r: pd.Series) -> list[str]:
    rule("5. Signal: which features separate the classes?")
    ranked = r.abs().sort_values(ascending=False)
    print("  Strongest ten (by |point-biserial r| with malignancy):")
    print()
    for name in ranked.index[:10]:
        print(f"    {r[name]:+.3f}   {pretty(name)}")
    print()
    print("  Weakest five:")
    print()
    for name in ranked.index[-5:]:
        print(f"    {r[name]:+.3f}   {pretty(name)}")
    print()
    by_summary = {
        meaning: float(ranked[[c for c in r.index if c.endswith(s)]].mean())
        for s, meaning in SUFFIX_MEANING.items()
    }
    print("  Mean |r| by summary type:")
    for meaning, value in by_summary.items():
        print(f"    {meaning:<6} {value:.3f}")
    print()
    wrap(
        "Two readings. First, the direction is one-sided: every strong "
        "correlation is positive, meaning malignant nuclei are bigger, more "
        "irregular and more concave -- no strong feature points the other "
        "way. Second, the 'worst' columns beat the 'mean' columns on average "
        f"({by_summary['worst']:.3f} vs {by_summary['mean']:.3f}) and the SE "
        f"columns are nearly useless ({by_summary['SE']:.3f}). That is the "
        "biology showing through the summary statistics: malignancy is "
        "indicated by the presence of some badly-behaved nuclei, so the "
        "extreme summary carries the signal and the average dilutes it. "
        "Symmetry, smoothness and fractal dimension barely move at all, and "
        "the bottom of the table is almost entirely SE columns."
    )
    print()
    wrap(
        "Caveat worth stating out loud: this ranking scores each feature "
        "alone, so it rewards duplicates (all three of radius/perimeter/area "
        "will rank highly together) and punishes features that are only "
        "useful in combination with another. It is a starting point for "
        "picking what to plot, not a feature-selection method."
    )
    # Pick the top four from *distinct* measurement clusters, so the pair plot
    # does not show the same axis four times.
    chosen: list[str] = []
    used: set[str] = set()
    size_cluster = {"radius", "perimeter", "area"}
    shape_cluster = {"compactness", "concavity", "concave_points"}
    for name in ranked.index:
        base = name[:-1]
        cluster = (
            "size" if base in size_cluster else "shape" if base in shape_cluster else base
        )
        if cluster in used:
            continue
        used.add(cluster)
        chosen.append(name)
        if len(chosen) == 4:
            break
    print()
    print(f"  Chosen for the pair plot (one per cluster): {[pretty(c) for c in chosen]}")
    return chosen


def report_pca(explained: np.ndarray) -> None:
    rule("6. Effective dimensionality")
    cum = np.cumsum(explained)
    for k in (1, 2, 3, 5, 10):
        print(f"  first {k:>2} components : {cum[k - 1]:6.1%} of total variance")
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    print()
    wrap(
        f"{cum[0]:.0%} of the variance lives on one axis and {cum[1]:.0%} on "
        f"two; {n95} of 30 components reach 95%. That is the redundancy from "
        "section 4 restated as a number -- the table is 30 columns wide but "
        "only a handful of directions wide. PC1 is essentially 'how big and "
        "irregular are these nuclei', which is also almost exactly the "
        "direction that separates the diagnoses, which is why the right-hand "
        "panel of the PCA figure splits so cleanly with no label information "
        "used in fitting it."
    )
    print()
    wrap(
        "Do not read that separation as an accuracy estimate. PCA was fitted "
        "on all 569 rows, the split is eyeballed rather than cross-validated, "
        "and 'nearly separable' is doing real work in that sentence -- the "
        "classes overlap in a band through the middle, and those are exactly "
        "the cases a clinician would find hard too."
    )


def closing_notes() -> None:
    rule("7. What to take into modelling")
    for i, note in enumerate(
        [
            "Standardise. Column scales span five orders of magnitude, so any "
            "distance- or penalty-based method is meaningless without it. Fit the "
            "scaler on the training fold only, or you have leaked test statistics "
            "into training.",
            "Beat 62.7%, not 50%. That is the always-benign baseline, and it is "
            "what the homework scripts compare against.",
            "Report recall on malignant alongside accuracy. The two error types "
            "have very different costs and a single accuracy number hides which "
            "one the model is making.",
            "Expect collinearity to scramble coefficients. Predictions stay fine; "
            "individual coefficients stop being interpretable. Keep one column per "
            "measurement cluster if interpretation matters.",
            "Prefer the 'worst' columns if you are trimming features, and drop the "
            "SE columns first -- they carry the least signal here.",
            "Use stratified cross-validation. 569 rows with a 63/37 split means an "
            "unstratified fold can drift several points in class ratio, which shows "
            "up as fold-to-fold noise you will mistake for model variance.",
        ],
        start=1,
    ):
        wrap(f"{i}. {note}", indent="  ")
        print()


# ---------------------------------------------------------------------------


def main() -> None:
    plt.rcParams.update(PLOT_STYLE)

    rule("0. Loading")
    X, y, variables, metadata = load()

    describe_shape(X, y, metadata)
    explain_features(X, variables)
    describe_target(y)

    rule("Figures")
    fig_class_balance(y)
    fig_distributions(X, y)
    corr = fig_correlation(X)
    r = fig_target_correlation(X, y)
    top = r.abs().sort_values(ascending=False).index.tolist()

    report_correlation(corr)
    chosen = report_target_correlation(r)

    rule("Figures (continued)")
    fig_pairplot(X, y, chosen)
    fig_standardised_boxplots(X, y, top)
    explained = fig_pca(X, y)

    report_pca(explained)
    closing_notes()

    print("=" * 78)
    print(f"  Seven figures written to {HERE.name}/")
    print("=" * 78)


if __name__ == "__main__":
    main()
