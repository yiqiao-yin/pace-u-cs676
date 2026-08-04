"""
make_metrics_figures.py — generate the figures for the Classification Metrics
notes (docs/12_classification_metrics.md).

    python tools/figures/make_metrics_figures.py

Writes:
    pics/12_metrics_01.png   which confusion-matrix cells each metric uses
    pics/12_metrics_02.png   ROC curves with AUC
    pics/12_metrics_03.png   why F1 uses the harmonic mean

Drawn from scratch with matplotlib rather than lifted from a slide or a
textbook, so the course owns them outright and they can be regenerated or
restyled at any time. Requires only numpy and matplotlib.
"""

import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

OUT = pathlib.Path(__file__).resolve().parents[2] / "pics"
SEED = 676

INK = "#1c1c21"      # text and rules
NUM = "#3d3d46"      # cell values
HL = "#7cc4ff"       # numerator cells — the ones the metric counts as "right"
DEN = "#d8ecff"      # denominator-only cells — counted, but not as successes
OFF = "#f2f2f4"      # cells the metric ignores entirely

# A worked example carried through both panels, so the numbers stay comparable.
TP, FP, FN, TN = 139, 12, 11, 138


def confusion_panel(ax, title, formula, numerator, denominator):
    """One 2x2 confusion matrix with the cells a given metric uses shaded."""
    cells = {"TP": (1, 1, TP), "FP": (1, 0, FP), "FN": (0, 1, FN), "TN": (0, 0, TN)}

    for name, (col, row, value) in cells.items():
        if name in numerator:
            face, weight = HL, "bold"
        elif name in denominator:
            face, weight = DEN, "normal"
        else:
            face, weight = OFF, "normal"
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=face,
                               edgecolor=INK, linewidth=1.1))
        ax.text(col + 0.5, row + 0.62, name, ha="center", va="center",
                fontsize=9.5, color=INK, fontweight=weight)
        ax.text(col + 0.5, row + 0.34, str(value), ha="center", va="center",
                fontsize=11, color=NUM, fontweight=weight)

    ax.set_xlim(-0.05, 2.05)
    ax.set_ylim(-0.05, 2.35)
    ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["pred 0", "pred 1"], fontsize=8.5)
    ax.set_yticks([0.5, 1.5]); ax.set_yticklabels(["true 0", "true 1"], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK, pad=16)
    ax.text(1.0, 2.13, formula, ha="center", va="center", fontsize=9, color=NUM)
    ax.set_aspect("equal")


def figure_one():
    """Four metrics, four shadings of the same table."""
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.3))
    acc = (TP + TN) / (TP + FP + FN + TN)
    sens = TP / (TP + FN)
    spec = TN / (TN + FP)
    prec = TP / (TP + FP)

    confusion_panel(axes[0], "Accuracy", f"(TP+TN) / all  =  {acc:.3f}",
                    {"TP", "TN"}, {"FP", "FN"})
    confusion_panel(axes[1], "Sensitivity (recall)", f"TP / (TP+FN)  =  {sens:.3f}",
                    {"TP"}, {"FN"})
    confusion_panel(axes[2], "Specificity", f"TN / (TN+FP)  =  {spec:.3f}",
                    {"TN"}, {"FP"})
    confusion_panel(axes[3], "Precision", f"TP / (TP+FP)  =  {prec:.3f}",
                    {"TP"}, {"FP"})

    fig.suptitle("The same confusion matrix, four questions — shaded cells are the ones each metric counts",
                 fontsize=11.5, color=INK, y=0.99)
    fig.text(0.5, 0.03,
             "Blue = numerator (what the metric treats as success).   "
             "Pale = denominator only.   Grey = ignored entirely.",
             ha="center", fontsize=8.5, color=NUM)
    fig.tight_layout(rect=[0, 0.10, 1, 0.94])   # leave room under the tick labels
    path = OUT / "12_metrics_01.png"
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)
    return path


def roc_from_scores(scores, labels):
    """Sweep the threshold and return (fpr, tpr) — the definition, not a library call."""
    order = np.argsort(-scores)
    labels = labels[order]
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    tpr = np.concatenate([[0], tp / tp[-1]])
    fpr = np.concatenate([[0], fp / fp[-1]])
    return fpr, tpr


def figure_two():
    """ROC curves for three separations of the same two classes."""
    rng = np.random.default_rng(SEED)
    n = 900
    labels = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.9))

    for sep, colour, name in [(2.2, "#1a7f37", "strong"),
                              (1.0, "#7cc4ff", "moderate"),
                              (0.25, "#d1242f", "weak")]:
        scores = np.concatenate([rng.normal(0, 1, n), rng.normal(sep, 1, n)])
        fpr, tpr = roc_from_scores(scores, labels)
        auc = np.trapezoid(tpr, fpr)
        ax1.plot(fpr, tpr, lw=2, color=colour, label=f"{name} signal — AUC {auc:.3f}")

    ax1.plot([0, 1], [0, 1], ls="--", lw=1.3, color=NUM, label="random guessing — AUC 0.500")
    ax1.plot(0, 1, marker="*", ms=15, color=INK, zorder=5)
    ax1.annotate("perfect classifier", xy=(0, 1), xytext=(0.20, 0.90), fontsize=8.5, color=INK,
                 arrowprops=dict(arrowstyle="->", color=INK, lw=0.9))
    ax1.set_xlabel("False positive rate  (1 − specificity)", fontsize=9.5)
    ax1.set_ylabel("True positive rate  (sensitivity)", fontsize=9.5)
    ax1.set_title("ROC — every threshold at once", fontsize=11, fontweight="bold", color=INK)
    ax1.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax1.set_xlim(-0.02, 1.02); ax1.set_ylim(-0.02, 1.04)
    ax1.grid(alpha=0.25, lw=0.6)
    ax1.set_aspect("equal")

    # The two-distribution picture the slides use for the power of a test. Smooth
    # densities rather than histograms, so the two error regions can be shaded
    # exactly — with overlapping histograms the labels end up over the wrong bars.
    sep = 1.0
    x = np.linspace(-4, 5, 700)
    pdf = lambda mu: np.exp(-0.5 * (x - mu) ** 2) / np.sqrt(2 * np.pi)
    neg, pos = pdf(0.0), pdf(sep)
    thr = 0.5

    ax2.plot(x, neg, color="#6b7076", lw=1.8)
    ax2.plot(x, pos, color="#2f7fbf", lw=1.8)
    ax2.fill_between(x, neg, alpha=0.16, color="#6b7076")
    ax2.fill_between(x, pos, alpha=0.16, color=HL)

    # False negatives: actually positive, but scored below the threshold.
    ax2.fill_between(x, pos, where=(x <= thr), color=HL, alpha=0.75)
    # False positives: actually negative, but scored above it.
    ax2.fill_between(x, neg, where=(x >= thr), color="#d1242f", alpha=0.45)

    ax2.axvline(thr, color=INK, lw=1.8)
    ax2.text(thr, 0.455, "decision threshold", ha="center", fontsize=8.5, color=INK)
    ax2.annotate("", xy=(thr + 0.75, 0.435), xytext=(thr - 0.75, 0.435),
                 arrowprops=dict(arrowstyle="<->", color=INK, lw=0.9))
    ax2.text(thr, 0.415, "slide it to move along the ROC", ha="center",
             fontsize=8, color=NUM, style="italic")

    ax2.annotate("false negatives\n(positive, scored low)",
                 xy=(thr - 0.55, 0.055), xytext=(-3.5, 0.20), fontsize=8.5, color="#1b4f72",
                 arrowprops=dict(arrowstyle="->", color="#1b4f72", lw=0.9))
    ax2.annotate("false positives\n(negative, scored high)",
                 xy=(thr + 0.55, 0.055), xytext=(2.2, 0.20), fontsize=8.5, color="#a01722",
                 arrowprops=dict(arrowstyle="->", color="#a01722", lw=0.9))

    ax2.text(-1.5, 0.36, "actually\nnegative", ha="center", fontsize=9, color="#4a4f55")
    ax2.text(2.6, 0.36, "actually\npositive", ha="center", fontsize=9, color="#2f7fbf")

    ax2.set_xlabel("model score", fontsize=9.5)
    ax2.set_ylabel("density", fontsize=9.5)
    ax2.set_ylim(0, 0.50)
    ax2.set_title("Where the curve comes from — one threshold, two error types",
                  fontsize=11, fontweight="bold", color=INK)
    ax2.grid(alpha=0.2, lw=0.6, axis="y")

    fig.tight_layout()
    path = OUT / "12_metrics_02.png"
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)
    return path


def figure_three():
    """Why F1 uses the harmonic mean: the commute, then the whole P-R plane."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.9))

    # --- left: the deck's commute argument, with numbers that divide cleanly ---
    x, y, d = 60.0, 20.0, 30.0          # out mph, back mph, one-way miles
    arith = (x + y) / 2
    harm = 2 * x * y / (x + y)          # == total distance / total time

    ax1.add_patch(Rectangle((0.10, 0.52), 0.80, 0.085, facecolor="#cfe8d4",
                            edgecolor=INK, linewidth=1.0))
    ax1.text(0.50, 0.562, f"distance each way: {d:.0f} miles", ha="center",
             va="center", fontsize=9, color=INK)
    ax1.annotate("", xy=(0.86, 0.70), xytext=(0.14, 0.70),
                 arrowprops=dict(arrowstyle="->", color="#1a7f37", lw=1.6))
    ax1.text(0.50, 0.735, f"out: {x:.0f} mph  →  takes {d / x:.1f} h", ha="center",
             fontsize=9, color="#1a7f37")
    ax1.annotate("", xy=(0.14, 0.42), xytext=(0.86, 0.42),
                 arrowprops=dict(arrowstyle="->", color="#a01722", lw=1.6))
    ax1.text(0.50, 0.355, f"back: {y:.0f} mph  →  takes {d / y:.1f} h", ha="center",
             fontsize=9, color="#a01722")
    # Drawn markers rather than emoji: DejaVu Sans (matplotlib's default, and all
    # anyone is guaranteed to have) has no emoji glyphs, so 🏠 renders as tofu.
    for cx, label in [(0.06, "home"), (0.94, "work")]:
        ax1.add_patch(Rectangle((cx - 0.045, 0.545), 0.09, 0.09, facecolor="white",
                                edgecolor=INK, linewidth=1.2, zorder=4))
        ax1.text(cx, 0.59, label, ha="center", va="center", fontsize=8,
                 color=INK, zorder=5)

    ax1.text(0.50, 0.235,
             f"{2 * d:.0f} miles in {d / x + d / y:.0f} hours  =  {harm:.0f} mph",
             ha="center", fontsize=10.5, color=INK, fontweight="bold")
    ax1.text(0.28, 0.115, f"arithmetic\n({x:.0f}+{y:.0f})/2 = {arith:.0f} mph",
             ha="center", fontsize=9, color="#a01722")
    ax1.text(0.72, 0.115, f"harmonic\n2xy/(x+y) = {harm:.0f} mph",
             ha="center", fontsize=9, color="#1a7f37", fontweight="bold")
    ax1.text(0.50, 0.02, "the mean that answers the question is the harmonic one",
             ha="center", fontsize=8.5, color=NUM, style="italic")

    ax1.set_xlim(0, 1); ax1.set_ylim(0, 0.88); ax1.axis("off")
    ax1.set_title("Averaging rates: there and back", fontsize=11,
                  fontweight="bold", color=INK)

    # --- right: F1 across the whole precision-recall plane ---
    g = np.linspace(0.001, 1, 400)
    P, R = np.meshgrid(g, g)
    f1 = 2 * P * R / (P + R)

    im = ax2.contourf(P, R, f1, levels=np.linspace(0, 1, 21), cmap="Blues")
    cs = ax2.contour(P, R, f1, levels=[0.2, 0.4, 0.6, 0.8], colors=INK,
                     linewidths=0.7, alpha=0.55)
    ax2.clabel(cs, inline=True, fontsize=7.5, fmt="%.1f")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.03, label="$F_1$")

    # One lopsided model and one balanced model with the SAME arithmetic mean.
    # xytext is absolute, not an offset, so neither box can drift off the axes.
    for (p, r, tx, ty) in [(0.98, 0.12, 0.54, 0.15),
                           (0.55, 0.55, 0.30, 0.85)]:
        ax2.plot(p, r, marker="o", ms=8, color="#d1242f", zorder=6,
                 markeredgecolor="white", markeredgewidth=1.2)
        ax2.annotate(f"precision {p:.2f}, recall {r:.2f}\n"
                     f"arithmetic {(p + r) / 2:.2f}   ·   $F_1$ {2 * p * r / (p + r):.2f}",
                     xy=(p, r), xytext=(tx, ty), fontsize=8, color=INK,
                     ha="center", va="center", zorder=7,
                     arrowprops=dict(arrowstyle="->", color=INK, lw=0.9),
                     bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=NUM, lw=0.6, alpha=0.95))

    ax2.set_xlabel("Precision", fontsize=9.5)
    ax2.set_ylabel("Recall", fontsize=9.5)
    ax2.set_title("Same arithmetic mean, very different $F_1$", fontsize=11,
                  fontweight="bold", color=INK)
    ax2.set_aspect("equal")
    ax2.set_xlim(0, 1); ax2.set_ylim(0, 1)

    fig.tight_layout()
    path = OUT / "12_metrics_03.png"
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)
    return path


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for p in (figure_one(), figure_two(), figure_three()):
        print(f"  wrote {p.relative_to(p.parents[1])}  ({p.stat().st_size / 1024:.0f} KB)")
