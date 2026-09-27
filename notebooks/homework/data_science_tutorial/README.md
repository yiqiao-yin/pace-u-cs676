# Data science tutorial

Worked demonstrations of the everyday data science work that sits *around* the
algorithms: loading a real dataset, looking at it properly, and reporting what
you found. Nothing here is graded. There are no blanks and no answer key — these
are examples to read and to run, not exercises to complete.

**This README covers every topic in this folder.** One topic per subfolder, each
holding its own scripts and its own output artifacts (figures, tables). The
commands for all of them are below and they are all the same shape — only the
path changes.

## How this folder is organised

```
data_science_tutorial/
├── README.md                      <- you are here; covers all topics
└── 01_exploratory_data_analysis/  <- one folder per topic
    ├── 01_eda.py                  <- the script
    └── eda_*.png                  <- the artifacts it writes, beside it
```

Each topic folder is self-contained: the script writes its figures next to
itself, so a topic can be read, run, or copied out on its own. Adding a topic
means adding a folder and a section to this README — not a second README.

## Running any of them

All topics share the **one `uv` environment** that covers the rest of
`notebooks/homework/`, so there is nothing per-topic to set up. Sync once:

```bash
cd notebooks/homework
uv sync
```

Then run whichever topic you want. Always run from `notebooks/homework`, not
from inside the topic folder:

```bash
uv run data_science_tutorial/01_exploratory_data_analysis/01_eda.py
```

Artifacts land in the topic's own folder regardless of where you invoked it
from — the scripts resolve paths relative to themselves, not to the working
directory.

## Topics

### 01 — Exploratory data analysis

`01_exploratory_data_analysis/01_eda.py`

```bash
uv run data_science_tutorial/01_exploratory_data_analysis/01_eda.py
```

A first pass over the **Breast Cancer Wisconsin (Diagnostic)** dataset (UCI
id=17): 569 fine-needle-aspirate images, 30 measurements each, labelled
malignant or benign. The point of an EDA pass is not to produce plots; it is to
arrive at a short list of defensible statements about the data, with figures
that support them.

The first run fetches from UCI and caches to `breast_cancer_wisconsin.csv`
inside the topic folder, so every later run works with no network. The seven
figures are rewritten each time and are byte-stable — a rerun produces no
spurious diff.

What it prints:

| Section | Question it answers |
| --- | --- |
| 1 | Shape, dtypes, missing values, duplicates |
| 2 | What the 30 columns actually measure (10 properties × mean/SE/worst) |
| 3 | Class balance, and the accuracy a constant predictor already gets |
| 4 | Redundancy — which features are the same feature measured twice |
| 5 | Signal — which features separate the classes, and which are noise |
| 6 | Effective dimensionality, via PCA done by hand in numpy |
| 7 | What all of that implies for modelling |

What it writes:

| File | Shows |
| --- | --- |
| `eda_01_class_balance.png` | 62.7% benign / 37.3% malignant, and the majority-class baseline |
| `eda_02_distributions_by_class.png` | Per-class histograms of the ten mean measurements |
| `eda_03_correlation_heatmap.png` | The 30×30 correlation matrix, blocked as mean \| SE \| worst |
| `eda_04_correlation_with_target.png` | Point-biserial correlation of every feature with malignancy |
| `eda_05_pairplot_top_features.png` | Scatter matrix of the four strongest features, one per cluster |
| `eda_06_standardised_boxplots.png` | All 30 features z-scored and split by diagnosis, sorted by separation |
| `eda_07_pca.png` | Scree plot and the two-component projection |

## Conventions

These hold across every topic folder, so a new one should follow them.

- **No scikit-learn.** Standardisation, correlations and PCA (via
  `np.linalg.svd`) are a few lines of numpy each. That is deliberate — the rest
  of `notebooks/homework/` is about writing the algorithm rather than calling
  it, and these tutorials keep the same register.
- **Figures follow the repo convention** in `tools/figures/README.md`: white
  background (the course site is dark and puts a white plate behind every
  image), deterministic output, no emoji.
- **Colour is consistent within a topic** — in the EDA topic benign is always
  blue and malignant always red, so any figure can be read without hunting for
  the legend.
- **Scripts write beside themselves.** Paths resolve from `__file__`, never from
  the working directory.
