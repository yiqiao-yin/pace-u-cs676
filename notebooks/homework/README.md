# Homework — algorithms from scratch

**CS676 Algorithms for Data Science · Pace University**

Five short exercises. Each one is a complete, runnable Python script with the
**core algorithm removed** — the fitting loop and the functions it calls.
Everything around it — the data, the metrics, the printing, the plots — is
written for you, so you can spend your time on the ten or fifteen lines that
actually do the learning.

That is deliberate. Reading `beta = beta - lr * grad` in a slide is not the same as
writing it and watching the loss fall. These exercises put you in the four or five
lines where the learning actually happens.

## Getting started

Three commands, from nothing to a running exercise:

```bash
git clone https://github.com/yiqiao-yin/pace-u-cs676.git
cd pace-u-cs676/notebooks/homework
uv sync
```

Then run any exercise with `uv run`, which uses that environment without you
having to activate anything:

```bash
uv run 01_lr.py
```

Every command in this README is written that way.

### New to uv? Start with `00_uv_tutorial.py`

If `uv run` is unfamiliar, run this before anything else:

```bash
uv run 00_uv_tutorial.py
```

**It is not graded and has no blanks.** It is a short script whose whole job is to
show you what `uv run` is doing. It prints which Python is executing it, which
packages `uv sync` installed, and what your command-line flags became — and then
asks you to change one line and run it again, which is the edit-save-rerun loop you
will use for the rest of the course.

The three commands worth knowing, all of which that script demonstrates:

| Command | What it does |
| --- | --- |
| `uv sync` | Build the environment. Once per clone, not once per script. |
| `uv run 01_lr.py` | Run a script inside that environment. |
| `uv run 01_lr.py --plot` | Anything after the filename is passed to the script, not to uv. |

Two things that trip people up:

- **You never activate anything.** There is no `source .venv/bin/activate` in this
  course. If you have used `venv` before, this is the habit to drop.
- **`ModuleNotFoundError` almost always means a missing `uv run`.** Running
  `python 01_lr.py` uses your system Python, which does not have the packages
  `uv sync` installed. Add the prefix.

Every script here accepts `--help`, which lists its flags:

```bash
uv run 01_lr.py --help
```

**The only thing you need installed first is [uv](https://docs.astral.sh/uv/).**
You do not need to set up Python yourself — this folder declares the version it
wants and uv will fetch a suitable interpreter if you do not already have one.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # macOS / Linux
brew install uv                                     # or, with Homebrew
```

Two things worth knowing, so nothing looks broken:

- **The first `uv sync` downloads about 37 MB** and takes a few seconds to a
  minute depending on your connection. Every run after that is instant, because
  the packages are cached. It needs the network only that first time.
- **You can skip `uv sync` entirely.** `uv run` sets the environment up on demand,
  so running an exercise straight after cloning works too. The explicit sync is
  just a way to get the downloading over with.

### Requirements

Only **numpy**. No scikit-learn, no statsmodels — using them defeats the purpose.
matplotlib comes along too, but only the optional `--plot` flag touches it.

This folder is one uv environment shared by all five exercises and the solutions,
so you sync once and never think about it again.

If you would rather not use uv, plain pip works just as well — the scripts import
nothing but numpy:

```bash
pip install numpy
pip install matplotlib     # optional, only for --plot
python 01_lr.py
```

## The exercises

| Script | Topic | What you write | Blanks | Session |
| --- | --- | --- | --- | --- |
| `00_uv_tutorial.py` | *How to run these scripts* | *nothing — not graded* | *0* | — |
| `01_lr.py` | Linear regression | the MSE gradient, then the descent loop | 2 | 03 |
| `02_logreg.py` | Logistic regression | the sigmoid, then the descent loop | 2 | 04 |
| `03_cv.py` | K-fold cross validation | the fold construction, then the rotation loop | 2 | 05 |
| `04_tree.py` | Decision tree | the exhaustive split search | 1 | 08 |
| `05_kmeans.py` | K-means clustering | the assign step, the update step, then the loop | 3 | 11 |

Do them in order. 02 reuses the loop you write in 01 with one substitution, and
seeing that for yourself is half the lesson.

**Most exercises have more than one blank.** Each script's docstring lists them in
a suggested order — always bottom-up, small helper first, then the loop that calls
it. The script itself will complain about the loop first, because that is what it
reaches soonest; ignore that and start with the helper.

## How to work

Run the script. It stops immediately:

```
$ uv run 01_lr.py
NotImplementedError: Homework: write the gradient descent loop.
                     See the YOUR TASK box just above for the steps.
```

**That traceback is the assignment starting, not the script breaking.** Every
exercise is built to stop at the first blank until you fill it in.

Open the file and find the box:

```
    # ┌─ YOUR TASK ──────────────────────────────────────────
    # │ Write the gradient descent loop.
    # │
    # │ Repeat `n_iter` times:
    # │   1. compute the current predictions
    # │   ...
    # └───────────────────────────────────────────────────────
    # YOUR CODE HERE — the gradient descent loop
    # Delete the raise below once you have written it.
    raise NotImplementedError(...)
```

Delete the `raise`, write the loop, run it again. The box tells you the steps and
warns you about the mistakes people actually make.

**Do not change anything outside the box.** The helper functions are already
correct, and the scripts check your work against a known answer — that check only
means something if you leave it alone.

## How you know you got it right

Every script grades itself. You are not guessing.

| Script | The check |
| --- | --- |
| `01_lr.py` | Compares your gradient descent against the closed-form solution. They should agree to ~6 decimal places, and it prints `PASS`. |
| `02_logreg.py` | Compares accuracy against the majority-class baseline. Beating 0.50 by a wide margin means the loop works. |
| `03_cv.py` | Validation error should exceed training error in most folds. If they are equal, validation rows leaked into training. |
| `04_tree.py` | Test accuracy must beat the majority baseline, and the learned tree is printed so you can read its rules. |
| `05_kmeans.py` | Inertia must fall on every pass. It prints `inertia decreased every pass: yes` — a `NO` means assign and update ran in the wrong order. |

## Options

```bash
uv run 01_lr.py --plot       # save a PNG
uv run 01_lr.py --report     # write a markdown lab report
uv run 03_cv.py --folds 5    # try a different k
uv run 05_kmeans.py --k 4 --seed 42
```

The `--report` flag writes a small markdown file summarising the run. That is the
shape a written submission should take: numbers in a table, then a paragraph saying
what they mean. Generated `.png` and `_report.md` files are gitignored — they are
your output, not repository content.

## Things worth noticing

Each script ends with numbers chosen to provoke a question. A few to look out for:

- **01** — the final MSE settles near 2.25, and no amount of training gets below it.
  Why? What did we put in the data that guarantees that floor?
- **03** — the resubstitution RMSE comes out *below* the noise level used to
  generate the data. That is arithmetically impossible for an honest model. What is
  it actually measuring?
- **04** — training accuracy is 1.0000. Is that a good model or a meaningless number?
- **05** — change `--seed` and re-run a few times. Sometimes the clustering is
  visibly wrong while the code is entirely correct. What does that tell you about
  the algorithm?

Bring answers to these. They are better exam preparation than the code is.

## Submitting

Submit through the [course form](https://airtable.com/appBjNPgdot15ZqO7/pagKL7hfbTouEflS9/form)
like every other assignment. Send the completed `.py` files. If you generated
reports with `--report`, include those too.

## Solutions

**Each solution is published after that homework's deadline has passed — not
before.** The exercise itself is available from day one; the worked answer appears
later, and only for the homework whose deadline is behind us.

So at any moment this folder holds two things:

| | Where | When it is there |
| --- | --- | --- |
| The exercise | [`01_lr.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/01_lr.py) | Always. All five, from the first day of the course. |
| The solution | [`solutions/01_lr_solution.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/solutions/01_lr_solution.py) | Only after that homework's deadline. |

**The deadlines are in [`DEADLINES.md`](../../DEADLINES.md), and that is the only
place they live.** If you want to know whether a solution is out yet, check the
date there. They change from semester to semester, and that one file is what gets
updated.

A solution link above will **404 until it is released** — that is the mechanism
working, not a broken link. Releasing is a manual step the instructor takes after
the deadline, one homework at a time, so expect a short gap between the deadline
passing and the file appearing.

Once released, run it the same way as everything else:

```bash
uv run solutions/01_lr_solution.py
```

Each solution is the exercise with the blanks filled in — **the `YOUR TASK` boxes
are still there**, now sitting directly above the code that answers them. Read the
box, read the answer, then go back to your own attempt and compare. It is meant to
be read next to what you wrote, not instead of it.

### Why they are held back

Asking an AI assistant to write these loops for you takes about ten seconds.
Nobody can stop you, and you will get full marks for that submission.

You will also have skipped the only part of the exercise that was ever going to
help you in the exam or the capstone — where nothing hands you a well-marked box
with the steps in it. The loops here are four to twelve lines each. Write them,
then read the solution and find out whether you were right.

### For the instructor

The solutions are generated, never hand-written:

```bash
uv run make_homework.py
```

That reads `answer/*_ans.py` and writes both the student script and
`solutions/*_solution.py` from the same source, so the two cannot drift apart.

`notebooks/homework/solutions/` is **gitignored permanently**, including for
solutions that are already public. Release is therefore always explicit:

```bash
# release — after the deadline in DEADLINES.md has passed
git add -f notebooks/homework/solutions/01_lr_solution.py
git commit -m "Release the homework 1 solution"

# un-release — e.g. at the start of a new semester
git rm --cached notebooks/homework/solutions/01_lr_solution.py
git commit -m "Withdraw the homework 1 solution"
```

The permanent ignore rule is the safety property: a solution can only ever enter
the repository through `git add -f`, so a stray `git add -A` can never publish one
early. There is no automation and no date check — the release happens when someone
reads `DEADLINES.md` and decides it should.

Two things worth knowing about un-releasing. It removes the file from `main`, so
the link above 404s and fresh clones do not have it, but **the file remains in the
repository history** — anyone who looks at an older commit can still read it.
Genuinely erasing it would mean rewriting history and force-pushing, which breaks
every existing clone and fork and is not worth doing for something that was
deliberately public for a semester. And it only affects the public repo: the
private mirror keeps every solution regardless, so nothing is ever lost.
