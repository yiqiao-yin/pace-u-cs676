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

## Requirements

Only **numpy**. No scikit-learn, no statsmodels — using them defeats the purpose.

```bash
pip install numpy
pip install matplotlib     # optional, only for --plot
```

## The exercises

| Script | Topic | What you write | Blanks | Session |
| --- | --- | --- | --- | --- |
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
$ python 01_lr.py
NotImplementedError: Homework: write the gradient descent loop.
                     See the YOUR TASK box just above for the steps.
```

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
python 01_lr.py --plot       # save a PNG (needs matplotlib)
python 01_lr.py --report     # write a markdown lab report
python 03_cv.py --folds 5    # try a different k
python 05_kmeans.py --k 4 --seed 42
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

## A note on solutions

The answer keys are not in this repository, and asking an AI assistant to write the
loop for you takes about ten seconds. Nobody can stop you, and you will get full
marks for that submission.

You will also have skipped the only part of the exercise that was ever going to
help you in the exam or the capstone — where nothing hands you a well-marked box
with the steps in it. The loops here are four to twelve lines each. Write them.
