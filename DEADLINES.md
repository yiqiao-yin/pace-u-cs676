# Deadlines — CS676, Fall 2026

**This file is the only place deadlines live.** If a date appears anywhere else in this
repository or contradicts what is written here, this file wins. Nothing else needs
updating when a date moves.

Everything is due **Friday at 11:59 PM Eastern**, submitted through the
[course form](https://airtable.com/appBjNPgdot15ZqO7/pagKL7hfbTouEflS9/form).

---

## The schedule

| Due (Friday, 11:59 PM EST) | What | Type |
| --- | --- | --- |
| **Sept 11, 2026** | Homework 1 — Linear regression | Pass / Fail |
| **Sept 18, 2026** | Homework 2 — Logistic regression | Pass / Fail |
| **Sept 25, 2026** | **Project 1** — Credibility scoring | Graded, 30% |
| **Oct 9, 2026** | Homework 3 — Cross validation | Pass / Fail |
| **Oct 16, 2026** | Homework 4 — Decision tree | Pass / Fail |
| **Nov 13, 2026** | **Project 2** — PersonaForge | Graded, 30% |
| **Nov 20, 2026** | Homework 5 — K-means clustering | Pass / Fail |
| **Dec 4, 2026** | **Project 3 proposal** | Pass / Fail |
| **Dec 18, 2026** | **Project 3** — Your own AI/ML project | Graded, 30% |

---

## Homework — five assignments, Pass / Fail

The five exercises live in [`notebooks/homework/`](notebooks/homework/). Each is a
runnable numpy script with the core algorithm removed; you write the missing part.

| Homework | Script | Topic |
| --- | --- | --- |
| 1 | [`01_lr.py`](notebooks/homework/01_lr.py) | Linear regression |
| 2 | [`02_logreg.py`](notebooks/homework/02_logreg.py) | Logistic regression |
| 3 | [`03_cv.py`](notebooks/homework/03_cv.py) | K-fold cross validation |
| 4 | [`04_tree.py`](notebooks/homework/04_tree.py) | Decision tree |
| 5 | [`05_kmeans.py`](notebooks/homework/05_kmeans.py) | K-means clustering |

**Graded Pass / Fail.** A submission that runs and produces the expected output passes.
Every script grades itself — it prints `PASS`, or compares your result against a known
answer — so you can tell before you submit.

**All five count. None are dropped.** Together with the per-session submission form,
they make up the 10% homework portion of your grade.

Nothing stops you finishing them early. They are all in the repository now, and each
one only needs numpy.

---

## Projects

**Projects 1 and 2 have exactly one deliverable and one deadline each.** There is no
deliverable 1, 2, 3 to hand in separately — you submit once, on the date above.

**Project 3 is the one exception: it has two submissions.** A short proposal on
**Dec 4**, so I can give you feedback while there is still time to act on it, and the
finished project on **Dec 18**.

What to build is described in full in each project's own README. Read it — it contains
the requirements, the rubric, and the point breakdown:

| Project | Due | What to follow |
| --- | --- | --- |
| **Project 1** — Credibility scoring | **Sept 25, 2026** | [`deliverable/project_1/README.md`](deliverable/project_1/README.md) |
| **Project 2** — PersonaForge | **Nov 13, 2026** | [`deliverable/project_2/README.md`](deliverable/project_2/README.md) |
| **Project 3** — proposal | **Dec 4, 2026** | [Project 3 spec](docs/12_capstone.md#project-3-your-own-aiml-project) |
| **Project 3** — final submission | **Dec 18, 2026** | [Project 3 spec](docs/12_capstone.md#project-3-your-own-aiml-project) |

### The Project 3 proposal

Due **Dec 4**, graded **Pass / Fail**. It is short — a title, what you intend to build,
the approach, and what you expect to end up with. A page is plenty.

It exists so you do not spend three weeks on something unworkable. Submit it, I will
respond, and you will know before you commit. The 100 points for Project 3 are all on
the final submission; the proposal is a gate, not a grade.

### About the "checkpoints" in Projects 1 and 2

The specs for **Projects 1 and 2** break the work into **Checkpoint 1, 2, and 3**.
Those are a **recommended order of work, not separate submissions.** They exist because
building the whole thing in one sitting the night before does not go well, and because
the checkpoints match the order the material is taught.

(Project 3 is different — its two dates above are real submissions, not checkpoints.)

You are free to ignore the sequence entirely and submit everything at once. Nothing is
graded on when you did which part — only on what you hand in by the deadline.

The point splits attached to each checkpoint (25 / 35 / 40 for Project 1, for example)
are how your **single submission** is marked. They tell you where the marks are, not
when things are due.

---

## Grading at a glance

| Component | Weight | Notes |
| --- | --- | --- |
| Homework | **10%** | Per-session submission form **and** the five Pass/Fail exercises. All five count. |
| Project 1 | **30%** | +5% bonus for a Hugging Face deployment |
| Project 2 | **30%** | +5% bonus for a Hugging Face deployment |
| Project 3 | **30%** | Required. Topic is yours. |
| | **100%** | **110% available with both bonuses** |

The letter-grade scale is in the [course README](README.md#letter-grades).

---

## Submitting

### One GitHub repository for the whole course

Before the first deadline, create **one** GitHub repository and use it for everything
you submit in this course. Not one per assignment — **one, all semester.**

**The repository name must contain the course number.** For example:

```
cs676-jane-doe          cs676-fall-2026          CS676-portfolio
```

Any of those is fine. `homework`, `my-project`, or `untitled-3` is not — with a class
of submissions to work through, a name without `cs676` in it is one I cannot place.

**It must be reachable.** Public is simplest. If you would rather keep it private, add
me as a collaborator so I can actually open it — a link I cannot load counts as nothing
submitted.

### Organize it so a stranger can find things

Something like this, and the exact names matter less than the fact that they are
obvious:

```
cs676-jane-doe/
├── README.md          <- what is here, and how to run it
├── homework/
│   ├── 01_lr.py
│   ├── 02_logreg.py
│   ├── 03_cv.py
│   ├── 04_tree.py
│   └── 05_kmeans.py
├── project_1/         <- credibility scoring
├── project_2/         <- personaforge
└── project_3/         <- your own project
```

A top-level `README.md` saying what each folder holds costs you five minutes and makes
everything after it easier to grade.

### Submit the form every week

Use the [course form](https://airtable.com/appBjNPgdot15ZqO7/pagKL7hfbTouEflS9/form) — you can also reach it from the iOS app, ✅ tab.

**Submit it weekly, not only when something is due.** The weekly submission is part of
the 10% homework grade, and it is how I see that you are moving.

**Give the same GitHub URL every time.** The one repository from above, the same link
each week. Do not create a new repository per assignment and do not send a link to a
single file — send the repository, and let the folder structure show what is finished.

Multiple submissions are allowed and **only the last one before the deadline is read**,
so submitting early and updating later costs you nothing.

### Late work

Late work on a project loses 5 points from that project's 100. If something is going
wrong, tell me before the deadline rather than after.
