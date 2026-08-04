---
sidebar_position: 12
title: "Classification Metrics"
sidebar_label: "12. Classification Metrics"
---

# Classification Metrics

:::note Draft

These notes were scaffolded from Chapter 12 of the course slides and have not yet
been reviewed. The definitions and formulas are standard, but the worked examples,
the dataset references, and the emphasis are placeholders — replace them with the
ones you use in lecture. Figures are noted where the slides have them; none have
been added to `pics/` yet.

:::

## Table of Contents

- [Why a separate chapter on metrics](#why-a-separate-chapter-on-metrics)
- [The confusion matrix](#the-confusion-matrix)
- [Accuracy, and why it misleads](#accuracy-and-why-it-misleads)
- [Sensitivity and specificity](#sensitivity-and-specificity)
- [Precision and recall](#precision-and-recall)
- [The F1 score](#the-f1-score)
- [The ROC curve](#the-roc-curve)
- [Connection to hypothesis testing](#connection-to-hypothesis-testing)
- [Choosing a metric](#choosing-a-metric)
- [Homework](#homework)

## Why a separate chapter on metrics
[Go back to TOC](#table-of-contents)

Loss functions train a model. Metrics tell you whether the trained model is any
good, and they are not the same thing.

We have used several losses already: least squares and mean squared error for
regression, and the multinomial likelihood and cross-entropy for classification.
Each is chosen because it is differentiable and well-behaved under gradient
descent — properties that matter to the optimiser, not to the person deciding
whether to deploy the model.

The quantities in this chapter are the opposite. Most are not differentiable, so you
cannot train on them directly. They exist to answer a different question: *given a
model that has already been fitted, what kinds of mistakes does it make, and do those
mistakes matter here?*

## The confusion matrix
[Go back to TOC](#table-of-contents)

Everything in this chapter is computed from one table. For a binary problem with an
actual label and a predicted label, every observation lands in exactly one of four
cells:

|  | Predicted positive | Predicted negative |
| --- | --- | --- |
| **Actually positive** | True Positive (TP) | False Negative (FN) |
| **Actually negative** | False Positive (FP) | True Negative (TN) |

The naming is worth pinning down, because it trips people up: the second word is
what the model *said*, and the first word is whether it was *right*. A false
negative is a case the model called negative and was wrong about.

$$
n = \mathrm{TP} + \mathrm{FP} + \mathrm{FN} + \mathrm{TN}
$$

> **Slides:** the deck introduces this as the "confusion table" and builds each
> subsequent metric by highlighting different cells of it. A figure per metric
> would work well here.

## Accuracy, and why it misleads
[Go back to TOC](#table-of-contents)

The first and most obvious metric — the fraction of predictions that were correct:

$$
\text{Accuracy} = \frac{\mathrm{TP} + \mathrm{TN}}{\mathrm{TP} + \mathrm{FP} + \mathrm{FN} + \mathrm{TN}}
$$

Accuracy is the metric people reach for first and the one that misleads most often,
because it treats every error as equally costly and is dominated by whichever class
is larger.

Consider a screening test for a condition affecting 1 in 1000 people. A model that
predicts "negative" for everyone, always, without looking at the data, achieves
**99.9% accuracy** — and is completely useless. It has never once identified a case.

This is why the homework harnesses in this course always print a **majority-class
baseline** next to accuracy. If your model cannot beat the score obtained by ignoring
the features entirely, its accuracy figure is telling you nothing.

## Sensitivity and specificity
[Go back to TOC](#table-of-contents)

These split the question by actual class, which is what makes them useful on
imbalanced data.

**Sensitivity**, also called the true positive rate or recall — of the cases that
really are positive, what fraction did we catch?

$$
\text{Sensitivity} = \text{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
$$

**Specificity** — of the cases that really are negative, what fraction did we
correctly leave alone?

$$
\text{Specificity} = \frac{\mathrm{TN}}{\mathrm{TN} + \mathrm{FP}}
$$

**False positive rate** is what is left over, and is the quantity plotted on the
horizontal axis of the ROC curve:

$$
\text{FPR} = 1 - \text{Specificity} = \frac{\mathrm{FP}}{\mathrm{TN} + \mathrm{FP}}
$$

The two trade against each other, and the thing doing the trading is the **decision
threshold**. A classifier does not output a class; it outputs a score. Calling
everything above 0.5 positive is a choice, not a fact, and moving that number moves
you along the trade-off.

## Precision and recall
[Go back to TOC](#table-of-contents)

Precision asks a different question from sensitivity — it conditions on what the
model *said* rather than on the truth:

$$
\text{Precision} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FP}}
\qquad
\text{Recall} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}}
$$

- **Precision**: when the model says positive, how often is it right?
- **Recall**: of everything that was positive, how much did it find?

Which one you care about depends entirely on which error is more expensive, and that
is a domain question rather than a statistical one. A spam filter that marks a real
invoice as spam has failed worse than one that lets some spam through, so precision
dominates. A cancer screen that misses a tumour has failed worse than one that flags
a healthy patient for a second test, so recall dominates.

You can always make one of them perfect. Predict positive for everything and recall
is 1.0. Predict positive only for the single case you are most certain about and
precision is likely 1.0. Neither model is useful, which is why they are reported as a
pair.

## The F1 score
[Go back to TOC](#table-of-contents)

F1 collapses the pair into one number — the **harmonic** mean of precision and
recall:

$$
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
$$

The harmonic mean is the point. An arithmetic mean of precision 1.0 and recall 0.0
gives 0.5, which flatters a model that never finds anything. The harmonic mean gives
**0**. It is only high when *both* components are high, which is exactly the
behaviour you want from a summary score.

> **To add:** the deck spends several slides on F1 (pages 323–330). Whatever worked
> example is used there should be reproduced here.

The general form weights recall $\beta$ times as heavily as precision:

$$
F_\beta = (1 + \beta^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{\beta^2 \cdot \text{Precision} + \text{Recall}}
$$

$F_2$ favours recall, $F_{0.5}$ favours precision, and $F_1$ treats them equally.

## The ROC curve
[Go back to TOC](#table-of-contents)

The **Receiver Operating Characteristic** curve plots true positive rate against
false positive rate as the decision threshold sweeps from one extreme to the other.

The name is a historical accident worth keeping. ROC analysis was developed during
the Second World War for reading radar signals — after Pearl Harbor, operators had to
decide whether a blip on a screen was an aircraft or noise, and the discipline needed
a way to describe how well an operator distinguished the two. The vocabulary of
"receiver" and "operating characteristic" comes directly from that work, and only
later moved into signal detection theory, medicine, and machine learning.

Reading the curve:

- The **vertical axis** is the true positive rate: it rises as more real positives are caught.
- The **horizontal axis** is the false positive rate, $1 - \text{specificity}$: it rises as more negatives are wrongly flagged.
- The **diagonal** is random guessing. A curve on the diagonal means the score carries no information.
- The **top-left corner** is perfection: every positive caught, no negatives flagged.

**AUC**, the area under the curve, summarises it as a single number with an
interpretation that is easy to state: it is the probability that a randomly chosen
positive case receives a higher score than a randomly chosen negative one. AUC of 0.5
is chance; 1.0 is perfect separation.

AUC's advantage is that it is **threshold-independent** — it evaluates the ranking
the model produces rather than any particular cut-off. That is also its limitation:
you eventually have to pick a threshold, and AUC will not pick it for you.

> **Figure needed:** an ROC curve with the diagonal marked, ideally with two or three
> models overlaid.

## Connection to hypothesis testing
[Go back to TOC](#table-of-contents)

These ideas are not new; they are the vocabulary of hypothesis testing wearing
different clothes.

| Hypothesis testing | Classification |
| --- | --- |
| Type I error ($\alpha$) | False positive |
| Type II error ($\beta$) | False negative |
| Power ($1 - \beta$) | Sensitivity / recall |
| Significance level | Decision threshold |

The two-distribution plot used to reason about the power of a test — one distribution
under the null, one under the alternative, with a cut-off between them — is the same
picture as a classifier's score distributions for the negative and positive classes.
Sliding the cut-off trades the two error types against each other in exactly the way
moving a classification threshold does.

If that framing is already familiar from statistics, none of this chapter is new
material. It is the same trade-off with new names.

## Choosing a metric
[Go back to TOC](#table-of-contents)

There is no default. The question is always *which mistake costs more here*, and that
is answered by the domain, not by the data.

| Situation | Reasonable choice |
| --- | --- |
| Balanced classes, errors equally costly | Accuracy |
| Imbalanced classes | F1, or precision/recall reported as a pair |
| False positives expensive | Precision |
| False negatives expensive | Recall / sensitivity |
| Comparing rankings, threshold not yet chosen | AUC |
| Need one number for imbalanced data | F1 |

Report more than one. A single number always hides something, and the confusion
matrix itself — four integers — is often the most honest thing you can put in a
report.

## Homework
[Go back to TOC](#table-of-contents)

**[`02_logreg.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/02_logreg.py)
— Logistic Regression from scratch**

That exercise already computes every metric in this chapter. Once your gradient
descent loop runs, it prints the full confusion matrix along with accuracy,
precision, recall, and F1, and compares them against the majority-class baseline:

```
  confusion matrix
                 pred 0   pred 1
      true 0       138       12
      true 1        11      139

  accuracy  0.9233
  precision 0.9205
  recall    0.9267
  f1        0.9236

  majority-class baseline: 0.5000   <- you must beat this
```

Two things worth doing after it runs. Change the decision threshold in
`predict_label()` away from 0.5 and watch precision and recall move in opposite
directions — that is the trade-off in this chapter, made concrete. Then work out
which threshold you would choose if a false positive cost ten times what a false
negative did.

See the [homework README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/README.md)
for setup, and [DEADLINES.md](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/DEADLINES.md)
for the due date.
