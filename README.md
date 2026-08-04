# CS676 Algorithms for Data Science

📖 **Course site: [yiqiao-yin.github.io/pace-u-cs676](https://yiqiao-yin.github.io/pace-u-cs676/)** — all lecture notes below, rendered with LaTeX math and per-session navigation.

🎞️ **Slide deck: [Interactive presentation](https://main.d3j8dqgo1nf8ma.amplifyapp.com)** — all 330 slides in your browser, with chapter navigation, full-text search, and a presenter mode. No download required.

🗓️ **[Deadlines](DEADLINES.md)** — every due date for homework and projects, in one file.

## Table of Contents
- [Deadlines](DEADLINES.md)
- [Syllabus](#syllabus)
- [Course Topics](#course-topics)
  - [Schedule and Weekly Learning Goals](#schedule-and-weekly-learning-goals)
  - [Session 01: Introduction](#session-01-introduction)
  - [Session 02: Basics in Statistical Learning](#session-02-basics-in-statistical-learning)
  - [Session 03: Linear Regression](#session-03-linear-regression)
  - [Session 04: Classification](#session-04-classification)
  - [Session 05: Sampling and Bootstrap](#session-05-sampling-and-bootstrap)
  - [Session 06: Model Selection & Regularization](#session-06-model-selection--regularization)
  - [Session 07: Going Beyond Linearity](#session-07-going-beyond-linearity)
  - [Session 08: Tree-based Methods and Midterm](#session-08-tree-based-methods-and-midterm)
  - [Session 09: Support Vector Machine](#session-09-support-vector-machine)
  - [Session 10: Deep Learning](#session-10-deep-learning)
  - [Session 11: Unsupervised Learning](#session-11-unsupervised-learning)
  - [Session 12: Classification Metrics](#session-12-classification-metrics)
  - [Session 13: Capstone Project Preparation](#session-13-capstone-project-preparation)
- [Homework — algorithms from scratch](#homework--algorithms-from-scratch)

## Syllabus

### Course Description
This course delves into essential algorithms for data analytics with a computational emphasis. Students will master Python and R to build algorithms and analyze data. Key topics include data reduction (data mapping, data dictionaries, scalable algorithms, big data), data visualization, regression modeling, and cluster analysis. The course also covers predictive analytics techniques such as k-nearest neighbors, naïve Bayes, time series forecasting, and analyzing streaming data. By the end of the course, students will be proficient in leveraging these algorithms to extract meaningful insights from large datasets.

[Back to TOC](#table-of-contents)

### Required Materials
Please see the following recommended text:
- [Introduction to Statistical Learning](https://www.statlearning.com/)
- [Notes on Agent-based Applications](https://www.amazon.com/dp/9999320023/ref=cm_sw_r_ffobk_cso_cp_apin_dp_HTEJCKH50ZSSMBBEDDWJ_1?newOGT=1)

**Note**: For all links in all course documents, if you are accessing this page from iOS app, hold the links and you can open in a new tab. I found that way easier to navigate. 

### Prerequisites/Corequisites
Prerequisites: Open to Data Science Majors.

### Course Objectives
Successful students will:
1. Develop proficiency in Python for data analytics.
2. Implement algorithms for data reduction, including data mapping and data dictionaries.
3. Utilize scalable algorithms to handle big data.
4. Gain insights from data through visualization, regression modeling, and cluster analysis.
5. Apply predictive analytics techniques such as k-nearest neighbors, naïve Bayes, and time series forecasting.
6. Analyze and interpret streaming data in real-time.

[Back to TOC](#table-of-contents)

### Course Structure
This course will be conducted online asynchronously, allowing students to learn at their own pace with flexible scheduling.

Each session will be divided into two main parts:
1. The lecture portion will last 1 hour, where key concepts and theoretical foundations will be covered.
2. The coding session will follow, lasting approximately 1-1.5 hours, depending on the content and complexity of the day's material.

A coding component is required for this course. We recommend using Google Colab, which allows students to write and execute Python code in a web-based environment, easily accessible through Google Drive.

#### Assessments
Students must demonstrate proficiency in the following areas:

1. **Data Engineering:** Handle, preprocess, and store large datasets efficiently.
2. **Data Visualization:** Create insightful visualizations to communicate data findings.
3. **Basic Machine Learning:** Understand and apply fundamental machine learning algorithms or tools.
4. **Basic API Calls:** Make and utilize API calls to interact with different data services.

We will be using [this link](https://airtable.com/appBjNPgdot15ZqO7/pagKL7hfbTouEflS9/form) for all of the submissions. You can also access this page from the iOS app (see ✅ tab). The link is set up that allows multiple submissions, but I will only read the last version you submit. This includes homework assignments, midterm, final projects, and extra credits. **We take this very seriously and you must fill this out after every single class.** Please see the following rubrics:

**Before the first deadline, set up your submission repository.** Three rules, and they
matter more than they look:

1. **One GitHub repository for the entire course** — not one per assignment.
2. **Its name must contain the course number**, for example `cs676-jane-doe` or `cs676-fall-2026`. A repository called `homework` or `untitled-3` is one I cannot place among a class of them.
3. **Submit the same URL every week.** Point the form at that one repository each time, and let the folder structure show what is finished — do not link individual files.

Keep it organized: a `homework/` folder and a folder per project, with a top-level
`README.md` saying what is where. If the repository is private, add me as a collaborator
— a link I cannot open counts as nothing submitted.

Full details, the suggested folder layout, and every deadline are in
**[DEADLINES.md](DEADLINES.md)**.

| Scores | Submission Status |
| --- | --- |
| 1 | Pass - Homework submitted |
| 0 | Fail - Homework not submitted |

### Lecture
The lectures are composed of slides and coding sessions. Both slides and Python notebooks will be used during the lecture. Depending on the material's content, slides and coding sessions may be presented in any order.

The slides and coding materials can be found in the course repo.

- **[Interactive slide deck](https://main.d3j8dqgo1nf8ma.amplifyapp.com)** — recommended. All 330 slides in the browser: chapter menu on the left, progress dots across the top, full-text search, and a presenter mode (press `f`). Use `←`/`→` to move between slides.
- **[Original PDF](docs/slide_doc/CS%20676%20Algorithms%20of%20Data%20Science.pdf)** — the same material as a download.

### Final Exam and Class Project
The final project will be an individual project that is submission based. It will be a culmination of everything you know. You have full flexibility of the content of the application. You'll need to include a backend, a frontend, and some form of invocation of LLM.

### Grading Policy
The assessments will count toward your grade as follows:

Your grade is built from four components. Each project is marked out of 100 points and then weighted as shown.

| Component | Weight | Bonus available | Details |
| --- | --- | --- | --- |
| Homework | **10%** | — | The per-session submission form **and** the five Pass/Fail exercises in [`notebooks/homework/`](notebooks/homework/). All five count — none are dropped. |
| Project 1 — Credibility Scoring | **30%** | **+5%** | [Starter code and rubric](deliverable/project_1/README.md) · [Spec](docs/13_capstone.md#project-1-credibility-score-for-articlessourcesreferences) |
| Project 2 — PersonaForge (agent-to-agent package) | **30%** | **+5%** | [Starter code and rubric](deliverable/project_2/README.md) · [Spec](docs/13_capstone.md#project-2-personaforge--build-an-agent-to-agent-package) |
| Project 3 — Your Own AI/ML Project | **30%** | — | Required. Take-home, your own idea. [Spec](docs/13_capstone.md#project-3-your-own-aiml-project) |
| | **100%** | **+10%** | **110% total available** |

**Both bonuses are earned by deploying a working app to Hugging Face Spaces** — 5% for Project 1, 5% for Project 2. This is real additional work, which is why it carries real additional credit. Submit the public Space URL with your deliverables.

**Every deadline is in [DEADLINES.md](DEADLINES.md)** — homework and projects, one file,
nothing repeated elsewhere.

**Each project is a single deliverable with a single deadline.** There is no
deliverable 1, 2, 3 to hand in separately. The project specs break the work into parts —
25 / 35 / 40 points for Project 1, for instance — but those show where the marks are in
your one submission, not when things are due. Work through them in any order, or all at
once. What to build is in each project's own README:
[Project 1](deliverable/project_1/README.md) · [Project 2](deliverable/project_2/README.md) ·
[Project 3](docs/13_capstone.md#project-3-your-own-aiml-project).

### Letter Grades

Your final percentage maps to a letter grade on this scale. Because 110% is available, the bonuses can carry you above 100% — a student who finishes everything and deploys both apps can absorb a weak deliverable elsewhere and still earn an A.

| Final Score | Letter |
| --- | --- |
| 95% and above | **A** |
| 90 – 94% | **A-** |
| 85 – 89% | **B+** |
| 80 – 84% | **B** |
| 75 – 79% | **B-** |
| 70 – 74% | **C+** |
| 65 – 69% | **C** |
| 60 – 64% | **C-** |
| Below 60% | **F** |

There is no rounding beyond the bands shown: an 89.9% is a B+, not an A-.

Late submissions for the midterm will incur a deduction of 5 points from the total score (100 points).

### Course Policies

#### During Class
- The class sessions will be open book and open laptop.
- Students are encouraged to use AI tools, including ChatGPT and Copilot, and may build their own chatbot if desired.

#### Attendance Policy
- Attendance will not be separately recorded but the weekly submission will help me understand your progress and attendance.

#### Policies on Incomplete Grades and Late Assignments
- All five homework assignments count toward your grade; none are dropped.
- There will be no make-up sessions for the midterm and final exams.

#### Academic Integrity and Honesty
Students must comply with the university policy on academic integrity found in the Code of Student Conduct.

#### Accommodations for Disabilities
Reasonable accommodations will be made for students with verifiable disabilities. Students must register with the Disability Services Office to take advantage of available accommodations.

Discrimination and harassment of any form are not tolerated. Retaliation against any person who complains about discrimination is also prohibited.

[Back to TOC](#table-of-contents)

## Course Topics

### Schedule and Weekly Learning Goals

The schedule is tentative and subject to change. The learning goals below should be viewed as the key concepts you should grasp after each week, and also as a study guide before each exam, and at the end of the semester. Each exam will test on the material that was taught up until 1 week prior to the exam. The applications in the second half of the semester tend to build on the concepts in the first half of the semester though, so it is still important to at least review those concepts throughout the semester.

#### Session 01: Introduction
- Overview of the course
- Importance of data science
- Introduction to Python (R is optional but Python is recommended)

For more details, please see: [01_introduction](docs/01_introduction.md)

[Back to TOC](#table-of-contents)

#### Session 02: Basics in Statistical Learning
- Understanding statistical learning
- Key concepts and definitions
- Examples of statistical learning applications

For more details, please see: [02_basics_in_stat_learning](docs/02_basics_in_stat_learning.md)

[Back to TOC](#table-of-contents)

#### Session 03: Linear Regression
- Simple linear regression
- Multiple linear regression
- Assessing the accuracy of the model

For more details, please see: [03_linear_regression](docs/03_linear_regression.md)

🧮 **Homework: [`01_lr.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/01_lr.py)** — you write the MSE gradient and the gradient descent loop. See the [homework README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/README.md).

[Back to TOC](#table-of-contents)

#### Session 04: Classification
- Logistic regression
- Linear discriminant analysis
- Performance measures for classification

For more details, please see: [04_classification](docs/04_classification.md)

🧮 **Homework: [`02_logreg.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/02_logreg.py)** — you write the sigmoid and the gradient descent loop. See the [homework README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/README.md).

[Back to TOC](#table-of-contents)

#### Session 05: Sampling and Bootstrap
- Importance of sampling
- Bootstrap methods
- Applications of sampling and bootstrap

For more details, please see: [05_sampling_and_bootstrap](docs/05_sampling_and_bootstrap.md)

🧮 **Homework: [`03_cv.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/03_cv.py)** — you write the fold construction and the k-fold rotation loop. See the [homework README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/README.md).

[Back to TOC](#table-of-contents)

#### Session 06: Model Selection & Regularization
- Criteria for model selection
- Ridge regression
- Lasso regression

For more details, please see: [06_model_selection](docs/06_model_selection.md)

[Back to TOC](#table-of-contents)

#### Session 07: Going Beyond Linearity
- Polynomial regression
- Step functions
- Basis functions and splines

For more details, please see: [07_going_beyond_linearity](docs/07_going_beyond_linearity.md)

[Back to TOC](#table-of-contents)

#### Session 08: Tree-based Methods and Midterm
- Decision trees
- Random forests
- Boosting
- Detailed analysis of random forests
- Advanced boosting techniques

For more details, please see: [08_tree_based_model](docs/08_tree_based_model.md)

🧮 **Homework: [`04_tree.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/04_tree.py)** — you write the exhaustive Gini split search. See the [homework README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/README.md).

[Back to TOC](#table-of-contents)

#### Session 09: Support Vector Machine
- Introduction to SVM
- SVM for classification
- SVM for regression

For more details, please see: [09_support_vector_machine](docs/09_support_vector_machine.md)

[Back to TOC](#table-of-contents)

#### Session 10: Deep Learning
- Fundamentals of deep learning
- Neural networks and architectures
- Applications in real-world problems

For more details, please see: [10_neural_networks](docs/10_neural_networks.md)

[Back to TOC](#table-of-contents)

#### Session 11: Unsupervised Learning
- Introduction to unsupervised metrics
- Evaluation of clustering methods
- Practical applications of unsupervised metrics

For more details, please see: [11_unsupervised](docs/11_unsupervised.md)

🧮 **Homework: [`05_kmeans.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/05_kmeans.py)** — you write the assign step, the update step, and the loop. See the [homework README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/README.md).

[Back to TOC](#table-of-contents)

#### Session 12: Classification Metrics
- Confusion matrix, accuracy, sensitivity and specificity
- Precision, recall, and the F1 score
- ROC curves and AUC
- The connection to Type I / Type II error in hypothesis testing

For more details, please see: [12_classification_metrics](docs/12_classification_metrics.md)

🧮 **Homework: [`02_logreg.py`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/notebooks/homework/02_logreg.py)** already computes every metric in this session — the confusion matrix, precision, recall and F1 — once your gradient descent loop runs.

[Back to TOC](#table-of-contents)

#### Session 13: Capstone Project Preparation
- Project guidelines
- Team formation
- Initial project planning

For more details, please see: [13_capstone](docs/13_capstone.md)

[Back to TOC](#table-of-contents)

## Notebooks
This folder contains the Jupyter notebooks used during the coding sessions. Please see the course notebook folder [here](./notebooks/).

A more complete list can be accessed [here](https://github.com/yiqiao-yin/WYNAssociates/tree/main/docs/ref-deeplearning). I update this folder with new latest new AI tools frequently.

### Homework — algorithms from scratch

**[`notebooks/homework/`](./notebooks/homework/)** holds five short exercises that
ask you to implement the algorithms yourself, in plain numpy. No scikit-learn, no
statsmodels — importing them skips the exercise.

Each one is a complete, runnable script with the **core algorithm removed**. The
data, the metrics, the printing and the plots are written for you, so your time goes
on the ten or fifteen lines that actually do the learning. Run a script and it stops
immediately with a `NotImplementedError` pointing at a boxed set of instructions:

| Exercise | Topic | What you write | Session |
| --- | --- | --- | --- |
| [`01_lr.py`](./notebooks/homework/01_lr.py) | Linear regression | the MSE gradient, then the descent loop | 03 |
| [`02_logreg.py`](./notebooks/homework/02_logreg.py) | Logistic regression | the sigmoid, then the descent loop | 04 |
| [`03_cv.py`](./notebooks/homework/03_cv.py) | K-fold cross validation | the fold construction, then the rotation loop | 05 |
| [`04_tree.py`](./notebooks/homework/04_tree.py) | Decision tree | the exhaustive Gini split search | 08 |
| [`05_kmeans.py`](./notebooks/homework/05_kmeans.py) | K-means clustering | the assign step, the update step, then the loop | 11 |

**Every script grades itself, so you are never left guessing.** `01_lr.py` checks
your gradient descent against the closed-form solution and prints `PASS` when they
agree. `02_logreg.py` and `04_tree.py` compare accuracy against the majority-class
baseline. `03_cv.py` shows training error below validation error in each fold.
`05_kmeans.py` asserts that inertia never rises.

The numbers are chosen to raise questions rather than just to be correct — why the
MSE in 01 cannot go below 2.25, why the resubstitution error in 03 lands *below* the
noise floor, why 1.0000 training accuracy in 04 means nothing, and why 05 sometimes
clusters badly while the code is entirely right. Bring answers to those; they are
better exam preparation than the code.

Start with the [homework README](./notebooks/homework/README.md), which covers setup,
the optional `--plot` and `--report` flags, and how to check your work. Submit through
the usual [course form](https://airtable.com/appBjNPgdot15ZqO7/pagKL7hfbTouEflS9/form).

[Back to TOC](#table-of-contents)

## Docs
This directory contains additional documentation files.

### Slide Doc
The slides are available two ways:

- **[Interactive presentation](https://main.d3j8dqgo1nf8ma.amplifyapp.com)** — 330 slides rendered in the browser with a chapter table of contents, dot progress bar, search across every slide, and keyboard navigation.
- **[Original PDF](./docs/slide_doc/CS%20676%20Algorithms%20of%20Data%20Science.pdf)** — the source document.

If you are accessing from the iOS app, please hold and open as a new link from the browser.

[Back to TOC](#table-of-contents)