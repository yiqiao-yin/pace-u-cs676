# CS676 Algorithms for DataScience

## Table of Contents
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
  - [Session 11: Unsupervised Metrics](#session-11-unsupervised-metrics)
  - [Session 12: Capstone Project Preparation](#session-12-capstone-project-preparation)
  - [Session 13: Capstone Project Work](#session-13-capstone-project-work)

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
4. Gain insights from data through visualization, regression modeling, and cluster analysis.`
5. Apply predictive analytics techniques such as k-nearest neighbors, naïve Bayes, and time series forecasting.
6. Analyze and interpret streaming data in real-time.

[Back to TOC](#table-of-contents)

### Course Structure
This course will be conducted in person, allowing for direct interaction and hands-on assistance.

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

We will be using [this link](https://airtable.com/embed/appBjNPgdot15ZqO7/pagKL7hfbTouEflS9/form) for all of the submissions. You can also access this page from the iOS app (see ✅ tab). The link is set up that allows multiple submissions, but I will only read the last verson you submit. This includes homework assignments, midterm, final projects, and extra credits. **We take this very seriously and you must fill this out after every single class.** Please see the following rubrics:

| Scores | No. of Submission | Formality | Length | Novelty | Business Oriented | Reference |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | Once | Simple and clear | Most length of string of the class | As little from Internet/ChatGPT as possible | Your idea is ready to start a company | Clear list of reference with bibliography summary |
| 2 | Twice | Clear but not simple | Good amount but not the most length of string of the class | Some from Internet/ChatGPT but some from yourself | Can be converted to business idea but not ready | List of reference but no summary |
| 1 | More than twice | Not clear not simple | A few lines or minimum of the class | All from Internet/ChatGPT | Not a business idea at all | No reference or no summary |

### Lecture
The lectures are composed of slides and coding sessions. Both slides and Python notebooks will be used during the lecture. Depending on the material's content, slides and coding sessions may be presented in any order.

The slides and coding materials can be found in the course repo. Please see [here](docs/slide_doc/CS%20676%20Algorithms%20of%20Data%20Science.pdf).

### Final Exam and Class Project
The final project will be a group activity with no more than five students per team. Each team will conduct a data science project culminating in a presentation, which will be recorded and evaluated by external judges.

### Grading Policy
The assessments will count toward your grade as follows:

- **10%** of your grade will be determined by homework assignments. You must fill up the assessment form after every session. You can be excused twice.
- **90%** of your grade will be determined by the projects of the class. Please read the following. 
  - project 1 - credit score:
    - **10%**: deliverable 1
    - **10%**: deliverable 2
    - **10%**: deliverable 3
  - project 2 - tinytroupe:
    - **10%**: deliverable 1
    - **10%**: deliverable 2
    - **10%**: deliverable 3
  - project 3 - to be determined:
    - **10%**: deliverable 1
    - **20%**: deliverable 2
- **30%** (bonus) will be additionally rewarded for extracurriculum activities. Please read [here](./docs/12_extraprojects.md) for detailed information. Please use the course submission form to indicate which project(s) you want to apply for. This can replace project 3 entirely. You may choose to replace project 1 or project 2 with bonus if you want to, although I would recommend to go through project 1 and project 2 to engage with the content of the class.

Late submissions for the midterm will incur a deduction of 5 points from the total score (100 points).

### Course Policies

#### During Class
- The class sessions will be open book and open laptop.
- Students are encouraged to use AI tools, including ChatGPT and Copilot, and may build their own chatbot if desired.

#### Attendance Policy
- Attendance will not be recorded.

#### Policies on Incomplete Grades and Late Assignments
- The lowest two homework grades will be dropped.
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
- Introduction to Python (R is optional by Python is recommended)

For lecture reocrdings, please see the following:

[![Class Recording](https://img.youtube.com/vi/98nOvfGRO9k/0.jpg)](https://www.youtube.com/embed/98nOvfGRO9k?si=HSPty1L_8OBeONjS)

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

[Back to TOC](#table-of-contents)

#### Session 04: Classification
- Logistic regression
- Linear discriminant analysis
- Performance measures for classification

For more details, please see: [04_classification](docs/04_classification.md)

[Back to TOC](#table-of-contents)

#### Session 05: Sampling and Bootstrap
- Importance of sampling
- Bootstrap methods
- Applications of sampling and bootstrap

For more details, please see: [05_sampling_and_bootstrap](docs/05_sampling_and_bootstrap.md)

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

#### Session 11: Unsupervised Metrics
- Introduction to unsupervised metrics
- Evaluation of clustering methods
- Practical applications of unsupervised metrics

For more details, please see: [11_unsupervised](docs/11_unsupervised.md)

[Back to TOC](#table-of-contents)

#### Session 12: Capstone Project Preparation
- Project guidelines
- Team formation
- Initial project planning

For more details, please see: [12_capstone](docs/12_capstone.md)

[Back to TOC](#table-of-contents)

#### Session 13: Capstone Project Work

We expect the students / teams to continue to work on their capstone project.

For more details, please see: [12_capstone](docs/12_capstone.md)

[Back to TOC](#table-of-contents)

## Notebooks
This folder contains Jupyter notebooks used for the project.

[Back to TOC](#table-of-contents)

## Docs
This directory contains additional documentation files.

### Slide Doc
PDF documents for presentation slides. Please see slide here [here](./docs/slide_doc/CS%20676%20Algorithms%20of%20Data%20Science.pdf). If you are accessing from iOS app, please hold and open as a new link from browser.

[Back to TOC](#table-of-contents)