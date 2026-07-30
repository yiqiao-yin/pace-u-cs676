---
sidebar_position: 12
title: "Capstone Projects"
sidebar_label: "12. Capstone Projects"
---

## Table of Contents

- [Capstone Projects](#capstone-projects)
   - [Project 1: Credibility Score for Articles/Sources/References](#project-1-credibility-score-for-articlessourcesreferences)
     - [Concept Overview](#concept-overview)
     - [Approach to Scoring Credibility](#approach-to-scoring-credibility)
     - [Starter Code, Setup, and Grading](#starter-code-setup-and-grading)
     - [Deliverable](#deliverable)
     - [Deliverable Deadline Breakdown](#deliverable-deadline-breakdown)
       - [Deliverable 1: Draft of the Python Function (Sept 19, 2025)](#deliverable-1-draft-of-the-python-function-sept-19-2025)
       - [Deliverable 2: Detailed Technique Report (Oct 3, 2025)](#deliverable-2-detailed-technique-report-oct-3-2025)
       - [Deliverable 3: Implementation into Live Applications (Oct 17, 2025)](#deliverable-3-implementation-into-live-applications-oct-17-2025)
   - [Project 2: TinyTroupe for Simulation](#project-2-tinytroupe-for-simulation)
     - [Concept Overview](#concept-overview-1)
     - [Approach to Simulating Feedback](#approach-to-simulating-feedback)
     - [Starter Code, Setup, and Grading](#starter-code-setup-and-grading-1)
     - [Deliverable](#deliverable-1)
     - [Deliverable Deadline Breakdown](#deliverable-deadline-breakdown-1)
       - [Deliverable 1: Draft of the App (Oct 31, 2025)](#deliverable-1-draft-of-the-app-oct-31-2025)
       - [Deliverable 2: Beta Version and Technical Report (Nov 7, 2025)](#deliverable-2-beta-version-and-technical-report-nov-7-2025)
       - [Deliverable 3: Final Delivery of Container-Ready App (Nov 14, 2025)](#deliverable-3-final-delivery-of-container-ready-app-nov-14-2025)
  - [Project 3: Your Own AI/ML Project](#project-3-your-own-aiml-project)
    - [Concept Overview](#concept-overview-2)
    - [Approach](#approach)
    - [Submission Process](#submission-process)
    - [Deliverable](#deliverable-2)
    - [Deliverable Deadline Breakdown](#deliverable-deadline-breakdown-2)
      - [Deliverable 1: Project Proposal and Initial Work](#deliverable-1-project-proposal-and-initial-work)
      - [Deliverable 2: Final Project Submission](#deliverable-2-final-project-submission)

## Capstone Projects

Please see the following projects.

## Project 1: Credibility Score for Articles/Sources/References

![graph](../pics/12_capstone_01.png)

### Concept Overview
[Go back to TOC](#table-of-contents)

The objective is to assess the credibility of articles, sources, or references through a credibility score. This proof of concept is grounded in the Retrieval-Augmented Generation (RAG) algorithm, which has become increasingly important in modern AI applications for providing accurate, source-backed responses. In today's information-rich environment, users are often overwhelmed by the sheer volume of available sources, making it difficult to distinguish between reliable and unreliable information. This project addresses this critical need by developing an automated system that can evaluate source credibility in real-time.

The use case involves:

- **Chatbot Integration**: Initially, we have a chatbot that employs the RAG algorithm for document-specific Q&A tasks. This chatbot serves as the primary interface where users interact with multiple information sources simultaneously. The integration ensures that users not only receive answers but also understand the reliability of the sources providing those answers.
- **Resource Aggregation**: RAG provides responses drawing from numerous resources across different domains, publications, and databases. These resources can vary significantly in their credibility, ranging from peer-reviewed academic papers to informal blog posts, making credibility assessment essential for maintaining response quality.

The challenge is to understand and evaluate the credibility of these resources through a scoring mechanism. This involves developing sophisticated algorithms that can analyze multiple factors such as source authority, publication quality, citation patterns, and content accuracy to generate meaningful credibility scores that users can trust and understand.

### Approach to Scoring Credibility
[Go back to TOC](#table-of-contents)

1. **Machine Learning-Based**: Utilize machine learning techniques to rate sources by analyzing features derived from those sources. This approach involves training models on large datasets of pre-labeled credible and non-credible sources, enabling the system to learn patterns and characteristics that indicate reliability. Features may include author credentials, publication metrics, citation counts, domain authority, content quality indicators, and temporal relevance. The ML approach offers the advantage of adaptability and can improve over time as more data becomes available.

2. **Rule-Based**: Define specific rules or heuristics to assess credibility based on established journalism and academic standards. These rules might include checking for proper citation practices, verifying author expertise in the subject matter, evaluating the reputation of publishing platforms, and assessing the presence of fact-checking processes. Rule-based systems provide transparency and interpretability, allowing users to understand exactly why a source received a particular credibility score. This approach is particularly valuable for domains with well-established credibility criteria.

3. **Hybrid Approach**: Combine both ML and rule-based methods for a comprehensive evaluation that leverages the strengths of both methodologies. The hybrid system can use rule-based components to establish baseline credibility assessments and handle edge cases, while ML components can identify subtle patterns and relationships that might be missed by predefined rules. This approach often provides the most robust and accurate credibility assessments by balancing interpretability with predictive power.

4. **Innovative Solutions**: Consider any other creative solutions that enhance credibility assessment beyond the traditional methods. This might include real-time fact-checking against multiple databases, sentiment analysis to detect bias, network analysis to understand source relationships and potential conflicts of interest, or blockchain-based verification systems. Innovative approaches could also involve crowd-sourcing credibility assessments, integrating social media sentiment, or using natural language processing to detect misleading language patterns.

### Starter Code, Setup, and Grading
[Go back to TOC](#table-of-contents)

**You do not start this project from an empty folder.** A working chatbot is provided
in [`deliverable/project_1`](https://github.com/yiqiao-yin/pace-u-cs676/tree/main/deliverable/project_1) — clone the
course repository and run it locally on macOS or Windows following the step-by-step
instructions in that folder's
[README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/deliverable/project_1/README.md).

The starter kit contains:

- `main.py` — a Streamlit chat app calling Claude with web search, which already displays a colour-coded credibility chip beside every source it cites.
- `credibility.py` — **the file you improve.** It holds a deliberately weak baseline implementation of `score_url()`, followed by a numbered list of twelve documented defects in that baseline. Each one is an invitation.
- `evaluate.py` — a harness that scores 24 labelled URLs and reports mean absolute error, band accuracy, and worst-case error. The provided baseline scores **MAE 0.142 / 66.7% band accuracy**. Beating that measurably is the point of the assignment.
- `test_credibility.py` — contract tests for the required input/output shape.

The rule-based scorer, the tests, and the evaluation harness all run **without an API
key**, so you can begin immediately and only spend credits once you reach the chat
interface.

**Weighting: this project is 30% of your course grade, marked out of 100 points**
(deliverable 1: 25, deliverable 2: 35, deliverable 3: 40). An additional **5% bonus**
is added to your course grade for deploying a working app to Hugging Face Spaces. The
detailed point-by-point rubric is in the project README, and the full course weighting
and letter-grade scale are in the [course
README](https://github.com/yiqiao-yin/pace-u-cs676#grading-policy).

### Deliverable
[Go back to TOC](#table-of-contents)

The deliverable includes the implementation of a feature within the chatbot to display a credibility score alongside source references. This feature represents a significant enhancement to the user experience by providing immediate, actionable information about source reliability. The implementation must be seamless, efficient, and user-friendly, ensuring that credibility information enhances rather than clutters the chatbot interface. The scoring system should be calibrated to provide meaningful distinctions between sources while avoiding false precision that might mislead users.

This feature will involve:

- **Python Function**: A function designed to evaluate the URL of each reference through comprehensive analysis of multiple credibility indicators. The function must be robust enough to handle various types of sources (academic papers, news articles, government publications, etc.) while being efficient enough for real-time application. It should implement error handling for cases where sources are inaccessible or insufficient data is available for analysis.
  - **Input Argument**: The URL of the reference, which serves as the primary identifier for the source to be evaluated.
  - **Output**: A JSON object containing structured credibility information that is both machine-readable and easily interpretable:
    ```json
    {
      "score": float,
      "explanation": string
    }
    ```
  - **Example Output**: The output provides a numerical score (typically between 0 and 1) along with a human-readable explanation of the scoring rationale:
    ```json
    {"score": 0.90, "explanation": "This source is considered credible based on its citation count and author credentials."}
    ```

### Deliverable Deadline Breakdown
[Go back to TOC](#table-of-contents)

#### Deliverable 1: Draft of the Python Function (Sept 19, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Develop a preliminary version of the Python function that evaluates the URL of each reference. This initial implementation serves as a proof-of-concept to demonstrate the feasibility of automated credibility assessment and establish the foundation for more sophisticated evaluation mechanisms. The focus at this stage is on creating a functional prototype that can process URLs and generate basic credibility scores, even if the scoring algorithm is simplified.
- **Deliverables**:
  - A working draft of the function with basic functionality to return a JSON object containing structured credibility information. The function should handle common URL formats, implement basic error handling for invalid inputs, and provide consistent output formatting. At this stage, the scoring mechanism may rely on simple heuristics or basic feature extraction, but it must demonstrate the core functionality:
    ```json
    {
      "score": float,
      "explanation": string
    }
    ```
  - Initial testing to validate input/output handling, including test cases for various URL types, edge cases for malformed inputs, and verification that the JSON output format is consistent and properly structured. The testing should also include performance benchmarks to ensure the function can handle reasonable loads without significant delays.

#### Deliverable 2: Detailed Technique Report (Oct 3, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Provide an in-depth analysis and report on the algorithmic approach and scientific research supporting the credibility scoring. This deliverable focuses on the theoretical foundation and empirical justification for the chosen methodology, ensuring that the credibility assessment system is grounded in established research and best practices. The report should demonstrate a thorough understanding of the credibility assessment domain and provide a roadmap for algorithmic improvements.
- **Deliverables**:
  - A comprehensive report covering multiple critical aspects of the credibility assessment system. The report should be written at a technical level appropriate for peer review and should include experimental validation of the chosen approach:
    - The underlying algorithm used and its rationale, including detailed explanations of feature selection, scoring mechanisms, and decision thresholds. This section should provide sufficient detail for reproduction and include discussions of algorithm complexity and scalability considerations.
    - Literature review of existing models and techniques for credibility assessment, covering both academic research and industry implementations. The review should identify gaps in current approaches and explain how the proposed solution addresses these limitations.
    - Justification of chosen methodologies, including both ML-based and rule-based approaches if applicable, with empirical evidence supporting the selection criteria. This should include comparative analysis of different approaches and discussion of trade-offs between accuracy, interpretability, and computational efficiency.
  - Documentation to guide future iterations and refinements, including detailed API specifications, algorithm parameters that may need tuning, and identified areas for improvement. The documentation should also include guidelines for maintaining and updating the credibility assessment model as new research becomes available.

#### Deliverable 3: Implementation into Live Applications (Oct 17, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Integrate the finalized Python function into live applications and ensure seamless operation with the chatbot. This final deliverable represents the transition from prototype to production-ready system, requiring careful attention to performance optimization, user experience design, and system reliability. The integration must be robust enough to handle real-world usage patterns while maintaining the quality and accuracy of credibility assessments.
- **Deliverables**:
  - Full implementation of the credibility scoring feature within the chatbot platform, including user interface components that display credibility scores in an intuitive and non-intrusive manner. The implementation should handle concurrent requests efficiently and provide fallback mechanisms for cases where credibility assessment fails or takes too long to complete.
  - Testing and validation to ensure correct functionality and user interaction across different scenarios, including unit tests for individual components, integration tests for the complete system, and user acceptance testing to validate the interface design. The testing should cover edge cases, error conditions, and performance under load.
  - Integration support using a provided application template to streamline the process, including deployment scripts, configuration management, and monitoring capabilities. The integration should be designed for easy maintenance and updates, with clear separation between the credibility assessment logic and the chatbot infrastructure.
  - Please follow the following rubrics for this deliverable!

**Project Deliverable Rubrics**

| **Aspect**                | **Requirements**                                                                                                                                                            |
|---------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Code Comments**         | Each section of code should include **three to five lines of comments**. Ensure the comments are clear and explanatory, providing context and purpose for each code block.  |
| **Novelty**               | Demonstrate novelty in your neural network model architecture. Provide a defense or counterargument for class-discussed assumptions with reasonable accuracy.               |
| **Model Accuracy**        | Aim for high accuracy in the credibility scoring model. While not heavily weighted, higher accuracy is preferred.                                                          |
| **Production-Ready Pipeline** | Ensure the model is production-ready by deploying it on **Hugging Face**. Include Python code in your notebook to demonstrate the deployment of the model artifact.      |

Feel free to adjust the content as per additional details or specifications you might have!

## Project 2: TinyTroupe for Simulation

![graph](../pics/12_capstone_02.png)

**Weighting: 30% of your course grade, marked out of 100 points, plus a +5% bonus for
deploying a working app to Hugging Face Spaces.** See the [course
README](https://github.com/yiqiao-yin/pace-u-cs676#grading-policy) for the full
weighting and the letter-grade scale.

### Starter Code, Setup, and Grading
[Go back to TOC](#table-of-contents)

**Project 1 asked you to improve a function inside someone else's app. This one asks
you to build a Python package.** A runnable skeleton called **PersonaForge** is
provided in
[`deliverable/project_2`](https://github.com/yiqiao-yin/pace-u-cs676/tree/main/deliverable/project_2)
— clone the course repository and follow the macOS and Windows instructions in that
folder's
[README](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/deliverable/project_2/README.md).

What the skeleton does: you talk to an agent in your terminal, ask it to invent
characters, and then tell those characters to talk to each other. The architecture is
one idea repeated — **a persona is a markdown file, an agent is that file plus a model,
and a conversation is agents taking turns.** Personas are written to `temp/` as `.md`
files you can open and edit by hand; change a line and the character behaves
differently on the next run.

It is a proper `uv` project with a `src/` layout, a real package under
`src/personaforge/`, and a `tests/` suite:

- `main.py` — entry point. `uv run main.py`
- `src/personaforge/persona.py` — persona files: generate, save, load, find
- `src/personaforge/agent.py` — markdown plus model becomes a character
- `src/personaforge/conversation.py` — the turn loop
- `src/personaforge/orchestrator.py` — **the weakest part.** It routes what you typed using regular expressions. Replacing that with genuine Claude tool use is the headline task of this project; the tool schemas you need are already sketched in a comment there.
- `tests/` — **39 tests that run offline in under a second**, because every test injects a fake model instead of calling the API

Run `uv run main.py --offline` to drive the entire app with no API key and no cost —
the scripted stub creates personas and runs conversations so you can see the shape of
the system before spending anything. Every module ends with a **YOUR TASK** comment
block listing what is wrong with it; those lists are the assignment.

For background on the Microsoft TinyTroupe library that inspired this project, see
[`tinytroupe_usage_guide.md`](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/deliverable/project_2/tinytroupe_usage_guide.md).
You are not required to use TinyTroupe itself — you are building your own package.

### Concept Overview
[Go back to TOC](#table-of-contents)

This project aims to demonstrate the use of simulation to generate feedback for features based on customer personas, addressing a critical challenge in modern product development. For example, a company introducing a new button or feature in their iOS app must survey beta customers from targeted demographics to gather feedback. However, this traditional process is expensive and time-consuming due to the need to pay contractors and incentivize participants with rewards, often resulting in limited sample sizes and potential bias in feedback collection. The process can take weeks or months, delaying product launches and increasing development costs significantly.

This project proposes an **AI-first solution** to simulate user feedback for features by modeling different customer personas through sophisticated agent-based simulation. The approach leverages artificial intelligence to create virtual users that behave according to realistic persona characteristics, providing rapid, cost-effective feedback that can inform design decisions early in the development process. By using AI agents to simulate diverse user perspectives, companies can test multiple feature variations quickly and identify potential issues before committing to expensive user studies. Recommended package: [TinyTroup](https://github.com/microsoft/TinyTroupe), which provides a robust framework for creating and managing multiple AI personas in conversational scenarios.

### Approach to Simulating Feedback
[Go back to TOC](#table-of-contents)

1. **Persona-Based Simulation**: Develop an AI model that generates realistic feedback based on predefined personas, such as tech-savvy users, casual users, elderly users, or users with accessibility needs. Each persona should have detailed characteristics including demographic information, technical proficiency levels, usage patterns, preferences, and behavioral tendencies. The simulation must account for how different personas would realistically interact with features, considering factors like cognitive load, prior experience, and contextual constraints. This approach ensures that feedback reflects genuine diversity in user perspectives rather than generic responses.

2. **Feature-Driven Inputs**: Allow the app to take feature descriptions as input and output persona-specific feedback that reflects how each user type would realistically respond. The system should be able to process various feature description formats (text descriptions, wireframes, mockups, or functional specifications) and generate contextually appropriate feedback. Input processing should handle both simple feature descriptions and complex interaction flows, ensuring that the generated feedback addresses usability, functionality, and user experience concerns specific to each persona's perspective.

3. **User Feedback Scenarios**: Simulate common scenarios such as beta feature rollouts, user onboarding experiences, feature discovery processes, and long-term usage patterns. The simulation should model realistic user journeys, including initial reactions, learning curves, adaptation over time, and potential abandonment points. Scenarios should cover both positive and negative user experiences, helping identify potential friction points and optimization opportunities that might not be apparent in traditional testing approaches.

4. **Feedback Analysis**: Aggregate the feedback to draw conclusions about user preferences, feature acceptance, and potential issues across different user segments. The analysis should identify patterns and themes in the simulated feedback, highlight consensus and disagreements between personas, and provide actionable recommendations for feature improvements. The system should generate comprehensive reports that include quantitative metrics (acceptance rates, usage likelihood) and qualitative insights (specific concerns, suggested improvements) to guide product development decisions.

### Deliverable
[Go back to TOC](#table-of-contents)

The deliverable for this project is an interactive app built using **Streamlit** or **Gradio** that can simulate user conversations and display feedback for a given feature and persona. This application serves as a comprehensive tool for product managers, UX designers, and development teams to rapidly prototype and evaluate feature concepts across diverse user segments. The app should provide an intuitive interface that makes persona-based simulation accessible to non-technical team members while offering sufficient depth and customization for detailed analysis.

The app will include:

- **Input Fields**: To specify the feature description and persona type, with support for detailed feature specifications including interaction flows, visual elements, and contextual information. Users should be able to select from predefined personas or create custom persona profiles with specific characteristics, demographics, and behavioral patterns. The input interface should guide users in providing sufficient detail for meaningful simulation while remaining easy to use.

- **Output Display**: A conversational output simulating feedback based on the persona's characteristics, presented in a realistic chat-like interface that mimics actual user feedback sessions. The output should include not only the feedback content but also metadata about the persona's reasoning, confidence levels, and potential follow-up questions. The display should support rich formatting to highlight key insights and concerns raised by different personas.

- **Functionality**: A user-friendly interface that allows users to test various features and personas iteratively, with capabilities for saving simulation results, comparing feedback across personas, and exporting reports for stakeholder review. The interface should support batch processing for testing multiple feature variations simultaneously and provide visualization tools to help identify patterns and trends in the simulated feedback.

### Deliverable Deadline Breakdown
[Go back to TOC](#table-of-contents)

#### Deliverable 1: Draft of the App (Oct 31, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Investigate agentic AI by using TinyTroupe package to understand the capabilities and limitations of persona-based simulation. This phase focuses on establishing familiarity with the technology, exploring different persona configurations, and evaluating the quality of generated conversations. The investigation should provide insights into how effectively AI agents can simulate realistic user behavior and identify areas where the simulation approach shows promise or needs improvement.
- **Deliverables**:
  - A walkthrough of the installation and usage, including detailed setup instructions, dependency management, and configuration options. The walkthrough should address common installation issues and provide troubleshooting guidance for different operating systems and environments. Include performance considerations and system requirements for optimal operation.
  - Initial persona simulation results demonstrating the range of personas that can be effectively simulated, with examples showing how different personality types, demographic characteristics, and usage contexts affect the generated feedback. Results should include both successful simulations and cases where the system produces less realistic or useful outputs.
  - Comments on the conversation stream quality, including analysis of how natural and realistic the generated conversations feel, identification of recurring patterns or limitations in the AI responses, and assessment of whether the personas maintain consistency throughout extended interactions. Comments should also evaluate the diversity and depth of insights generated by different persona types.
  - Deliver a `.md` file where conversation history can be found, organized by persona type and feature being evaluated, with annotations explaining the context and significance of key exchanges. The file should serve as a reference for understanding how different personas respond to various types of features and interaction scenarios.

#### Deliverable 2: Beta Version and Technical Report (Nov 7, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Complete the bulk of the app development and submit a draft app that demonstrates the full potential of persona-based feature simulation. This deliverable represents the core implementation phase where all major features are integrated and tested, resulting in a functional application that can be used for real product development scenarios. The focus is on creating a robust, user-friendly tool that provides valuable insights while being accessible to non-technical team members.
- **Deliverables**:
  - A beta version of agentic AI app with different personas that showcases the full range of simulation capabilities, including multiple predefined personas with diverse characteristics, customizable persona creation tools, and comprehensive feature evaluation workflows. The app should handle various types of feature descriptions and generate meaningful, actionable feedback that reflects realistic user perspectives and concerns.
  - A detailed repository covering multiple aspects of the implementation and demonstrating technical depth:
    - The simulation algorithm design, including detailed documentation of how personas are modeled, how feature descriptions are processed, how conversations are generated, and how feedback is synthesized. The design should explain the underlying AI architecture and decision-making processes.
    - A live conversation can be initiated from your UI, with real-time simulation capabilities that allow users to interact with personas dynamically, ask follow-up questions, and explore different aspects of feature feedback. The conversation interface should feel natural and engaging.
    - Use cases and examples of your own choice that demonstrate the practical value of the simulation approach, including examples from different industries, various types of features (UI elements, workflows, content), and different stages of product development (early concept, detailed design, pre-launch validation).
  - Feedback from a second round of instructor review, with documented responses to suggestions and improvements made based on initial feedback. This should include explanations of design decisions, trade-offs considered, and areas identified for future enhancement.

#### Deliverable 3: Final Delivery of Container-Ready App (Nov 14, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Deliver a fully functional app ready for deployment that can be used in real-world product development scenarios. This final deliverable ensures that the simulation tool is production-ready, scalable, and maintainable, with comprehensive documentation and testing to support ongoing use and development. The objective includes optimizing performance, ensuring reliability, and providing the necessary infrastructure for sustainable operation.
- **Deliverables**:
  - A live app deployed on cloud such as HuggingFace, with proper load balancing, error handling, and monitoring capabilities to ensure consistent availability and performance. The deployment should include appropriate security measures, user authentication if needed, and backup/recovery procedures to protect against data loss or service interruption.
  - Finalized persona database with diverse customer profiles representing a wide range of demographics, technical skill levels, usage contexts, and behavioral patterns. The database should be well-documented, easily expandable, and include validation measures to ensure persona consistency and realism. Each persona should have comprehensive characteristics that enable nuanced, realistic feedback generation.
  - Integration and deployment documentation covering all aspects of system setup, configuration, maintenance, and troubleshooting. Documentation should include API specifications, database schemas, deployment procedures, monitoring guidelines, and update processes. The documentation should enable other developers to maintain and enhance the system effectively.
  - End-to-end testing and validation of app functionality across different scenarios, user loads, and edge cases. Testing should include performance benchmarks, accuracy validation of persona simulations, user acceptance testing with actual product teams, and stress testing to ensure the system can handle realistic usage patterns.

By implementing this simulation app, the project demonstrates how AI can streamline feature feedback collection, reducing costs and accelerating the go-to-market strategy. The result is a scalable, efficient solution for user feedback analysis that can transform how companies approach user research and product validation, potentially reducing feedback collection time from weeks to minutes while maintaining the quality and diversity of insights needed for informed decision-making.

## Project 3: Your Own AI/ML Project
[Go back to TOC](#table-of-contents)

![graph](../pics/12_capstone_03.png)

**Weighting: 30% of your course grade, marked out of 100 points.** This is a required
take-home project on a topic of your own choosing. No Hugging Face bonus applies here —
the two 5% deployment bonuses are attached to Projects 1 and 2. See the [course
README](https://github.com/yiqiao-yin/pace-u-cs676#grading-policy) for the full
weighting and the letter-grade scale.

### Concept Overview
[Go back to TOC](#table-of-contents)

**This project is required and is worth 30% of your course grade.** Unlike Projects 1 and 2, the topic is yours to choose: this is where you show what you can build without a specification handed to you.

This project is entirely open-ended and allows you to pursue your own interests. You can choose from a wide variety of project types, including but not limited to:

- Building a chatbot or conversational AI system
- Writing a research paper on an AI/ML topic
- Training and deploying machine learning models
- Developing an agentic AI application
- Creating a data analysis or visualization tool
- Implementing a computer vision or NLP application
- Experimenting with LLMs and prompt engineering
- Building an AI-powered web application
- Any other AI/ML-related project of your choice

The key requirement is that your project demonstrates meaningful engagement with AI or machine learning concepts covered in the course.

### Approach
[Go back to TOC](#table-of-contents)

Your approach will depend entirely on the project you choose. Consider the following when planning your project:

1. **Scope Appropriately**: Choose a project you can genuinely finish in the time available. A small system that works end to end beats an ambitious one that does not run.

2. **Leverage Course Concepts**: Try to incorporate concepts, tools, or techniques we've covered in class (RAG, agentic AI, LLMs, machine learning pipelines, etc.).

3. **Focus on Learning**: This is an opportunity to explore something you're genuinely interested in, so prioritize learning and experimentation.

4. **Document Your Work**: Maintain clear documentation of your process, decisions, and results.

### Submission Process
[Go back to TOC](#table-of-contents)

To submit your project, follow these steps:

1. **Create a GitHub Repository**: Host your project code, documentation, and any supporting materials in a GitHub repository.

2. **Include a README**: Your repository should have a comprehensive README.md file that explains:
   - What your project does
   - Why you chose this project
   - Technologies and tools used
   - How to run/reproduce your work
   - Results and conclusions

3. **Submit via wyn360search.com**:
   - Navigate to [wyn360search.com](https://wyn360search.com)
   - Log in to your account
   - Go to the REPO page
   - Click the 'ADD REPOSITORY' button
   - Enter your GitHub repository URL
   - Complete the submission

Once you have added your repository through the website, the instructors will be able to review your work and provide feedback.

### Deliverable
[Go back to TOC](#table-of-contents)

Your deliverable will vary based on your chosen project, but should generally include:

- A GitHub repository containing all project files
- A comprehensive README.md file
- Source code (if applicable)
- Documentation explaining your methodology and results
- Any necessary configuration files (requirements.txt, environment files, etc.)
- Results, outputs, or demonstrations of your work

### Deliverable Deadline Breakdown
[Go back to TOC](#table-of-contents)

#### Deliverable 1: Project Proposal and Initial Work
[Go back to TOC](#table-of-contents)

- **Objective**: Define your project scope and begin initial development or research.

- **Deliverables**:
  - A project proposal (can be in your README.md) that outlines:
    - Project title and description
    - Objectives and goals
    - Planned approach and technologies
    - Expected outcomes
  - Initial work demonstrating progress on your project
  - A GitHub repository with your initial commits

#### Deliverable 2: Final Project Submission
[Go back to TOC](#table-of-contents)

- **Objective**: Complete your project and submit it for evaluation.

- **Deliverables**:
  - A complete GitHub repository with all project files
  - Comprehensive documentation and README
  - Working code, trained models, or completed research paper (depending on your project type)
  - Submission through the wyn360search.com website as described above
  - A brief reflection on what you learned and any challenges you encountered

**Note**: Submit early enough to get feedback. Work that arrives after the last session cannot be reviewed before grades are due.
