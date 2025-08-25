# Table of Contents

- [Capstone Projects](#capstone-projects)
   - [Project 1: Credibility Score for Articles/Sources/References](#project-1-credibility-score-for-articlessourcesreferences)
     - [Concept Overview](#concept-overview)
     - [Approach to Scoring Credibility](#approach-to-scoring-credibility)
     - [Deliverable](#deliverable)
     - [Deliverable Deadline Breakdown](#deliverable-deadline-breakdown)
       - [Deliverable 1: Draft of the Python Function (Sept 12, 2025)](#deliverable-1-draft-of-the-python-function-sept-12-2025)
       - [Deliverable 2: Detailed Technique Report (Sept 19, 2025)](#deliverable-2-detailed-technique-report-sept-19-2025)
       - [Deliverable 3: Implementation into Live Applications (Sept 26, 2025)](#deliverable-3-implementation-into-live-applications-sept-26-2025)
   - [Project 2: TinyTroupe for Simulation](#project-2-tinytroupe-for-simulation)
     - [Concept Overview](#concept-overview-1)
     - [Approach to Simulating Feedback](#approach-to-simulating-feedback)
     - [Deliverable](#deliverable-1)
     - [Deliverable Deadline Breakdown](#deliverable-deadline-breakdown-1)
       - [Deliverable 1: Draft of the App (Oct 3, 2025)](#deliverable-1-draft-of-the-app-oct-3-2025)
       - [Deliverable 2: Beta Version and Technical Report (Oct 10, 2025)](#deliverable-2-beta-version-and-technical-report-oct-10-2025)
       - [Deliverable 3: Final Delivery of Container-Ready App (Oct 17, 2025)](#deliverable-3-final-delivery-of-container-ready-app-oct-17-2025)
  - [Project 3: Agentic AI for Machine Learning](#project-3-agentic-ai-for-machine-learning)
    - [Concept Overview](#concept-overview-2)
    - [Approach to Simulating Feedback](#approach-to-simulating-feedback-1)
    - [Deliverable](#deliverable-2)
    - [Deliverable Deadline Breakdown](#deliverable-deadline-breakdown-2)
      - [Optional Deliverable 1: First Draft (Oct 24, 2025)](#optional-deliverable-1-first-draft-oct-24-2025)
      - [Optional Deliverable 2: Second Draft (Dec 5, 2025)](#optional-deliverable-2-second-draft-dec-5-2025)

# Capstone Projects

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

#### Deliverable 1: Draft of the Python Function (Sept 12, 2025)
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

#### Deliverable 2: Detailed Technique Report (Sept 19, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Provide an in-depth analysis and report on the algorithmic approach and scientific research supporting the credibility scoring. This deliverable focuses on the theoretical foundation and empirical justification for the chosen methodology, ensuring that the credibility assessment system is grounded in established research and best practices. The report should demonstrate a thorough understanding of the credibility assessment domain and provide a roadmap for algorithmic improvements.
- **Deliverables**:
  - A comprehensive report covering multiple critical aspects of the credibility assessment system. The report should be written at a technical level appropriate for peer review and should include experimental validation of the chosen approach:
    - The underlying algorithm used and its rationale, including detailed explanations of feature selection, scoring mechanisms, and decision thresholds. This section should provide sufficient detail for reproduction and include discussions of algorithm complexity and scalability considerations.
    - Literature review of existing models and techniques for credibility assessment, covering both academic research and industry implementations. The review should identify gaps in current approaches and explain how the proposed solution addresses these limitations.
    - Justification of chosen methodologies, including both ML-based and rule-based approaches if applicable, with empirical evidence supporting the selection criteria. This should include comparative analysis of different approaches and discussion of trade-offs between accuracy, interpretability, and computational efficiency.
  - Documentation to guide future iterations and refinements, including detailed API specifications, algorithm parameters that may need tuning, and identified areas for improvement. The documentation should also include guidelines for maintaining and updating the credibility assessment model as new research becomes available.

#### Deliverable 3: Implementation into Live Applications (Sept 26, 2025)
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

#### Deliverable 1: Draft of the App (Oct 3, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Investigate agentic AI by using TinyTroupe package to understand the capabilities and limitations of persona-based simulation. This phase focuses on establishing familiarity with the technology, exploring different persona configurations, and evaluating the quality of generated conversations. The investigation should provide insights into how effectively AI agents can simulate realistic user behavior and identify areas where the simulation approach shows promise or needs improvement.
- **Deliverables**:
  - A walkthrough of the installation and usage, including detailed setup instructions, dependency management, and configuration options. The walkthrough should address common installation issues and provide troubleshooting guidance for different operating systems and environments. Include performance considerations and system requirements for optimal operation.
  - Initial persona simulation results demonstrating the range of personas that can be effectively simulated, with examples showing how different personality types, demographic characteristics, and usage contexts affect the generated feedback. Results should include both successful simulations and cases where the system produces less realistic or useful outputs.
  - Comments on the conversation stream quality, including analysis of how natural and realistic the generated conversations feel, identification of recurring patterns or limitations in the AI responses, and assessment of whether the personas maintain consistency throughout extended interactions. Comments should also evaluate the diversity and depth of insights generated by different persona types.
  - Deliver a `.md` file where conversation history can be found, organized by persona type and feature being evaluated, with annotations explaining the context and significance of key exchanges. The file should serve as a reference for understanding how different personas respond to various types of features and interaction scenarios.

#### Deliverable 2: Beta Version and Technical Report (Oct 10, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Complete the bulk of the app development and submit a draft app that demonstrates the full potential of persona-based feature simulation. This deliverable represents the core implementation phase where all major features are integrated and tested, resulting in a functional application that can be used for real product development scenarios. The focus is on creating a robust, user-friendly tool that provides valuable insights while being accessible to non-technical team members.
- **Deliverables**:
  - A beta version of agentic AI app with different personas that showcases the full range of simulation capabilities, including multiple predefined personas with diverse characteristics, customizable persona creation tools, and comprehensive feature evaluation workflows. The app should handle various types of feature descriptions and generate meaningful, actionable feedback that reflects realistic user perspectives and concerns.
  - A detailed repository covering multiple aspects of the implementation and demonstrating technical depth:
    - The simulation algorithm design, including detailed documentation of how personas are modeled, how feature descriptions are processed, how conversations are generated, and how feedback is synthesized. The design should explain the underlying AI architecture and decision-making processes.
    - A live conversation can be initiated from your UI, with real-time simulation capabilities that allow users to interact with personas dynamically, ask follow-up questions, and explore different aspects of feature feedback. The conversation interface should feel natural and engaging.
    - Use cases and examples of your own choice that demonstrate the practical value of the simulation approach, including examples from different industries, various types of features (UI elements, workflows, content), and different stages of product development (early concept, detailed design, pre-launch validation).
  - Feedback from a second round of instructor review, with documented responses to suggestions and improvements made based on initial feedback. This should include explanations of design decisions, trade-offs considered, and areas identified for future enhancement.

#### Deliverable 3: Final Delivery of Container-Ready App (Oct 17, 2025)
[Go back to TOC](#table-of-contents)

- **Objective**: Deliver a fully functional app ready for deployment that can be used in real-world product development scenarios. This final deliverable ensures that the simulation tool is production-ready, scalable, and maintainable, with comprehensive documentation and testing to support ongoing use and development. The objective includes optimizing performance, ensuring reliability, and providing the necessary infrastructure for sustainable operation.
- **Deliverables**:
  - A live app deployed on cloud such as HuggingFace, with proper load balancing, error handling, and monitoring capabilities to ensure consistent availability and performance. The deployment should include appropriate security measures, user authentication if needed, and backup/recovery procedures to protect against data loss or service interruption.
  - Finalized persona database with diverse customer profiles representing a wide range of demographics, technical skill levels, usage contexts, and behavioral patterns. The database should be well-documented, easily expandable, and include validation measures to ensure persona consistency and realism. Each persona should have comprehensive characteristics that enable nuanced, realistic feedback generation.
  - Integration and deployment documentation covering all aspects of system setup, configuration, maintenance, and troubleshooting. Documentation should include API specifications, database schemas, deployment procedures, monitoring guidelines, and update processes. The documentation should enable other developers to maintain and enhance the system effectively.
  - End-to-end testing and validation of app functionality across different scenarios, user loads, and edge cases. Testing should include performance benchmarks, accuracy validation of persona simulations, user acceptance testing with actual product teams, and stress testing to ensure the system can handle realistic usage patterns.

By implementing this simulation app, the project demonstrates how AI can streamline feature feedback collection, reducing costs and accelerating the go-to-market strategy. The result is a scalable, efficient solution for user feedback analysis that can transform how companies approach user research and product validation, potentially reducing feedback collection time from weeks to minutes while maintaining the quality and diversity of insights needed for informed decision-making.

## Project 3: Agentic AI for Machine Learning
[Go back to TOC](#table-of-contents)

![graph](../pics/12_capstone_03.png)

### Concept Overview
[Go back to TOC](#table-of-contents)

We aim to create an intuitive platform that empowers executives and non-technical professionals to leverage advanced data science tools without requiring deep technical knowledge. By integrating built-in functionalities, users can seamlessly interact with machine learning models and perform essential data tasks.

Due to time limit of the semester, this will be an optional project.

### Approach to Simulating Feedback
[Go back to TOC](#table-of-contents)

This project utilizes proprietary agentic AI tools specifically designed to simplify and automate complex data science workflows. By embedding tools we’ve developed, the solution will include guided interactions, enabling users to efficiently complete tasks such as model selection, data preparation, and visualization.

### Deliverable
[Go back to TOC](#table-of-contents)

- A `.py` script containing a Python function that encapsulates the core functionality of the agentic AI tool.
- A `requirements.txt` file documenting all the package versions required to run the Python script.
- A `.json` file containing key metadata, including keywords and sample payloads, to demonstrate the required inputs and expected outputs for the Python function.

### Deliverable Deadline Breakdown
[Go back to TOC](#table-of-contents)

#### Optional Deliverable 1: First Draft (Oct 24, 2025)
[Go back to TOC](#table-of-contents)

- **Optional Deliverable 1**: First Draft of the Python Script, Requirements File, and JSON Metadata File. This initial deliverable focuses on establishing the core architecture and demonstrating the feasibility of the agentic AI approach. The draft should include a working prototype that showcases the key functionality, even if not all features are fully implemented. The emphasis should be on proving the concept and establishing a solid foundation for further development.

The python script must follow the following template to ensure compatibility with the agentic AI framework. The provided code snippet is written in Python with a specific decorator function-like syntax and includes a comment. This template establishes a standardized interface for registering and executing AI agent functions, enabling seamless integration with the broader agentic AI ecosystem. The decorator pattern allows for dynamic function discovery and registration, which is essential for building flexible, extensible AI systems. Here's a detailed breakdown:

```python
@register_function("send_sms")
def send_sms(payload: Dict[str, str], secrets: Dict[str, str], event_stream: list) -> Dict[str, Any]:
    # Code to send email goes here!
    pass
```

1. **Decorator: `@register_function("send_sms")`**
   - The line starting with `@` is a Python decorator that serves as a registration mechanism for agentic AI functions. It is used to modify the behavior of the function below it, adding metadata and registration capabilities that enable the AI system to discover and utilize the function automatically.
   - `register_function` appears to be a custom or library-provided decorator which registers the `send_sms` function under the name `"send_sms"`. This registration system is crucial for agentic AI applications as it allows the AI agent to dynamically discover available functions and select appropriate ones based on user requests or system needs. The decorator could signify that the function can be accessed or utilized elsewhere within a framework, plugin system, or API, enabling modular and extensible AI agent capabilities.

2. **Function Definition: `def send_sms(...)`**
   - The function named `send_sms` takes three parameters that follow the standardized interface for agentic AI functions, ensuring consistency and predictability across all agent capabilities:
     - `payload`: A dictionary (`Dict[str, str]`) where both keys and values are strings. This likely contains the data necessary to send an SMS, such as a message body or recipient's phone number. The standardized payload format enables the AI agent to pass user-requested information in a consistent manner, regardless of the specific function being called.
     - `secrets`: Another dictionary which stores sensitive information, with string keys and values, possibly containing credentials or keys required for sending an SMS. This parameter is crucial for maintaining security in agentic AI systems, as it provides a secure way to pass authentication credentials and API keys without exposing them in logs or user interfaces.
     - `event_stream`: A list that might be used to log events or manage asynchronous operations related to sending SMS. This parameter enables comprehensive monitoring and debugging of agent actions, allowing the system to track the progress of operations, handle errors gracefully, and provide detailed feedback to users about the status of their requests.

3. **Return Type Annotation: `-> Dict[str, Any]`**
   - The function is supposed to return a dictionary where the keys are strings, and the values can be any data type. This standardized return format ensures that all agentic AI functions provide consistent output that can be easily processed by the AI agent and presented to users in a meaningful way. The return dictionary might represent the result or status of the SMS sending operation, including success indicators, error messages, confirmation details, or any other relevant information that the AI agent needs to communicate back to the user.

4. **Comment: `# Code to send email goes here!`**
   - This is an internal comment indicating where the logic for sending an email (or possibly an SMS, given the context mismatch) should be implemented. In a production agentic AI system, this is where the actual business logic would reside, including error handling, validation, external API calls, and result formatting.
   - Note the discrepancy; it mentions "send email," while the function is named `send_sms`. This type of inconsistency should be carefully avoided in production code, as it can lead to confusion and maintenance issues. Proper documentation and naming conventions are especially important in agentic AI systems where functions may be automatically selected and executed based on their names and descriptions.

5. **`pass` Statement**
   - The `pass` keyword is used as a placeholder and means that the function currently doesn't execute any operations. It's a no-op used when a statement is syntactically required but no action is needed or defined. In the context of agentic AI development, this placeholder approach allows developers to establish the function interface and registration mechanism before implementing the full functionality, enabling iterative development and testing of the agent system architecture.

The intention behind this code is to set up a structure for sending SMS messages, potentially using a framework where functions are registered via decorators for use by agentic AI systems. This architectural approach enables AI agents to dynamically discover and utilize various capabilities without hardcoded function calls, making the system highly extensible and maintainable. However, the actual implementation of sending the SMS is not yet complete, which is typical during the initial development phases where the focus is on establishing proper interfaces and integration patterns before implementing specific business logic.

The key words association must be provided in a `.json` file that uses the following template to enable natural language processing and function selection by the AI agent. This metadata file is crucial for helping the AI agent understand when and how to use each registered function, as it provides the semantic mapping between user requests and available capabilities. The JSON structure should be comprehensive enough to handle various ways users might express their intent while being specific enough to avoid ambiguous function selection:

```json
{
    "send_email": {
        "trigger_word": ["send email", "notify via email"],
        "sample_payload": {"email": "string", "subject": "string"},
        "prerequisite": null
    }
}
```

#### Optional Deliverable 2: Second Draft (Dec 5, 2025)
[Go back to TOC](#table-of-contents)

- **Optional Deliverable 2**: Second Draft with Revisions and Final Adjustments, incorporating feedback from initial testing and addressing any identified limitations or bugs. The second draft should represent a significant improvement over the initial version, with enhanced functionality, better error handling, improved documentation, and validated performance across different use cases. This final version should be ready for integration into production environments and include comprehensive testing results and performance benchmarks. The deliverable should demonstrate mastery of agentic AI concepts and provide a robust, scalable solution that can serve as a foundation for future enterprise applications.
