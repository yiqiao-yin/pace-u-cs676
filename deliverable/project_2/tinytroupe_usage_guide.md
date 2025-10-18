# TinyTroupe Usage Guide

## Overview

TinyTroupe is an experimental Python library that allows the simulation of people with specific personalities, interests, and goals using Large Language Models (LLMs). This guide covers the basic usage to help you get started with persona-based simulations.

## Prerequisites

Before using TinyTroupe, you need:

- **Python 3.10 or higher**
- **API Access**: Either Azure OpenAI Service or OpenAI GPT-4 API
  - For Azure OpenAI: Set `AZURE_OPENAI_KEY` and `AZURE_OPENAI_ENDPOINT` environment variables
  - For OpenAI: Set `OPENAI_API_KEY` environment variable

## Installation

Install TinyTroupe directly from the GitHub repository:

```bash
# Create a new Python environment (recommended)
conda create -n tinytroupe python=3.10
conda activate tinytroupe

# Install from GitHub
pip install git+https://github.com/microsoft/TinyTroupe.git@main
```

## Core Concepts

TinyTroupe provides two key abstractions:

1. **TinyPerson**: Agents with specific personalities that receive stimuli and act upon them
2. **TinyWorld**: Environment where agents exist and interact

## Basic Usage

### 1. Creating a TinyPerson

#### Using Pre-defined Agents

TinyTroupe provides example agents you can use immediately:

```python
from tinytroupe.examples import create_lisa_the_data_scientist

# Create a pre-defined persona
lisa = create_lisa_the_data_scientist()
lisa.listen_and_act("Tell me about your life.")
```

#### Loading from JSON Specification

Create an agent from a JSON file:

```python
from tinytroupe.agent import TinyPerson

# Load agent from JSON specification
person = TinyPerson.load_specification("path/to/agent.json")
```

Example JSON structure:

```json
{
    "type": "TinyPerson",
    "persona": {
        "name": "Lisa Carter",
        "age": 28,
        "occupation": {
            "title": "Data Scientist",
            "organization": "Microsoft",
            "description": "Works on M365 Search team analyzing user behavior..."
        },
        "personality": {
            "traits": [
                "Curious and loves to learn new things",
                "Analytical problem solver"
            ]
        }
    }
}
```

#### Programmatic Definition

Define an agent using Python code:

```python
from tinytroupe.agent import TinyPerson

# Create a new person
person = TinyPerson("John")

# Define characteristics
person.define("age", 35)
person.define("nationality", "American")
person.define("occupation", {
    "title": "Software Engineer",
    "organization": "Tech Corp",
    "description": "Builds web applications..."
})

person.define("personality", {
    "traits": [
        "Creative and innovative",
        "Detail-oriented"
    ]
})
```

### 2. Using TinyPersonFactory

Generate agents automatically using LLMs:

```python
from tinytroupe.factory import TinyPersonFactory

# Create factory with context
factory = TinyPersonFactory(context="A hospital in São Paulo.")

# Generate a person with specific characteristics
person = factory.generate_person(
    "Create a Brazilian doctor who likes pets and loves heavy metal."
)

# Generate multiple people at once
people = factory.generate_people(number_of_people=10, parallelize=True)
```

Create factory from demographic data:

```python
factory = TinyPersonFactory.create_factory_from_demography(
    demography_description_or_file_path="path/to/demography.json",
    population_size=50,
    context="Urban professionals in technology sector"
)
```

### 3. Creating a TinyWorld

Set up an environment for agent interactions:

```python
from tinytroupe.environment import TinyWorld

# Create a world with multiple agents
world = TinyWorld("Chat Room", [lisa, oscar])

# Make agents accessible to each other
world.make_everyone_accessible()

# Start interaction
lisa.listen("Talk to Oscar to know more about him")

# Run simulation for 4 steps
world.run(4)
```

### 4. Agent Interactions

Agents can interact through various methods:

```python
# Agent listens to input
agent.listen("What are your thoughts on AI?")

# Agent sees something
agent.see("A beautiful sunset")

# Agent listens and acts
agent.listen_and_act("Introduce yourself")

# Agent acts
agent.act()
```

## Common Use Cases

### 1. Customer Interview

```python
# Create personas
consultant = TinyPerson("Consultant")
banker = create_banker_persona()

# Set up conversation
world = TinyWorld("Interview Room", [consultant, banker])
world.make_everyone_accessible()

# Start interview
consultant.listen_and_act("Ask the banker about their workflow challenges")
world.run(5)
```

### 2. Product Feedback Simulation

```python
# Create diverse user personas
factory = TinyPersonFactory(context="Software users")
users = factory.generate_people(number_of_people=5, parallelize=True)

# Create focus group
focus_group = TinyWorld("Focus Group", users)
focus_group.make_everyone_accessible()

# Present product feature
for user in users:
    user.listen("What do you think about this new AI writing assistant feature?")

focus_group.run(3)
```

### 3. Advertisement Evaluation

```python
# Create target audience personas
personas = [create_tech_savvy_user(), create_casual_user(), create_elderly_user()]

# Show them an ad
world = TinyWorld("Ad Evaluation", personas)
ad_description = "A smartwatch that tracks health metrics and sends alerts"

for persona in personas:
    persona.listen(f"Rate this product: {ad_description}")
    persona.act()
```

## Configuration

Customize behavior through `config.ini`:

```ini
[OpenAI]
API_TYPE=openai  # or "azure"
MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small

[Simulation]
MAX_SIMULATION_STEPS=100
ENABLE_PARALLEL_AGENTS=True

[Logging]
LOGLEVEL=INFO
```

Programmatic configuration override:

```python
from tinytroupe import config_manager

# Override settings at runtime
config_manager.update("cache_api_calls", True)
config_manager.update("action_generator_enable_quality_checks", True)
```

## Simulation Management

### Caching Simulation State

Save and restore simulation states to reduce costs:

```python
from tinytroupe import control

# Begin recording
control.begin("simulation.cache.json")

# Run simulation steps
world.run(5)

# Save checkpoint
control.checkpoint()

# Continue simulation
world.run(5)

# End recording
control.end()
```

### Extracting Results

Extract and analyze simulation results:

```python
from tinytroupe.extraction import ResultsExtractor

# Extract specific information from interactions
extractor = ResultsExtractor()
results = extractor.extract_results_from_agents(personas,
    extraction_objective="Summarize each person's opinion on the product"
)
```

## Best Practices

1. **Define detailed personas**: The more specific your persona definitions, the more realistic the behavior
2. **Use caching**: Enable API call caching to reduce costs during development
3. **Iterate gradually**: Start with simple interactions and gradually increase complexity
4. **Validate outputs**: Always review generated content before using it for decisions
5. **Use checkpoints**: Save simulation states at important milestones
6. **Leverage parallelization**: Generate multiple personas or run simulations in parallel when possible

## Important Notes

- **Research tool only**: TinyTroupe is for simulation and research, not direct decision-making
- **Review outputs**: AI-generated content may include unrealistic or inaccurate results
- **Responsible use**: Do not use for simulating sensitive situations or to deceive people
- **Cost awareness**: LLM API calls can be expensive; use caching strategies

## Additional Resources

- Official Repository: https://github.com/microsoft/TinyTroupe
- Paper (preprint): https://arxiv.org/abs/2507.09788
- Example Notebooks: Available in the `examples/` folder of the repository

## Quick Start Example

Here's a complete minimal example to get started:

```python
from tinytroupe.examples import create_lisa_the_data_scientist
from tinytroupe.agent import TinyPerson
from tinytroupe.environment import TinyWorld

# Create personas
lisa = create_lisa_the_data_scientist()
oscar = TinyPerson("Oscar")
oscar.define("age", 30)
oscar.define("occupation", {"title": "Architect", "organization": "Awesome Inc."})

# Create world
world = TinyWorld("Meeting Room", [lisa, oscar])
world.make_everyone_accessible()

# Simulate interaction
lisa.listen("Introduce yourself to Oscar")
world.run(3)

# Check results
print("Simulation complete!")
```

This guide provides the foundation for using TinyTroupe in your persona simulation projects. For more advanced features and detailed examples, refer to the official documentation and example notebooks.
