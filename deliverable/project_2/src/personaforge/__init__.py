"""
PersonaForge — an agent-to-agent persona simulation package.

The idea in four lines:

    spec  = generate_persona("a patient with chronic back pain", llm)
    path  = save_persona(spec)              # writes temp/<name>.md
    agent = agent_from_spec(spec, llm)      # markdown + model = a mind
    run_conversation([doctor, patient], topic="the test results")

A persona is a markdown file. An agent is that file plus a model. A conversation
is agents taking turns. Everything else is plumbing.
"""

from .agent import PersonaAgent, agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM, ClaudeLLM, ScriptedLLM, default_llm
from .orchestrator import Orchestrator, Result
from .persona import (
    PersonaSpec,
    find_persona,
    generate_persona,
    list_personas,
    load_persona,
    save_persona,
)

__version__ = "0.1.0"

__all__ = [
    "PersonaAgent",
    "agent_from_spec",
    "Conversation",
    "Turn",
    "run_conversation",
    "save_transcript",
    "LLM",
    "ClaudeLLM",
    "ScriptedLLM",
    "default_llm",
    "Orchestrator",
    "Result",
    "PersonaSpec",
    "find_persona",
    "generate_persona",
    "list_personas",
    "load_persona",
    "save_persona",
]
