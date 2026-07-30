"""
agent.py — markdown file + model = an agent with its own mind.

This is the smallest interesting idea in the package. A `PersonaAgent` holds a
`PersonaSpec` (which came from a `.md` file) and an `LLM`. When you ask it to
speak, it builds a system prompt out of that markdown and sends the transcript
so far. The persona file *is* the agent's brain — nothing else distinguishes
the doctor from the patient.

Because the transcript is passed in rather than stored, agents are stateless
between turns. That is a design decision with consequences; see YOUR TASK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .llm import LLM
from .persona import PersonaSpec

# Wrapped around the persona markdown to put the model in character and keep it
# from narrating. Without the last two rules models tend to write stage
# directions and answer for everyone in the room.
_IN_CHARACTER_TEMPLATE = """You are role-playing a character in a simulated conversation.

Everything between the markers below defines who you are. Inhabit it completely:
speak in this person's voice, hold their opinions, want what they want.

=== BEGIN CHARACTER ===
{persona}
=== END CHARACTER ===

Rules:
- Speak only as {name}. Never speak for anyone else.
- Reply with dialogue only. No stage directions, no asterisks, no narration.
- Stay in character even if the conversation goes somewhere unexpected.
- Keep replies to a few sentences unless the moment genuinely calls for more.
- You are {name}. Do not mention that you are an AI or a simulation."""


@dataclass
class PersonaAgent:
    """One character in a conversation."""

    spec: PersonaSpec
    llm: LLM

    @property
    def name(self) -> str:
        return self.spec.name

    def system_prompt(self) -> str:
        """The persona markdown, wrapped in the in-character instructions."""
        return _IN_CHARACTER_TEMPLATE.format(persona=self.spec.body, name=self.spec.name)

    def respond(self, transcript: List["Turn"], max_tokens: int = 1024) -> str:  # noqa: F821
        """
        Produce this character's next line.

        The transcript is rewritten from this agent's point of view: its own past
        lines become `assistant` turns and everyone else's become `user` turns.
        That is what makes the model feel like a participant rather than an
        observer reading a script.
        """
        messages: List[dict] = []
        for turn in transcript:
            if turn.speaker == self.name:
                messages.append({"role": "assistant", "content": turn.text})
            else:
                messages.append({"role": "user", "content": f"{turn.speaker}: {turn.text}"})

        # The API requires the first message to be from the user, and consecutive
        # same-role turns get merged, so an empty or assistant-first transcript
        # needs an opener.
        if not messages or messages[0]["role"] != "user":
            messages.insert(0, {"role": "user", "content": "(begin the conversation)"})

        return self.llm.complete(self.system_prompt(), messages, max_tokens=max_tokens)


def agent_from_spec(spec: PersonaSpec, llm: LLM) -> PersonaAgent:
    """Convenience constructor, so callers don't import the dataclass directly."""
    return PersonaAgent(spec=spec, llm=llm)


# =============================================================================
# YOUR TASK
# =============================================================================
#
#  1. Agents have no memory between conversations. Run the same doctor twice and
#     they will not remember the first patient. Where should that memory live —
#     back in the .md file, or somewhere new?
#
#  2. Every turn re-sends the entire transcript. A long conversation gets
#     expensive and eventually hits the context window.
#
#  3. `respond()` cannot decide to stay quiet, ask for a pause, or end the
#     conversation. Every agent speaks every time it is asked.
#
#  4. Nothing stops an agent breaking character, and nothing detects it when it
#     happens. What would a check look like?
#
#  5. There is no notion of what an agent knows privately versus what has been
#     said out loud. A doctor and a patient should not share a memory.
