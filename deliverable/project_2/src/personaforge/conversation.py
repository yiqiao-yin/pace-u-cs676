"""
conversation.py — put two or more agents in a room and let them talk.

The loop is deliberately dumb: agents take strict turns in the order given, for
a fixed number of turns, and then it stops. No agent decides who speaks next and
no agent can interrupt. Making that smarter is one of the more interesting
things you can do with this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .agent import PersonaAgent


@dataclass
class Turn:
    """One line of dialogue."""

    speaker: str
    text: str


@dataclass
class Conversation:
    """A completed dialogue, plus the setup that produced it."""

    topic: str
    participants: List[str]
    turns: List[Turn] = field(default_factory=list)

    def transcript(self) -> str:
        """Render as plain text, suitable for saving or printing."""
        header = f"# Conversation: {self.topic}\n\nParticipants: {', '.join(self.participants)}\n\n"
        return header + "\n\n".join(f"**{t.speaker}:** {t.text}" for t in self.turns)


def run_conversation(
    agents: List[PersonaAgent],
    topic: str,
    turns: int = 6,
    on_turn: Optional[Callable[[Turn], None]] = None,
) -> Conversation:
    """
    Run a round-robin conversation between the given agents.

    :param agents:  two or more PersonaAgents. They speak in list order.
    :param topic:   seeds the opening; each agent sees it as the situation.
    :param turns:   total lines of dialogue, split across the agents.
    :param on_turn: called after each line, so a UI can print as it goes rather
                    than waiting for the whole conversation to finish.
    """
    if len(agents) < 2:
        raise ValueError("a conversation needs at least two agents")

    conversation = Conversation(topic=topic, participants=[a.name for a in agents])

    # The topic is injected as a scene-setting turn from a narrator. Every agent
    # sees it as context but nobody has to answer it directly.
    history: List[Turn] = [Turn(speaker="Scene", text=topic)]

    for i in range(turns):
        speaker = agents[i % len(agents)]
        text = speaker.respond(history)
        turn = Turn(speaker=speaker.name, text=text)

        history.append(turn)
        conversation.turns.append(turn)

        if on_turn is not None:
            on_turn(turn)

    return conversation


def save_transcript(conversation: Conversation, directory, filename: Optional[str] = None):
    """Write the transcript to disk next to the personas that produced it."""
    from pathlib import Path

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = filename or "conversation.md"
    path = directory / name
    path.write_text(conversation.transcript(), encoding="utf-8")
    return path


# =============================================================================
# YOUR TASK
# =============================================================================
#
#  1. Turn-taking is strict round-robin. Real conversations do not alternate
#     politely. Who should decide who speaks next — a rule, or a model?
#
#  2. Conversations end after N turns regardless of whether anything was
#     resolved. There is no notion of the discussion being finished.
#
#  3. The "Scene" turn is a hack: a fake speaker that exists only to inject the
#     topic. Is there a cleaner way to set a scene?
#
#  4. Nobody observes the conversation. A third agent could summarize it, judge
#     it, or score whether the doctor actually answered the patient's question —
#     which is the natural bridge to the AI-judge requirement in the final
#     presentation rubric.
#
#  5. `save_transcript` always writes to the same filename, so each conversation
#     overwrites the last one.
#
#  6. If one agent's reply fails, the whole conversation raises and everything
#     said so far is lost.
