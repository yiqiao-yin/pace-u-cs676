"""
orchestrator.py — the agent you talk to.

You type "I want to create a persona patient" and something has to decide that
you meant `create_persona("patient")`. That decision is intent routing, and it
happens here.

The baseline router is regular expressions. It works, it is free, it is
deterministic, and it is obviously not what a modern agent does — a real one
would hand Claude a set of tools and let the model choose. Replacing this router
with genuine tool use is the headline task of this project, and the tool schemas
you would need are sketched at the bottom of this file.

Anything the router does *not* recognize is passed to the model as ordinary
conversation, so the orchestrator still feels like something you are talking to
rather than a command prompt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional

from .agent import agent_from_spec
from .conversation import Conversation, Turn, run_conversation, save_transcript
from .llm import LLM
from .persona import (
    DEFAULT_PERSONA_DIR,
    PersonaSpec,
    find_persona,
    generate_persona,
    list_personas,
    save_persona,
)

_STAGE_MANAGER_SYSTEM = """You are the stage manager of a persona simulation workshop.

The user builds fictional characters and has them talk to each other. You help them
decide who to create and what those characters should discuss.

Be brief — two or three sentences. When the user seems ready to act, remind them they
can say things like "create a persona doctor", "list personas", or "have them talk
about the test results"."""


@dataclass
class Result:
    """What the orchestrator did, in a form the UI can render."""

    reply: str
    events: List[str] = field(default_factory=list)
    conversation: Optional[Conversation] = None


class Orchestrator:
    """Routes what the user typed to the right part of the package."""

    def __init__(self, llm: LLM, persona_dir: Path = DEFAULT_PERSONA_DIR) -> None:
        self.llm = llm
        self.persona_dir = Path(persona_dir)
        self.history: List[dict] = []

    # -- intents ------------------------------------------------------------

    def _create_persona(self, description: str) -> Result:
        """Author a persona, write it to disk, and report where it landed."""
        if not description.strip():
            return Result(reply="Tell me what kind of persona — for example, 'create a persona doctor'.")

        spec = generate_persona(description, self.llm)
        path = save_persona(spec, self.persona_dir)
        return Result(
            reply=f"Created **{spec.name}** ({spec.role}) — {spec.summary}",
            events=[f"wrote {path}"],
        )

    def _list_personas(self) -> Result:
        """Show what is currently in the persona folder."""
        specs = list_personas(self.persona_dir)
        if not specs:
            return Result(reply="No personas yet. Try: create a persona patient")

        lines = [f"- **{s.name}** ({s.role}) — {s.summary}" for s in specs]
        return Result(reply=f"{len(specs)} persona(s) in `{self.persona_dir}`:\n" + "\n".join(lines))

    def _converse(
        self,
        topic: str,
        who: Optional[List[str]] = None,
        turns: int = 6,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """Wake up two personas and let them talk."""
        specs: List[PersonaSpec] = []

        if who:
            for name in who:
                found = find_persona(name, self.persona_dir)
                if found is None:
                    return Result(reply=f"I couldn't find a persona matching '{name}'.")
                specs.append(found)
        else:
            specs = list_personas(self.persona_dir)[:2]

        if len(specs) < 2:
            return Result(reply="I need at least two personas before they can talk. Create another one first.")

        agents = [agent_from_spec(s, self.llm) for s in specs]
        if on_start is not None:
            on_start(topic)
        conversation = run_conversation(agents, topic=topic, turns=turns, on_turn=on_turn)
        path = save_transcript(conversation, self.persona_dir, filename="conversation.md")

        return Result(
            reply=f"{len(conversation.turns)} turns between {' and '.join(s.name for s in specs)}.",
            events=[f"wrote {path}"],
            conversation=conversation,
        )

    def _chat(self, text: str) -> Result:
        """Nothing matched, so just talk to the user."""
        self.history.append({"role": "user", "content": text})
        reply = self.llm.complete(_STAGE_MANAGER_SYSTEM, self.history, max_tokens=512)
        self.history.append({"role": "assistant", "content": reply})
        return Result(reply=reply)

    # -- routing ------------------------------------------------------------

    def handle(
        self,
        text: str,
        on_turn: Optional[Callable[[Turn], None]] = None,
        on_start: Optional[Callable[[str], None]] = None,
    ) -> Result:
        """
        Work out what the user meant and do it.

        Patterns are tried in order and the first match wins, so the ordering
        below is load-bearing — a fact that should make you suspicious of the
        whole approach.
        """
        lowered = text.strip().lower()

        # "create a persona doctor" / "make a persona who is a nurse"
        create = re.search(
            r"(?:create|make|add|build|generate)\s+(?:a\s+|an\s+|new\s+)*persona\s*(?:for|of|who is|that is|:)?\s*(.*)",
            lowered,
        )
        if create:
            return self._create_persona(create.group(1).strip())

        # "list personas" / "who do I have"
        if re.search(r"\b(list|show|what|who)\b.*\bpersona", lowered) or lowered in {"list", "ls"}:
            return self._list_personas()

        # "have them talk about X" / "start a conversation between A and B"
        converse = re.search(
            r"\b(?:talk|conversation|converse|chat|discuss|speak)\b",
            lowered,
        )
        if converse:
            topic_match = re.search(r"\babout\s+(.*)$", lowered)
            topic = topic_match.group(1).strip() if topic_match else "an initial consultation"

            who_match = re.search(r"\bbetween\s+(.*?)(?:\s+about\b|$)", lowered)
            who = None
            if who_match:
                who = [w.strip() for w in re.split(r"\s+and\s+|,", who_match.group(1)) if w.strip()]

            return self._converse(topic=topic, who=who, on_turn=on_turn, on_start=on_start)

        return self._chat(text)


# =============================================================================
# YOUR TASK
# =============================================================================
#
#  1. REPLACE THE REGEX ROUTER WITH REAL TOOL USE. This is the main event.
#     Claude supports tool calling: you describe the tools, the model picks one
#     and fills in the arguments. Roughly:
#
#         tools = [
#             {
#                 "name": "create_persona",
#                 "description": "Create and save a new persona character.",
#                 "input_schema": {
#                     "type": "object",
#                     "properties": {
#                         "description": {"type": "string",
#                                         "description": "What kind of person to create"},
#                     },
#                     "required": ["description"],
#                 },
#             },
#             {"name": "list_personas", ...},
#             {"name": "start_conversation", ...},   # participants, topic, turns
#         ]
#
#     Then loop: send the message with `tools=`, and while
#     `response.stop_reason == "tool_use"`, run the requested tool, append a
#     `tool_result` block, and send it back. The model handles "actually, make
#     it two doctors and a nurse, and have them argue about the diagnosis" —
#     which the regexes below will never manage.
#
#  2. The router cannot ask a clarifying question. "Create a persona" with no
#     description gets a canned string instead of "what kind of person?".
#
#  3. Pattern order decides meaning. "list the personas then have them talk"
#     only ever lists.
#
#  4. `_converse` silently picks the first two personas when you do not name
#     anyone, even if you have six.
#
#  5. `turns` is hardcoded to 6 and cannot be set from the conversation.
#
#  6. `self.history` grows without limit and is only used by `_chat`, so the
#     orchestrator has no memory of the personas it created.
