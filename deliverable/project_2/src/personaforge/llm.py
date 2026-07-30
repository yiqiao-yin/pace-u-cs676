"""
llm.py — the thin layer between a persona and a model.

Everything in this package that needs to "think" goes through the `LLM`
protocol below. There are two implementations:

    ClaudeLLM    real Claude calls. Needs ANTHROPIC_API_KEY.
    ScriptedLLM  returns canned strings. No key, no network, no cost.

That split is deliberate and it is the reason this project is testable. The
unit tests in `tests/` inject a ScriptedLLM, so they run in milliseconds for
free, and `main.py --offline` uses one too so you can see the whole app work
before you have configured anything.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

# Claude Opus 5 is the most capable model. While you are iterating you may want
# "claude-haiku-4-5" — a persona conversation makes one call per turn, and those
# add up. Say which model produced your submitted transcripts.
DEFAULT_MODEL = "claude-opus-5"


class LLM(Protocol):
    """Anything that can turn a system prompt plus a transcript into a reply."""

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        ...


@dataclass
class ClaudeLLM:
    """
    Real Claude. One method, deliberately — this package only ever needs to say
    "here is who you are, here is the conversation so far, what do you say next".

    The client is created lazily so importing this module never requires a key.
    """

    model: str = DEFAULT_MODEL
    _client: Any = field(default=None, repr=False)

    def _get_client(self) -> Any:
        if self._client is None:
            import anthropic

            if not os.getenv("ANTHROPIC_API_KEY"):
                raise RuntimeError(
                    "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key, "
                    "or run in offline mode: uv run main.py --offline"
                )
            self._client = anthropic.Anthropic()
        return self._client

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        """Send one request and return the concatenated text of the reply."""
        response = self._get_client().messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return "[declined to respond]"

        return "".join(block.text for block in response.content if block.type == "text").strip()


@dataclass
class ScriptedLLM:
    """
    A fake model that replays a fixed list of replies, then loops.

    Used by the tests and by `--offline`. It records every call it received in
    `.calls`, which makes it easy to assert that a persona's markdown actually
    reached the system prompt.
    """

    replies: List[str] = field(default_factory=lambda: ["(scripted reply)"])
    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


_DEMO_DIALOGUE = [
    "I understand. Could you tell me more about when the symptoms started?",
    "That has been going on for about three weeks now, mostly in the mornings.",
    "Have you noticed anything that makes it better or worse?",
    "Coffee seems to make it worse, and lying down helps a little.",
    "Thank you, that is useful. I would like to run a couple of tests.",
    "Whatever you think is best, doctor. I just want to feel normal again.",
]

_DEMO_PERSONA = """---
name: {name}
role: {role}
summary: a demo persona generated in offline mode
---

# {name}

## Background
This persona was produced by the offline stub, not by a model. Run without
`--offline` to get a real character written by Claude.

## Personality
- Placeholder
- Placeholder

## How they speak
- In whatever the scripted stub happens to return
"""


@dataclass
class DemoLLM:
    """
    The offline stub used by `--offline`.

    A single ScriptedLLM cannot serve both jobs: asked to author a persona it
    would return a line of dialogue, and you would get a character named
    "Unnamed". So this one looks at the system prompt and answers in the shape
    the caller expects — persona markdown for the author prompt, dialogue for
    everything else.
    """

    calls: List[Dict[str, Any]] = field(default_factory=list)

    def complete(self, system: str, messages: List[Dict[str, str]], max_tokens: int = 2048) -> str:
        self.calls.append({"system": system, "messages": list(messages)})
        request = messages[-1]["content"] if messages else ""

        # The persona-author prompt is the only one that asks for frontmatter.
        if "role: <one lowercase word" in system:
            description = request.replace("Write a persona for:", "").strip()
            words = [w for w in description.split() if w.isalpha()]
            role = words[0].lower() if words else "person"
            return _DEMO_PERSONA.format(name=description[:40].title() or "Demo Persona", role=role)

        if "stage manager" in system.lower():
            return "Offline mode — I'm a stub. Try: create a persona doctor"

        return _DEMO_DIALOGUE[(len(self.calls) - 1) % len(_DEMO_DIALOGUE)]


def default_llm(offline: bool = False, model: Optional[str] = None) -> LLM:
    """Pick an implementation. Central so you only change this in one place."""
    if offline:
        return DemoLLM()
    return ClaudeLLM(model=model or DEFAULT_MODEL)
