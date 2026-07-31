"""
persona.py — a persona is a markdown file on disk.

That is the whole idea of this package. There is no persona database and no
in-memory registry. A persona is a `.md` file in `temp/`, and anything that can
read that file can wake the persona up. You can open one in a text editor,
change a sentence, and the agent behaves differently on the next run.

File format — frontmatter for the machine, markdown for the model:

    ---
    name: Maria Delgado
    role: patient
    summary: 58-year-old with poorly controlled type 2 diabetes
    ---

    # Maria Delgado

    ## Background
    ...

The frontmatter parser here is about fifteen lines and handles exactly the
three keys above. That is intentional — see YOUR TASK at the bottom.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .llm import LLM

# Where persona files live. Relative to wherever you launched the app.
DEFAULT_PERSONA_DIR = Path("temp")


@dataclass
class PersonaSpec:
    """One persona: the metadata we index on, plus the markdown that is its mind."""

    name: str
    role: str
    summary: str
    body: str  # markdown; becomes the agent's system prompt

    def slug(self) -> str:
        """Filesystem-safe stem, e.g. 'Maria Delgado' -> 'maria-delgado'."""
        s = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return s or "unnamed"

    def to_markdown(self) -> str:
        """Serialize to the frontmatter + body format shown above."""
        return (
            "---\n"
            f"name: {self.name}\n"
            f"role: {self.role}\n"
            f"summary: {self.summary}\n"
            "---\n\n"
            f"{self.body.strip()}\n"
        )

    @classmethod
    def from_markdown(cls, text: str) -> "PersonaSpec":
        """
        Parse a persona file back into a spec.

        Missing frontmatter is tolerated: we fall back to sensible defaults and
        treat the whole file as the body, so a hand-written persona still loads.
        """
        meta = {"name": "Unnamed", "role": "unknown", "summary": ""}
        text = text.strip()

        # Models sometimes wrap the whole document in a code fence.
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1] if lines[-1].strip().startswith("```") else lines[1:])
            text = text.strip()

        body = text

        if text.startswith("---"):
            rest = text[3:].lstrip("\n")
            closing = re.search(r"^---\s*$", rest, re.MULTILINE)
            if closing:
                # Well-formed: frontmatter is everything up to the closing fence.
                front, body = rest[:closing.start()], rest[closing.end():]
            else:
                # The closing '---' is missing, which the model does drop from
                # time to time. Rather than throw the whole document away, read
                # `key: value` lines from the top and treat the first line that
                # is not one as the start of the body.
                front_lines, body_lines, in_front = [], [], True
                for line in rest.splitlines():
                    if in_front and re.match(r"^[A-Za-z_][A-Za-z0-9_ -]*:", line):
                        front_lines.append(line)
                    elif in_front and not line.strip():
                        continue            # blank line inside the header block
                    else:
                        in_front = False
                        body_lines.append(line)
                front, body = "\n".join(front_lines), "\n".join(body_lines)

            for line in front.splitlines():
                if ":" in line:
                    key, _, value = line.partition(":")
                    key = key.strip().lower()
                    if key in meta:
                        meta[key] = value.strip()

        return cls(name=meta["name"], role=meta["role"], summary=meta["summary"], body=body.strip())


def save_persona(spec: PersonaSpec, directory: Path = DEFAULT_PERSONA_DIR) -> Path:
    """Write the persona to `<directory>/<slug>.md`, creating the folder if needed."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{spec.slug()}.md"
    path.write_text(spec.to_markdown(), encoding="utf-8")
    return path


def load_persona(path: Path) -> PersonaSpec:
    """Read one persona file from disk."""
    return PersonaSpec.from_markdown(Path(path).read_text(encoding="utf-8"))


def list_personas(directory: Path = DEFAULT_PERSONA_DIR) -> List[PersonaSpec]:
    """Every persona currently saved, sorted by filename. Empty if none yet."""
    directory = Path(directory)
    if not directory.exists():
        return []
    return [load_persona(p) for p in sorted(directory.glob("*.md"))]


def find_persona(name: str, directory: Path = DEFAULT_PERSONA_DIR) -> Optional[PersonaSpec]:
    """
    Look a persona up by name, slug, or role — case-insensitively.

    Users type "the doctor" or "maria", not exact filenames, so we accept a
    loose match. This is crude; see YOUR TASK.
    """
    needle = name.strip().lower()
    for spec in list_personas(directory):
        if needle in (spec.name.lower(), spec.slug(), spec.role.lower()):
            return spec
    for spec in list_personas(directory):
        if needle in spec.name.lower() or needle in spec.role.lower():
            return spec
    return None


# -----------------------------------------------------------------------------
# Generating a persona with the model
# -----------------------------------------------------------------------------

_AUTHOR_SYSTEM = """You write character specifications for a multi-agent simulation.

Given a short description, write a persona as a markdown document. Output ONLY the
markdown — no preamble, no code fences.

Use exactly this structure:

---
name: <a specific, realistic full name>
role: <one lowercase word, e.g. patient, doctor, teacher, customer>
summary: <one line, under 15 words>
---

# <name>

## Background
Two or three sentences. Be concrete and specific — age, situation, history.

## Personality
Three or four bullet points about temperament and outlook.

## How they speak
Two or three bullets on voice: vocabulary, sentence length, verbal habits.

## What they want
One or two sentences on their goal in a conversation.

## What they avoid
One or two sentences on what they will not say or do.

Write a specific individual, not a type. Give them a quirk. Avoid stereotypes,
and do not make them a spokesperson for a demographic."""


def generate_persona(description: str, llm: LLM) -> PersonaSpec:
    """
    Ask the model to author a persona from a one-line description.

    :param description: e.g. "a patient with type 2 diabetes who distrusts doctors"
    :param llm:         any LLM implementation (real or scripted)
    :return:            the parsed PersonaSpec — not yet saved to disk
    """
    markdown = llm.complete(
        system=_AUTHOR_SYSTEM,
        messages=[{"role": "user", "content": f"Write a persona for: {description}"}],
        max_tokens=2048,
    )

    spec = PersonaSpec.from_markdown(markdown)

    # If the model ignored the frontmatter format we still want something usable,
    # so fall back to the raw description rather than failing the whole command.
    if spec.name == "Unnamed":
        spec.name = description[:40].strip().title() or "Unnamed"
        spec.summary = description.strip()

    return spec


# =============================================================================
# YOUR TASK
# =============================================================================
#
#  1. `find_persona` matches on substrings, so "doc" could match a persona named
#     "Dorothy". Two personas with the same role are indistinguishable. Decide
#     what the right disambiguation behaviour is and implement it.
#
#  2. `save_persona` silently overwrites an existing file with the same slug.
#     Creating two patients named Maria loses the first one.
#
#  3. The frontmatter parser handles three fixed keys and no quoting. A summary
#     containing a colon will parse in a way you will not enjoy. It does now
#     tolerate a missing closing '---', which the model drops often enough that
#     the app looked broken without it — but that is one failure mode handled,
#     not a robust parser.
#
#  4. Nothing validates what the model returned. If it emits prose instead of the
#     requested structure, you get a persona whose body is an apology. Nothing
#     checks that `role` is one word, that `name` looks like a name, or that the
#     body has the sections the prompt asked for.
#
#  5. Personas cannot be edited from inside the app — only by opening the file.
#     Should "make the doctor more skeptical" be a command?
#
#  6. There is no notion of a persona changing during a conversation. Real agents
#     might learn something and update their own file. Should they?
