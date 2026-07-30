"""
cli.py — the REPL loop.

`main.py` at the project root is a two-line wrapper around `main()` here, so the
app can be started either way:

    uv run main.py
    uv run personaforge
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from . import ui
from .llm import default_llm
from .orchestrator import Orchestrator
from .persona import DEFAULT_PERSONA_DIR


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PersonaForge — agent-to-agent persona simulation.")
    parser.add_argument(
        "--offline",
        action="store_true",
        help="use scripted replies instead of calling Claude (no API key needed)",
    )
    parser.add_argument(
        "--persona-dir",
        default=str(DEFAULT_PERSONA_DIR),
        help=f"where persona .md files are stored (default: {DEFAULT_PERSONA_DIR})",
    )
    parser.add_argument("--model", default=None, help="override the Claude model id")
    return parser


def main() -> int:
    """Run the terminal app until the user quits. Returns a process exit code."""
    args = build_parser().parse_args()
    load_dotenv()

    llm = default_llm(offline=args.offline, model=args.model)
    orchestrator = Orchestrator(llm=llm, persona_dir=Path(args.persona_dir))

    ui.banner(offline=args.offline, persona_dir=args.persona_dir)

    while True:
        try:
            text = ui.user_prompt()
        except (EOFError, KeyboardInterrupt):
            ui.console.print("\n[dim]bye[/dim]")
            return 0

        if not text:
            continue

        # Slash commands are handled here rather than in the orchestrator: they
        # are about the session, not about personas.
        if text in {"/quit", "/exit", "/q"}:
            ui.console.print("[dim]bye[/dim]")
            return 0
        if text == "/help":
            ui.help_text()
            continue
        if text == "/clear":
            ui.console.clear()
            continue

        # Dialogue is printed as each turn arrives rather than in one block at
        # the end, so a six-turn conversation feels like a conversation.
        def on_turn(turn):
            ui.dialogue_line(turn.speaker, turn.text)

        try:
            result = orchestrator.handle(text, on_turn=on_turn, on_start=ui.conversation_header)
        except Exception as exc:  # noqa: BLE001 — a crashed turn shouldn't kill the session
            ui.error(f"{type(exc).__name__}: {exc}")
            continue

        ui.agent_says(result.reply)
        for message in result.events:
            ui.event(message)

    return 0
