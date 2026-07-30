"""
ui.py — the terminal interface.

Thin on purpose. Everything here is presentation: no persona logic, no model
calls. If you want to swap the terminal for a web app later, this is the only
module you should have to replace.
"""

from __future__ import annotations

from typing import Dict

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

console = Console()

# Colours are assigned to speakers on first sight so each character keeps a
# consistent colour for the whole session.
_PALETTE = ["cyan", "magenta", "green", "yellow", "blue", "red"]
_assigned: Dict[str, str] = {}


def speaker_colour(name: str) -> str:
    """Stable colour for a speaker, assigned in order of first appearance."""
    if name not in _assigned:
        _assigned[name] = _PALETTE[len(_assigned) % len(_PALETTE)]
    return _assigned[name]


def banner(offline: bool, persona_dir: str) -> None:
    """Startup panel: what this is and how to drive it."""
    mode = "[yellow]OFFLINE[/yellow] (scripted replies, no API calls)" if offline else "[green]LIVE[/green] (Claude)"
    console.print(
        Panel(
            f"[bold]PersonaForge[/bold] — agent-to-agent persona simulation\n\n"
            f"Mode: {mode}\n"
            f"Personas: [dim]{persona_dir}/[/dim]\n\n"
            "[bold]Try:[/bold]\n"
            "  create a persona patient with chronic back pain\n"
            "  create a persona doctor who is direct and busy\n"
            "  list personas\n"
            "  have them talk about the test results\n\n"
            "[dim]/help for commands · /quit to exit[/dim]",
            border_style="blue",
            title="CS676 Project 2",
        )
    )


def help_text() -> None:
    """Command reference."""
    console.print(
        Panel(
            "[bold]Say things like[/bold]\n"
            "  create a persona <description>\n"
            "  list personas\n"
            "  have them talk about <topic>\n"
            "  start a conversation between <name> and <name> about <topic>\n\n"
            "[bold]Slash commands[/bold]\n"
            "  /help    this message\n"
            "  /clear   clear the screen\n"
            "  /quit    exit\n\n"
            "Anything else is treated as ordinary conversation with the stage manager.",
            border_style="dim",
            title="help",
        )
    )


def user_prompt() -> str:
    """Read one line from the user."""
    return console.input("\n[bold blue]you[/bold blue] › ").strip()


def agent_says(text: str) -> None:
    """The orchestrator speaking to the user."""
    console.print("\n[bold blue]stage[/bold blue] › ", end="")
    console.print(Markdown(text))


def dialogue_line(speaker: str, text: str) -> None:
    """One line of persona-to-persona dialogue, printed as it arrives."""
    colour = speaker_colour(speaker)
    line = Text()
    line.append(f"{speaker}: ", style=f"bold {colour}")
    line.append(text)
    console.print(line)


def event(message: str) -> None:
    """Side effects worth surfacing — files written, mostly."""
    console.print(f"  [dim]· {message}[/dim]")


def error(message: str) -> None:
    console.print(f"[bold red]error[/bold red] {message}")


def conversation_header(topic: str) -> None:
    console.print(f"\n[dim]── conversation: {topic} ──[/dim]\n")
