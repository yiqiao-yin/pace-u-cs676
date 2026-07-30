"""
main.py — entry point.

    uv run main.py              # talk to Claude (needs ANTHROPIC_API_KEY)
    uv run main.py --offline    # scripted replies, no key, no cost

All the work happens in the `personaforge` package under `src/`. This file
exists so there is an obvious thing to run.
"""

import sys

from personaforge.cli import main

if __name__ == "__main__":
    sys.exit(main())
