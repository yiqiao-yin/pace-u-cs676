# Project 2 — PersonaForge: An Agent-to-Agent Package

**CS676 Algorithms for Data Science · Pace University**

> **Weight: 30% of your course grade · 100 points · +5% bonus for a live Hugging Face deployment**
>
> **One deliverable, one deadline.** Due date: **[DEADLINES.md](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/DEADLINES.md)** — the only place dates live. The parts below are where the marks are, not a schedule.

Project 1 gave you a function to improve inside someone else's app. This one is
different: **you are building a Python package.** You get a skeleton that runs, and
your job is to turn it into something worth installing.

---

## Table of Contents

- [What it does](#what-it-does)
- [⚠️ You need your own Anthropic API key](#-you-need-your-own-anthropic-api-key)
- [Setup](#setup) — [macOS / Linux](#macos--linux) · [Windows](#windows)
- [Run it](#run-it)
- [How it works](#how-it-works)
- [Project layout](#project-layout)
- [Running the tests](#running-the-tests)
- [What to build](#what-to-build)
- [Deliverables and grading](#deliverables-and-grading)
- [Bonus: deploy to Hugging Face (+5%)](#bonus-deploy-to-hugging-face-5)
- [Troubleshooting](#troubleshooting)

---

## What it does

You talk to an agent in your terminal. You ask it to invent characters. Then you make
those characters talk to each other and watch what happens.

```
you › create a persona patient with chronic back pain who distrusts doctors

stage › Created Maria Delgado (patient) — 58-year-old with chronic lumbar pain
  · wrote temp/maria-delgado.md

you › create a persona doctor who is direct and running forty minutes late

stage › Created Dr. Samuel Reyes (doctor) — overbooked internist, blunt bedside manner
  · wrote temp/dr-samuel-reyes.md

you › have them talk about the MRI results

── conversation: the mri results ──

Maria Delgado: I've been waiting three weeks for someone to tell me what this means.
Dr. Samuel Reyes: I know, and I'm sorry about that. Let me pull it up now.
...
```

Each persona is **a markdown file on disk**. Open `temp/maria-delgado.md` in an editor,
change a line, and she behaves differently next run. That is the whole architecture:
a persona is a file, an agent is that file plus a model, a conversation is agents
taking turns.

---

## ⚠️ You need your own Anthropic API key

**Personas and conversations do not work without one.** Get a key at
[console.anthropic.com](https://console.anthropic.com/), then copy `.env.example` to
`.env` and paste it in. The key is yours and **the calls are billed to your account**.

Budget for it: **one API call per conversation turn**, plus one per persona you
create. A six-turn conversation between two personas you just made is eight calls. Use
`--model claude-haiku-4-5` while iterating.

You can build almost all of this project without spending anything:

| Works with no key | Needs your key |
| --- | --- |
| `uv run pytest` — all 39 tests | `uv run main.py` (no flag) |
| `uv run main.py --offline` — the entire app | real personas written by Claude |
| every module you are asked to extend | real agent-to-agent dialogue |

`--offline` swaps in a scripted model, so you can create personas, list them, and run
conversations for free. Use it for everything except prompt quality.

### A bug worth knowing about

The live path has been run end to end — personas authored, saved, and holding character
across a multi-turn conversation. One bug surfaced doing it, and it is instructive.

**The model intermittently omits the closing `---` of the frontmatter.** Roughly one
persona in three. The parser required it, so the whole document fell through to the
fallback and you got a character called `Unnamed` with `role: unknown` — no error, no
warning, just a broken persona.

The parser now tolerates it. But notice the shape of the failure: the model was not
wrong in any way a human would care about, the output was perfectly readable, and the
code was strict about something that did not matter. **You will hit this class of bug
constantly when a model is producing structured text.** Decide what your code should be
strict about, and be generous about everything else.

---

## Setup

You need **Python 3.10+**, **git**, and **[uv](https://docs.astral.sh/uv/)**.

This project is `uv`-native. If you do not have uv:

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### macOS / Linux

```bash
git clone https://github.com/yiqiao-yin/pace-u-cs676.git
cd pace-u-cs676/deliverable/project_2

uv sync                 # creates .venv and installs everything, including the package itself
cp .env.example .env
nano .env               # add your ANTHROPIC_API_KEY
```

### Windows

Use **PowerShell**.

```powershell
git clone https://github.com/yiqiao-yin/pace-u-cs676.git
cd pace-u-cs676\deliverable\project_2

uv sync
copy .env.example .env
notepad .env
```

You never need to activate the virtual environment manually — `uv run` handles it.

---

## Run it

**Start here. This needs no API key and costs nothing:**

```bash
uv run main.py --offline
```

Offline mode swaps the real model for a scripted one that replays canned dialogue. The
whole app works — you can create personas, list them, and run a conversation — you just
get fixed replies. It exists so you can see the shape of the thing on day one.

**Then with a real model:**

```bash
uv run main.py
```

Useful flags:

| Flag | Effect |
|---|---|
| `--offline` | scripted replies, no API key, no cost |
| `--model claude-haiku-4-5` | cheaper model while developing |
| `--persona-dir mydir` | store personas somewhere other than `temp/` |

Inside the app: `/help`, `/clear`, `/quit`.

---

## How it works

Four ideas, one per module.

**A persona is a markdown file** (`persona.py`). Frontmatter for the fields the code
indexes on, markdown body for everything the model reads:

```markdown
---
name: Maria Delgado
role: patient
summary: 58-year-old with chronic lumbar pain
---

# Maria Delgado

## Background
Retired schoolteacher, diagnosed eight years ago...

## How they speak
- Long sentences, lots of context before the point
```

**An agent is that file plus a model** (`agent.py`). `PersonaAgent.system_prompt()`
wraps the markdown body in in-character instructions. Nothing else distinguishes the
doctor from the patient — same class, same model, different file.

**A conversation is agents taking turns** (`conversation.py`). Each agent sees the
transcript rewritten from its own point of view: its lines as `assistant`, everyone
else's as `user`. That is what makes it feel like a participant rather than something
reading a script.

**The orchestrator decides what you meant** (`orchestrator.py`). You type "I want to
create a persona patient" and something maps that to `create_persona("patient")`.
Right now that something is **a pile of regular expressions**. See
[What to build](#what-to-build).

`llm.py` sits under all of it with two implementations — `ClaudeLLM` and `ScriptedLLM`
— behind one `complete()` method. That split is why the tests are fast and free.

---

## Project layout

```
project_2/
├── main.py                     entry point — thin wrapper
├── pyproject.toml              uv project, src layout, deps, pytest config
├── src/personaforge/           THE PACKAGE — your work goes here
│   ├── __init__.py             public API
│   ├── llm.py                  ClaudeLLM / ScriptedLLM behind one protocol
│   ├── persona.py              PersonaSpec, generate/save/load/find
│   ├── agent.py                markdown + model = a mind
│   ├── conversation.py         the turn loop
│   ├── orchestrator.py         intent routing  ← the weakest part
│   ├── ui.py                   terminal rendering (rich)
│   └── cli.py                  the REPL
├── tests/                      pytest, all offline
│   ├── test_llm.py
│   ├── test_persona.py
│   ├── test_agent.py
│   └── test_conversation.py
└── temp/                       personas and transcripts land here (gitignored)
```

Every module ends with a **YOUR TASK** comment block listing what is wrong with it.
Those lists are the assignment. Read all six before you plan your work.

---

## Running the tests

```bash
uv run pytest              # all of them
uv run pytest -v           # with names
uv run pytest tests/test_persona.py
```

**39 tests, no API key, well under a second.** They pass on a fresh clone — if they do not,
your environment is wrong, not your code.

They run offline because every test injects `ScriptedLLM` instead of calling Claude.
When you add features, add tests the same way. **A pull request full of code and no
tests is not a finished pull request**, and that principle is worth points here.

---

## What to build

The skeleton runs end to end and is deliberately shallow. Roughly ordered by payoff:

### 1. Replace the regex router with real tool use

`orchestrator.handle()` is a stack of `re.search` calls. It cannot ask a clarifying
question, cannot handle "make it two doctors and a nurse and have them argue", and
breaks the moment you phrase something a way the author didn't anticipate.

A real agent hands Claude a set of tools and lets the model choose. The tool schemas
you need are sketched in a comment at the bottom of `orchestrator.py`. This single
change turns a command parser into an agent, and it is the most valuable thing you can
do in this project.

### 2. Give agents memory

Agents are stateless between conversations. Run the same doctor twice and they will not
remember the first patient. Where should that memory live — appended back into the
persona's own `.md` file, or somewhere new? Whichever you choose, defend it.

### 3. Make turn-taking interesting

Right now it is strict round-robin for a fixed number of turns. Real conversations are
not polite. Who decides who speaks next? Can an agent stay silent? Can it end the
conversation because the question got answered?

### 4. Add an observer

Nobody watches the conversation. A third agent could summarize it, score whether the
doctor actually answered the patient's question, or flag when someone broke character.
This is the natural bridge to the AI-judge requirement in
[`docs/14_final_guidance.md`](../../docs/14_final_guidance.md).

### 5. Make it a package someone else could use

It is called a package, so treat it like one: a clean public API in `__init__.py`,
docstrings that explain intent, errors that say what to do next, and a version number
that means something. Someone should be able to `import personaforge` and build
something you did not anticipate.

**You do not have to do all five.** A depth-first project that does one of these
properly beats five half-finished ones. Choose, and justify the choice in your report.

---

## Deliverables and grading

**100 points total = 30% of your course grade**, plus a **+5% bonus** for a Hugging
Face deployment.

**This project has one deliverable and one deadline.** You submit once, through the
course form. The date is in [DEADLINES.md](https://github.com/yiqiao-yin/pace-u-cs676/blob/main/DEADLINES.md) — the only place deadlines live.

The three parts below are **where the marks are**, not a schedule. Work through them in
any order, or all at once — nothing is graded on when you did which piece.

### Part 1 — Working package (25 points)

| | Points |
|---|---|
| `uv sync && uv run main.py` works on a clean clone | 6 |
| At least one substantive extension beyond the skeleton | 9 |
| Your own tests for what you added, passing | 6 |
| Package structure stays coherent — no logic dumped into `main.py` | 4 |

### Part 2 — Beta version and technical report (35 points)

| | Points |
|---|---|
| Design write-up: what you changed and why | 10 |
| A saved transcript your system produced, with commentary on what worked | 8 |
| Honest failure analysis — where agents break character, lose the thread, or cost too much | 9 |
| Test coverage of the parts you own | 8 |

### Part 3 — Final container-ready app (40 points)

| | Points |
|---|---|
| Feature completeness against your own stated scope | 12 |
| Code quality: comments at course standard, clear naming, no dead code | 8 |
| Robustness — bad input, missing personas, API failures handled | 8 |
| Novelty, defended in the presentation | 7 |
| Live demo runs during your slot | 5 |

---

## Bonus: deploy to Hugging Face (+5%)

Worth **an extra 5% on your course grade**. The catch: this is a *terminal* app, and
Hugging Face Spaces serves web pages. You have two honest options.

1. **Wrap it in a web UI.** Add a Gradio or Streamlit front end that calls the same
   `personaforge` package. If your package boundaries are clean this is a small file —
   which is rather the point of building a package.
2. **Ship the package and a browser terminal.** Gradio can render a chat interface that
   drives the orchestrator directly.

Either way: create a Space, choose the matching SDK, add `ANTHROPIC_API_KEY` under
**Settings → Variables and secrets**, and submit the public URL.
**Never commit your key** — a key pushed to a public Space must be revoked immediately.

---

## Troubleshooting

**`uv: command not found`**
Install it with the command in [Setup](#setup), then restart your terminal.

**`ModuleNotFoundError: No module named 'personaforge'`**
Run through `uv run`, not a bare `python main.py`. `uv run main.py` puts the installed
package on the path; plain `python` does not.

**`RuntimeError: ANTHROPIC_API_KEY is not set`**
Either add the key to `.env`, or run `uv run main.py --offline`. The file must be named
exactly `.env` — Windows Notepad silently saves `.env.txt`.

**The app runs but every reply is the same canned line**
You are in offline mode. Drop `--offline`.

**Conversations cost more than expected**
Every turn is one API call, so a six-turn conversation is six calls plus one per
persona created. Use `--model claude-haiku-4-5` while developing, and `--offline` when
you are working on anything that isn't prompt quality.

**Personas from an old run keep showing up**
They are files. `rm temp/*.md` (macOS/Linux) or `del temp\*.md` (Windows).
