# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for **CS676 Algorithms for Data Science** (Pace University), authored by the instructor. It is primarily a *content* repo — Markdown lecture notes, Jupyter notebooks, and slide PDFs — plus a small number of reference applications students are asked to build on. There is no repo-wide build, test suite, or package; each app under `deliverable/` is its own self-contained project.

`main` is the only branch — the former `2025fall` semester branch was fast-forwarded into it and deleted. Work directly on `main` unless asked otherwise; pushing to it triggers the GitHub Pages deploy.

## Layout and how the pieces relate

- `README.md` — the syllabus and the canonical entry point. Every session links out to `docs/NN_topic.md`. When editing session content, keep the README's Session list, the file name, and the in-file Table of Contents in sync.
- `docs/01…11_*.md` — one file per lecture session. They follow a fixed convention: a Table of Contents at the top, `[Go back to TOC](#table-of-contents)` anchors under each heading, math in LaTeX, and figures referenced as `../pics/NN_topic_MM.png`. New images go in `pics/` using that same `NN_topic_MM` naming.
- `docs/12_capstone.md` — the spec that drives everything in `deliverable/`. It defines Projects 1–3, their required input/output contracts, per-deliverable deadlines, and grading rubrics. **Deadlines live only here and in the README grading section** — when a date shifts, both must be updated (see commits `f063d38`, `ad0982f`).
- `docs/13_final_guidance.md` — final presentation rubric (front-end 10% / back-end 20% / API 30% / system design 40%).
- `notebooks/session_N/` — the coding half of each lecture. Notebooks are written for Google Colab (pip installs inline, no shared environment), so they are not expected to run against a repo-level venv.
- `deliverable/` — instructor reference implementations of the capstone projects.
- The site is a **Docusaurus 3** app rooted at the repo top level (`docusaurus.config.ts`, `sidebars.ts`, `src/`, `static/`), reading `docs/` in place — which is why the `../pics/` image references in the notes resolve. `.github/workflows/deploy-docusaurus.yml` builds and publishes it to GitHub Pages on push to `main`; `config.yml` is an unmodified starter CI workflow that does nothing meaningful. Doc URLs drop the numeric filename prefix: `docs/01_introduction.md` serves at `/docs/introduction`.

## deliverable/project_1 — credibility-score chatbot

Streamlit + Anthropic starter kit, managed with **uv** (`pyproject.toml` + `uv.lock`) or plain `pip` (`requirements.txt`). Python ≥3.10; **`anthropic>=0.120`** is required — 0.69 predates the `output_config` parameter `credibility.py` depends on and fails at runtime.

```bash
cd deliverable/project_1
uv sync
cp .env.example .env          # fill in keys
uv run streamlit run main.py
```

`credibility.py` is the file students are graded on and the only one that matters here. It implements `score_url(url) -> {"score": float, "explanation": str}` in two layers — rule-based URL inspection that needs no API key, plus an optional Claude judgment blended at `RULE_WEIGHT`. **The baseline is deliberately weak**, and the twelve defects listed in its `KNOWN WEAKNESSES` block are the assignment. Do not "improve" it unprompted; that removes the exercise.

`main.py` is the Streamlit app: chat UI, optional SerpAPI search, Claude with the server-side `web_search_20260209` tool, citation extraction, and a colour-coded credibility chip per source. Langfuse tracing is optional and degrades to a no-op decorator when the keys are absent.

`evaluate.py` scores 24 labelled URLs (MAE / band accuracy / worst error; baseline **0.142 / 66.7% / 0.410**) and `test_credibility.py` holds 21 contract tests. **Both run with no API key** — that offline path is a design requirement, not an accident, so keep it working.

## deliverable/project_2 — PersonaForge

A `uv` project with a **src layout** and an installable package (`src/personaforge/`), plus `tests/`. Run it with `uv run main.py` or `uv run main.py --offline`; test with `uv run pytest`.

The architecture is one idea repeated: **a persona is a markdown file** (`temp/*.md`, frontmatter + body), **an agent is that file plus a model** (`agent.py` makes the body the system prompt), **a conversation is agents taking turns** (`conversation.py`). `llm.py` puts `ClaudeLLM` and offline stubs behind one `complete()` protocol — that seam is why all **39 tests run offline in under a second**, and why `--offline` works with no API key. Preserve it.

`orchestrator.py` routes intent with regular expressions and is **deliberately the weakest module**; replacing it with Claude tool use is the headline student task, and the schemas are sketched in place. Every module ends with a `YOUR TASK` block — those lists are the assignment, so don't quietly fix the items.

`tinytroupe_usage_guide.md` remains as background on the Microsoft library that inspired the project. `tinytroupeproj/` is a **vendored clone of microsoft/TinyTroupe**, gitignored — do not commit or edit it.

## Conventions

- Docs are the product. Prose is deliberately verbose and explanatory — match that register rather than condensing when editing lecture notes or project specs.
- Never commit `.env` files. The root `.gitignore` covers `.env`/`.env.*` at any depth with an `!.env.example` negation, plus private keys and runtime output. Note `*.egg-info/` with a trailing slash does **not** match until the directory exists — use the slashless form.
- Both starter kits must keep working **without an API key** (project 1: rules + tests + evaluator; project 2: `--offline` + pytest). A student's first run should never require a key.
- Python apps here use `uv`. Project 1 also ships a `requirements.txt` for pip users, matching what `docs/13_final_guidance.md` asks students for.
- Grading lives in two places that must agree: the table in the root `README.md` (homework 10%, projects 30/30/30, +5% Hugging Face bonus on projects 1 and 2, letter scale) and the per-project rubrics in each `deliverable/*/README.md`. `docs/12_capstone.md` is authoritative for deadlines.
