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
- `.github/workflows/jekyll-gh-pages.yml` publishes the repo to GitHub Pages on push to `main`; `config.yml` is an unmodified starter CI workflow that does nothing meaningful.

## deliverable/project_1 — credibility-score chatbot

Streamlit + Anthropic reference app, managed with **uv** (`pyproject.toml` + `uv.lock`, Python ≥3.9).

```bash
cd deliverable/project_1
uv sync
cp .env.example .env          # fill in keys
uv run streamlit run main.py
```

`main.py` is a single-file app: a Streamlit chat UI, optional SerpAPI web search injected into the last user message as extra context, and `get_claude_response()` wrapped in Langfuse's `@observe()` decorator with `update_current_trace()` supplying user/session/tag metadata. Required env vars are listed in `.env.example` (`ANTHROPIC_API_KEY`, `SERPAPI_API_KEY`, and the three `LANGFUSE_*` values). Claude is called with the server-side `web_search_20250305` tool in addition to SerpAPI.

The credibility-score function itself is specified in `docs/12_capstone.md`, not implemented here: it takes a URL and must return `{"score": float, "explanation": string}`.

## deliverable/project_2 — TinyTroupe simulation

`tinytroupe_usage_guide.md` is the student-facing guide (TinyPerson / TinyWorld, JSON persona specs, `listen_and_act`). `tinytroupeproj/` is a **vendored clone of microsoft/TinyTroupe** that is untracked in git — do not commit it, and do not edit files under `tinytroupeproj/tinytroupe/` as if they were course code. Only `main.py` and `pyproject.toml` at that directory's root are project-local. TinyTroupe needs `OPENAI_API_KEY` (or the Azure pair) and Python ≥3.10.

## Conventions

- Docs are the product. Prose is deliberately verbose and explanatory — match that register rather than condensing when editing lecture notes or project specs.
- Never commit `.env` files; `deliverable/project_1/.gitignore` covers its own.
- Python apps here use `uv`, not pip/requirements.txt — but note `docs/13_final_guidance.md` asks *students* for a `requirements.txt`, so don't "fix" that guidance to match the reference apps.
