# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

Course materials for **CS676 Algorithms for Data Science** (Pace University), authored by the instructor. It is mostly a *content* repo — Markdown lecture notes, Jupyter notebooks, slides — wrapped in a Docusaurus site, plus two starter-kit applications students extend and a set of from-scratch homework exercises.

`main` is the only branch. Work directly on it unless asked otherwise; pushing triggers the GitHub Pages deploy.

## Two published surfaces

| | Where | Built from |
| --- | --- | --- |
| Course site | <https://yiqiao-yin.github.io/pace-u-cs676/> | Docusaurus, `.github/workflows/deploy-docusaurus.yml`, on push to `main` |
| Slide deck | <https://main.d3j8dqgo1nf8ma.amplifyapp.com> | `tools/slide_deck/`, deployed to AWS Amplify **by hand** |

The site is a **Docusaurus 3** app rooted at the repo top level (`docusaurus.config.ts`, `sidebars.ts`, `src/`, `static/`) reading `docs/` in place — which is why `../pics/` references in the notes resolve. `npm run build` validates every internal link and anchor, so **run it after editing any doc**. Doc URLs drop the numeric filename prefix: `docs/01_introduction.md` serves at `/docs/introduction`, so renaming a heading changes an anchor but renaming a file changes a URL.

## Layout

- `README.md` — the syllabus and canonical entry point. Sessions link out to `docs/NN_topic.md`; five also link to a homework exercise.
- `docs/01…11_*.md` — one file per lecture session. Fixed convention: Table of Contents at the top, `[Go back to TOC](#table-of-contents)` under each heading, LaTeX math, figures as `../pics/NN_topic_MM.png`. Five end with a `## Homework` section.
- `DEADLINES.md` — **the single source of truth for every date.** Homework and project deadlines live here and nowhere else. If you are asked to change a date, change this file only; do not reintroduce dates into the capstone spec or the project READMEs, which were deliberately stripped of them.
- `docs/12_capstone.md` — the spec driving `deliverable/`. Describes *what* to build, never *when*. **Projects 1 and 2 are one deliverable each**; their "Checkpoint 1/2/3" sections are a recommended order of work and a point breakdown for a single submission, not separate hand-ins. **Project 3 is the exception** — a Pass/Fail proposal and then the final project, two real submissions, so its section is headed "The Two Submissions" instead.
- `docs/13_final_guidance.md` — presentation rubric (front-end 10% / back-end 20% / API 30% / system design 40%).
- `notebooks/session_N/` — coding sessions, written for Google Colab (inline pip installs, no shared venv).
- `notebooks/homework/` — five from-scratch exercises. See below.
- `deliverable/` — the two starter kits.
- `tools/slide_deck/` — generator for the slide web app.

## notebooks/homework — the exercises

Five numpy-only scripts, each complete except for the core algorithm, which the student writes.

**Never edit the student scripts directly.** They are generated:

```bash
python notebooks/homework/make_homework.py
```

`answer/*_ans.py` are the real sources. Each marks its solution with `# BEGIN SOLUTION: description` / `# END SOLUTION`; the generator copies the file, drops the `_ans` suffix, and replaces each block with a `NotImplementedError` stub at the right indentation. Editing a student file by hand is silently undone on the next run.

**`notebooks/homework/answer/` is gitignored in this repo and must never be committed here.** Do not commit it, and do not paste solution code into any tracked file.

It *is* backed up, to a **private mirror** — a second git directory (`.git-full`) over this same working tree, pushed to `yiqiao-yin/pace-u-cs676-full`. Use the `./full` wrapper for it:

```bash
./full status
./full add -A && ./full commit -m "sync" && ./full push
```

`git ...` is the public repo, `./full ...` is the private one. The answer keys are force-added there, so the public `.gitignore` rule cannot hide them from the mirror — and cannot leak them into the public repo either. After changing an answer key, regenerate the student scripts, commit those publicly, and sync the mirror.

Blank counts are 2 / 2 / 2 / 1 / 3 (`01_lr`, `02_logreg`, `03_cv`, `04_tree`, `05_kmeans`). Every script grades itself — closed-form comparison, majority-class baseline, train-vs-validation gap, or monotonically falling inertia — and every answer key must keep printing `PASS`. The `# ┌─ YOUR TASK` boxes teach without giving code; keep that register if you add one.

## deliverable/project_1 — credibility-score chatbot

Streamlit + Anthropic. **uv** (`pyproject.toml` + `uv.lock`) or `pip` (`requirements.txt`). Python ≥3.10; **`anthropic>=0.120`** — 0.69 predates the `output_config` parameter `credibility.py` needs and fails at runtime.

```bash
cd deliverable/project_1
uv sync && cp .env.example .env
uv run streamlit run main.py
python test_credibility.py     # 21 tests, no key needed
python evaluate.py             # baseline MAE 0.142 / 66.7% / 0.410
```

`credibility.py` is the graded file: `score_url(url) -> {"score": float, "explanation": str}`, rules plus an optional Claude judgment blended at `RULE_WEIGHT`. **The baseline is deliberately weak** — the twelve entries in its `KNOWN WEAKNESSES` block are the assignment, so don't fix them unprompted.

`main.py` is the app: chat, optional SerpAPI, Claude with `web_search_20260209`, citation extraction, colour-coded chips. Langfuse is optional and degrades to a no-op decorator.

## deliverable/project_2 — PersonaForge

A `uv` project, **src layout**, installable package plus `tests/`.

```bash
cd deliverable/project_2
uv sync
uv run main.py --offline     # no key, no cost
uv run pytest                # 39 tests, under a second
```

One idea repeated: **a persona is a markdown file** (`temp/*.md`), **an agent is that file plus a model** (`agent.py` makes the body the system prompt), **a conversation is agents taking turns**. `llm.py` puts `ClaudeLLM` and the offline stubs behind one `complete()` protocol — that seam is why the tests are free and `--offline` works. Preserve it.

`orchestrator.py` routes intent with regexes and is **deliberately the weakest module**; replacing it with Claude tool use is the headline student task, with schemas sketched in place. Every module ends with a `YOUR TASK` block — the assignment, not a to-do list.

`tinytroupeproj/` is a vendored clone of microsoft/TinyTroupe, gitignored — do not commit or edit.

## tools/slide_deck — the slide web app

Renders the 330-page course PDF into one self-contained HTML presentation (~19 MB, base64 WebP page images). **Pages are rendered, not rebuilt from text, because the equations in the PDF are images** — the text layer for page 84 ends mid-sentence where the formula should be. Extracted text still drives titles, contents, and search.

```bash
pip install pypdfium2 pypdf pillow
python tools/slide_deck/build_deck.py out.html tools/slide_deck/deck_template.html
```

The output is **not committed** — it is a build artifact, ~30 seconds to regenerate. Redeploy instructions (zip → `aws amplify create-deployment` → upload → `start-deployment`) are in `tools/slide_deck/README.md`.

## Conventions and traps

- **Docs are the product.** Prose is deliberately verbose and explanatory — match that register rather than condensing.
- **Changing the slide PDF does not update the published deck.** `tools/slide_deck/` must be re-run and the result re-uploaded to Amplify by hand; there is no pipeline. The deck will silently serve the old slides otherwise.
- **Run `npm run build` after editing docs.** It fails the build on broken internal links and anchors, which is the only check that catches a renamed heading.
- **Both starter kits must work with no API key** (project 1: rules + tests + evaluator; project 2: `--offline` + pytest). A student's first run should never need one.
- **The live Claude paths have been verified** (2026-07-31) and two real bugs were fixed in the process: `ask_claude` read citations off text blocks, but `web_search_20260209` returns them in `web_search_tool_result` blocks and leaves `block.citations` as `None`, so the app silently showed zero sources; and `PersonaSpec.from_markdown` rejected persona documents where the model omitted the closing `---`, which it does roughly one time in three. Measured result for the scorer: MAE 0.142 → 0.086 and band accuracy 66.7% → 83.3% with the LLM layer on.
- **`deliverable/project_1/.env` holds a real API key. Do not spend it without asking.**
- **The private mirror has Actions disabled.** It inherits `.github/workflows/` from the shared working tree, and `configure-pages` fails there because the mirror has no Pages site. Actions are switched off at the repo level rather than by changing files, since both repos share one tree. If you add a workflow, it will not run on the mirror — that is intended.
- **Never commit `.env`.** Root `.gitignore` covers `.env`/`.env.*` at any depth with an `!.env.example` negation, plus private keys and runtime output. Note `*.egg-info/` with a trailing slash does **not** match until the directory exists — use the slashless form.
- **Students submit one GitHub repository for the whole course**, named to contain the course number, with the same URL given to the form weekly. Stated in `DEADLINES.md` and the syllabus; keep the two consistent.
- **Homework is Pass/Fail and all five count** — no dropped grades. The 10% covers the per-session form *and* the five exercises. The older "lowest two dropped" rule was removed and should not come back.
- **`clawdeck-app.com` is offered as optional compute** in `docs/01_introduction.md` and the Project 3 spec. It is the instructor's own product, so **every mention must disclose that and name Colab alongside it**. Do not add a mention elsewhere without the disclosure, and do not remove the disclosure from an existing one.
- **Grading lives in two places that must agree**: the table in the root `README.md` (homework 10%, projects 30/30/30, +5% Hugging Face bonus on projects 1 and 2, letter scale A at 95% down to C- at 60%) and the per-project rubrics in each `deliverable/*/README.md`. `docs/12_capstone.md` is authoritative for deadlines.
- Python apps use `uv`; project 1 also ships `requirements.txt` because `docs/13_final_guidance.md` asks students for one.
