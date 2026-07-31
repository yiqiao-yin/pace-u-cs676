# Project 1 — Credibility Scoring for Sources

**CS676 Algorithms for Data Science · Pace University**

> **Weight: 30% of your course grade · 100 points · +5% bonus for a live Hugging Face deployment**

This is the first of the three projects that make up the bulk of your grade. You get
a working chatbot for free. What you build is the part that decides whether a source
can be trusted.

---

## Table of Contents

- [The problem](#the-problem)
- [What you are given](#what-you-are-given)
- [⚠️ You need your own Anthropic API key](#-you-need-your-own-anthropic-api-key)
- [Setup](#setup) — [macOS / Linux](#macos--linux) · [Windows](#windows)
- [Run it](#run-it)
- [The one function you are graded on](#the-one-function-you-are-graded-on)
- [Measuring your work](#measuring-your-work)
- [Ideas worth pursuing](#ideas-worth-pursuing)
- [Deliverables and grading](#deliverables-and-grading)
- [Bonus: deploy to Hugging Face (+5%)](#bonus-deploy-to-hugging-face-5)
- [Troubleshooting](#troubleshooting)

---

## The problem

A chatbot that answers from web sources is only as trustworthy as the sources it
picked. Ask it a medical question and it may cite *The New England Journal of
Medicine* — or a blog post written this morning by someone with a supplement to sell.
The answer looks identical either way.

Your task is to make that difference visible. Given a URL, return a credibility score
and an explanation a reader can act on:

```python
score_url("https://www.nature.com/articles/s41586-021-03819-2")
# {"score": 0.95, "explanation": "'nature.com' is a domain we recognize; served over HTTPS."}

score_url("https://randomblog.blogspot.com/2024/03/my-thoughts.html")
# {"score": 0.27, "explanation": "'blogspot.com' is a domain we recognize; served over HTTPS."}
```

Those two explanations are, frankly, not good enough. That is the point.

---

## What you are given

| File | What it is | Do you edit it? |
|---|---|---|
| **`credibility.py`** | **The scorer. Your work goes here.** | **Yes — this is the assignment** |
| `main.py` | Streamlit chat app; calls Claude, renders sources with coloured chips | Rarely |
| `evaluate.py` | Scores 24 labelled URLs and reports your error | Extend the label set |
| `test_credibility.py` | Contract tests — checks the output *shape*, not quality | Add your own cases |
| `.env.example` | Template for API keys | Copy to `.env` |
| `requirements.txt` | Dependencies for `pip` users | No |
| `pyproject.toml` | Dependencies for `uv` users | No |

The app already handles the Claude call, web search, citation extraction, session
state, and the UI. **None of that is what you are graded on.**

---

## ⚠️ You need your own Anthropic API key

**The chat does not work without one.** Get a key at
[console.anthropic.com](https://console.anthropic.com/), then copy `.env.example` to
`.env` and paste it in. The key is yours and **the calls are billed to your account** —
nobody else's.

You can do a large part of this assignment before spending anything:

| Works with no key | Needs your key |
| --- | --- |
| `python test_credibility.py` — 21 contract tests | the chat itself (`streamlit run main.py`) |
| `python evaluate.py` — the 24-URL harness | `python evaluate.py --llm` |
| the whole rule-based scoring layer | `credibility.llm_opinion()` |
| the sidebar URL scorer in the running app | |

That split is deliberate. Improving the rule layer, measuring it, and defending the
result is most of the grade, and none of it costs money.

### What the key buys you

Turning the LLM layer on is not decoration — it measurably improves the scorer:

| | MAE | Band accuracy | Worst error |
| --- | --- | --- | --- |
| `python evaluate.py` (rules only) | 0.142 | 66.7% | 0.410 |
| `python evaluate.py --llm` | **0.086** | **83.3%** | **0.230** |

Most of that gain is on the held-out domains the lookup table has never seen. A JAMA
article scores **0.52** on rules alone and **0.70** with the model, because the model
knows what JAMA is and a hand-written domain list does not.

That is also your warning: it is easy to "improve" the scorer by just calling the model
more. The report asks you to justify an *algorithm*, and "I asked Claude" is not one.

### A bug worth knowing about

The first live run of `ask_claude()` returned **zero sources**. The function read
citations off the text blocks, which is the obvious place — but `web_search_20260209`
returns them in `web_search_tool_result` blocks instead, and `block.citations` is
`None`. Nothing crashed and nothing was logged; the app just quietly showed no sources.

It is fixed. Keep the failure mode in mind: an API returning an empty list where you
expected data is far harder to notice than one that raises.

---

## Setup

You need **Python 3.10 or newer** and **git**. Check with `python --version`.

### macOS / Linux

```bash
# 1. Clone the course repo and enter this project
git clone https://github.com/yiqiao-yin/pace-u-cs676.git
cd pace-u-cs676/deliverable/project_1

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
cp .env.example .env
nano .env          # or: open -e .env
```

### Windows

Use **PowerShell** (not the old `cmd` prompt).

```powershell
# 1. Clone the course repo and enter this project
git clone https://github.com/yiqiao-yin/pace-u-cs676.git
cd pace-u-cs676\deliverable\project_1

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API key
copy .env.example .env
notepad .env
```

> **PowerShell blocks the activate script?** You'll see *"running scripts is disabled
> on this system."* Fix it once, for your user only:
> ```powershell
> Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
> ```

### Using `uv` instead (either platform)

If you have [uv](https://docs.astral.sh/uv/), it replaces steps 2 and 3:

```bash
uv sync
```

and prefix commands with `uv run` (e.g. `uv run streamlit run main.py`).

---

## Run it

**Verify your setup first — this needs no API key at all:**

```bash
python test_credibility.py     # expect: 21 passed, 0 failed
python evaluate.py             # expect: MAE 0.142, band accuracy 66.7%
```

Those two numbers are your **baseline**. Write them down. Your job is to improve them.

**Then start the app:**

```bash
streamlit run main.py
```

It opens at `http://localhost:8501`. The sidebar shows which keys it found and lets
you score any URL directly — useful for testing without burning chat tokens.

---

## The one function you are graded on

Open `credibility.py`. It is heavily commented and ends with a numbered
**KNOWN WEAKNESSES** list — twelve real defects in the code you have been handed.

The baseline works in two layers:

1. **Rules** — inspects the URL string. Runs with no API key, which is why the app
   works immediately after cloning.
2. **LLM** — one Claude call that judges the URL. Runs only when `ANTHROPIC_API_KEY`
   is set, and silently degrades to rules-only when it is not.

They are blended with a constant (`RULE_WEIGHT = 0.6`) that was never tested against
anything. That constant is one of the twelve defects.

**Do not change the contract.** `score_url()` must keep returning
`{"score": float in [0,1], "explanation": str}`. Everything downstream depends on it,
and `test_credibility.py` enforces it.

---

## Measuring your work

`evaluate.py` scores 24 labelled URLs and reports three numbers:

| Metric | Baseline | Meaning |
|---|---|---|
| Mean absolute error | **0.142** | Average distance from the expected score. Lower is better. |
| Band accuracy | **66.7%** | How often the HIGH/MEDIUM/LOW chip is the right colour. |
| Worst single error | **0.410** | Your most embarrassing miss. |

The label set is split deliberately. The first block uses domains that already appear
in the baseline's lookup table — it does well there, and that flatters it. The block
marked **HELD OUT** uses domains the table has never seen, and that is where the
baseline collapses: it scores a JAMA medical journal article at **0.52** because
`jamanetwork.com` is a `.com` and it knows nothing else.

A lookup table cannot generalize. Anything that can is worth more than a longer table.

**The labels are one instructor's judgment, not ground truth.** If you think a label
is wrong, argue with it in your report — a well-defended disagreement earns credit.

---

## Ideas worth pursuing

Roughly ordered by payoff. You are not expected to do all of these.

**Read the page.** The whole rule layer inspects a string. Fetching the page and
looking for a named author, a publication date, citations, or a corrections policy is
the single largest available improvement.

**Use real metadata instead of guesses.** [OpenAlex](https://openalex.org/) and
[Crossref](https://www.crossref.org/) both expose citation counts, venue, and
retraction status over free APIs with no key required. That turns "hybrid rule-based
and ML" from a phrase in the assignment into something you actually built.

**Learn the weights instead of picking them.** Every number in `DOMAIN_SCORES`,
`TLD_SCORES`, and `PATH_PENALTIES` was typed by hand. Fit them to labelled data with
a regression, and use lasso to report which features actually carry signal
(**Session 06**).

**Calibrate, don't just be accurate.** A 0.7 should mean "right about 70% of the
time." Right now it means "some numbers happened to add to 0.7." A reliability curve
or a Brier score tells you how wrong that is (**Session 05**).

**Quantify your uncertainty.** An unknown domain and a famous journal both return a
bare point estimate. Bootstrap a confidence interval and show it in the UI
(**Session 05**).

**Make the explanation carry its weight.** Right now it is rule names joined by
semicolons. It should tell a reader *why they should care*. Explanation quality is
graded separately from score accuracy.

---

## Deliverables and grading

**100 points total = 30% of your course grade**, plus a **+5% bonus** for a Hugging
Face deployment. Submit through the course form.

| Deliverable | Due (Friday) |
| --- | --- |
| 1 — Working function and tests | **Sept 25, 2026** |
| 2 — Technique report | **Oct 2, 2026** |
| 3 — Working integrated application | **Oct 9, 2026** |

The [capstone spec](../../docs/12_capstone.md) and the
[course README](../../README.md#grading-policy) are authoritative if anything here
disagrees with them.

### Deliverable 1 — Working function and tests (25 points)

| | Points |
|---|---|
| `score_url()` returns the correct contract for valid URLs | 8 |
| Malformed input handled without raising (`test_credibility.py` passes) | 5 |
| At least one substantive improvement over the baseline, clearly identified | 7 |
| Your own added test cases beyond the ones provided | 5 |

### Deliverable 2 — Technique report (35 points)

A written report, 4–8 pages.

| | Points |
|---|---|
| Description of your algorithm and why you chose it | 10 |
| Literature review — existing approaches to credibility assessment, cited | 8 |
| **Quantitative before/after** using `evaluate.py`, with a results table | 10 |
| Honest discussion of what still fails and why | 7 |

The measured comparison is the core of this deliverable. "It seems better" earns
nothing; "MAE fell from 0.142 to 0.081, driven mostly by the held-out block" earns
full marks.

### Deliverable 3 — Working integrated application (40 points)

| | Points |
|---|---|
| Scores display in the running app, legibly and without clutter | 10 |
| Scorer is robust — no crashes on dead links, timeouts, or odd URLs | 8 |
| Code comments: 3–5 explanatory lines per section (course standard) | 7 |
| Novelty — something beyond a longer lookup table, defended | 10 |
| Live demo runs during your presentation slot | 5 |

---

## Bonus: deploy to Hugging Face (+5%)

Getting the app running on a public URL is worth **an extra 5% on your course grade**.
It is genuinely more work, which is why it is worth points.

1. Create a **Space** at [huggingface.co/new-space](https://huggingface.co/new-space).
   Choose the **Streamlit** SDK and the free CPU tier.
2. Push `main.py`, `credibility.py`, and `requirements.txt` to the Space repo.
3. Put this at the top of the Space's own `README.md` so it launches the right file:

   ```yaml
   ---
   title: Credibility Scored Chatbot
   sdk: streamlit
   app_file: main.py
   pinned: false
   ---
   ```

4. Add `ANTHROPIC_API_KEY` under **Settings → Variables and secrets → New secret**.
   **Never commit your key** — a key pushed to a public Space is a key you must
   immediately revoke.
5. Submit the public Space URL alongside your other deliverables.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'anthropic'`**
Your virtual environment isn't active. Look for `(.venv)` at the start of your
prompt; if it's missing, re-run the activate command from [Setup](#setup).

**`TypeError: Messages.create() got an unexpected keyword argument 'output_config'`**
Your `anthropic` package is too old. `credibility.py` needs **0.70 or newer**:
```bash
pip install --upgrade anthropic
```

**The app loads but every answer is an error**
Check the sidebar. If "Anthropic API key" shows ❌, your `.env` was not found or the
key is malformed. The file must be named exactly `.env` (not `.env.txt` — Windows
Notepad does this silently) and sit in this directory.

**`streamlit: command not found`**
Activate the venv, or run it as a module: `python -m streamlit run main.py`.

**Everything is slow**
Each chat turn makes a Claude call plus one scoring call per source. Turn off the
SerpAPI checkbox, or set `JUDGE_MODEL = "claude-haiku-4-5"` in `credibility.py` while
developing. Say which model produced your submitted numbers.

**I want to work without spending API credits**
You can do most of the assignment that way. `python evaluate.py` and
`python test_credibility.py` run the rule layer only and never call the API. Only the
chat itself and `evaluate.py --llm` need a key.
