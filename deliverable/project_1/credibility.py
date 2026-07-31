"""
credibility.py — THIS IS THE FILE YOU IMPROVE.
===============================================================================

Everything else in this project is scaffolding. The chatbot works, the UI works,
the tracing works. What does NOT work well is the function below: `score_url()`.

Your job for Project 1 is to make it better. Read the KNOWN WEAKNESSES section
at the bottom of this file — every item on that list is a bug you are invited to
fix, and every one of them is worth points.

-------------------------------------------------------------------------------
THE CONTRACT (do not change this)
-------------------------------------------------------------------------------
    score_url("https://arxiv.org/abs/1706.03762")

    -> {"score": 0.9, "explanation": "arxiv.org is a recognized preprint ..."}

    score:       float in [0.0, 1.0].  0 = not credible, 1 = highly credible.
    explanation: str. A human-readable reason for the score.

Your grader, the evaluation harness (`evaluate.py`), the test suite
(`test_credibility.py`), and the Streamlit app all depend on this exact shape.
If you change the keys or the types, everything downstream breaks.

-------------------------------------------------------------------------------
HOW THE BASELINE WORKS
-------------------------------------------------------------------------------
Two layers, combined at the end:

    Layer 1 (rules)  Pure string inspection of the URL. No network, no API key.
                     Always runs. This is why the app works before you have
                     configured anything.

    Layer 2 (LLM)    One Claude call that judges the URL. Only runs when
                     ANTHROPIC_API_KEY is set and `use_llm` is not False.
                     Skipped silently otherwise.

The final score is a weighted blend of the two. See `score_url()`.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

# The model used for the Layer 2 judgment. Claude Opus 5 is the most capable
# model; switch to "claude-haiku-4-5" if you are scoring many URLs and want to
# cut cost, or "claude-sonnet-5" for a middle option. Scoring quality will move
# with this choice, so note in your report which model your numbers came from.
JUDGE_MODEL = "claude-opus-5"

# How much each layer contributes to the final score. These two must sum to 1.0.
# Tuning this split is one of the easiest wins available to you.
RULE_WEIGHT = 0.6
LLM_WEIGHT = 0.4


# =============================================================================
# LAYER 1 — RULE-BASED SIGNALS
# =============================================================================
# Each table below is a hand-written guess about what makes a source credible.
# They are deliberately short and deliberately naive. Extending them is the
# fastest way to improve your score on the evaluation set, but note that a
# longer lookup table is not the same thing as a better *algorithm* — the
# report asks you to justify your approach, not just your word list.

# Exact-domain judgments. Highest-confidence signal we have.
DOMAIN_SCORES: Dict[str, float] = {
    # Peer-reviewed / archival
    "nature.com": 0.95,
    "science.org": 0.95,
    "nejm.org": 0.95,
    "thelancet.com": 0.95,
    "pubmed.ncbi.nlm.nih.gov": 0.92,
    "arxiv.org": 0.75,          # preprint: NOT peer reviewed
    "biorxiv.org": 0.70,        # preprint: NOT peer reviewed
    # Reference
    "wikipedia.org": 0.65,
    "britannica.com": 0.80,
    # Mainstream press
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "bbc.com": 0.82,
    "nytimes.com": 0.80,
    "wsj.com": 0.80,
    # User-generated / self-published
    "medium.com": 0.35,
    "substack.com": 0.35,
    "blogspot.com": 0.25,
    "wordpress.com": 0.25,
    "reddit.com": 0.25,
    "quora.com": 0.20,
    "x.com": 0.15,
    "twitter.com": 0.15,
    # Satire — factually false by design, which the LLM layer often misses
    "theonion.com": 0.05,
    "clickhole.com": 0.05,
    "babylonbee.com": 0.05,
}

# Fallback when the exact domain is unknown. Coarse and easy to fool.
TLD_SCORES: Dict[str, float] = {
    ".gov": 0.88,
    ".edu": 0.82,
    ".mil": 0.85,
    ".org": 0.60,
    ".com": 0.50,
    ".net": 0.48,
    ".io": 0.45,
    ".biz": 0.30,
    ".info": 0.30,
    ".xyz": 0.25,
}

# Substrings in the URL path that hint at self-published or low-edit content.
PATH_PENALTIES: Dict[str, float] = {
    "/blog/": -0.10,
    "/opinion/": -0.08,
    "/sponsored/": -0.20,
    "/press-release/": -0.15,
    "/advertorial/": -0.25,
    "/forum/": -0.12,
    "/comments/": -0.12,
}

# Neutral starting point for a URL we know nothing about.
NEUTRAL_SCORE = 0.5


@dataclass
class Signal:
    """One piece of evidence that moved the score, kept so we can explain it."""

    name: str      # short machine-readable label, e.g. "known_domain"
    value: float   # the score or delta this signal contributed
    reason: str    # human-readable sentence for the explanation field


def _normalize_domain(url: str) -> str:
    """
    Pull a bare lowercase domain out of a URL.

    Strips the scheme, any userinfo, the port, and a leading "www.". Returns an
    empty string when the URL has no host at all, which the caller treats as a
    malformed input.
    """
    host = (urlparse(url).netloc or "").lower()
    host = host.split("@")[-1]      # drop user:pass@
    host = host.split(":")[0]       # drop :port
    if host.startswith("www."):
        host = host[4:]
    return host


def _match_known_domain(domain: str) -> Optional[Tuple[str, float]]:
    """
    Look the domain up in DOMAIN_SCORES, allowing subdomains to match.

    "en.wikipedia.org" matches the "wikipedia.org" entry, and "arxiv.org"
    matches itself. We check the exact domain first so a more specific entry
    always wins over a more general one.
    """
    if domain in DOMAIN_SCORES:
        return domain, DOMAIN_SCORES[domain]
    for known, score in DOMAIN_SCORES.items():
        if domain.endswith("." + known):
            return known, score
    return None


def rule_based_signals(url: str) -> List[Signal]:
    """
    Inspect the URL string and return every signal that fired.

    This runs with no network access and no API key, which is what makes the
    app usable straight after `git clone`. It is also the reason the baseline
    is weak: a URL string alone tells you almost nothing about whether the
    page behind it is any good.
    """
    signals: List[Signal] = []
    parsed = urlparse(url)
    domain = _normalize_domain(url)

    # Signal 1: exact or suffix match against our hand-written domain table.
    match = _match_known_domain(domain)
    if match:
        known, score = match
        signals.append(Signal("known_domain", score, f"'{known}' is a domain we recognize"))
    else:
        # Signal 2: fall back to the top-level domain. Very coarse.
        for tld, score in TLD_SCORES.items():
            if domain.endswith(tld):
                signals.append(Signal("tld", score, f"'{tld}' domains score {score:.2f} by default"))
                break
        else:
            signals.append(Signal("unknown", NEUTRAL_SCORE, "unrecognized domain and TLD"))

    # Signal 3: HTTPS. Weak evidence — a scam site can buy a certificate too.
    if parsed.scheme == "https":
        signals.append(Signal("https", 0.02, "served over HTTPS"))
    elif parsed.scheme == "http":
        signals.append(Signal("no_https", -0.05, "served over plain HTTP"))

    # Signal 4: path keywords suggesting opinion, sponsorship, or user content.
    path = (parsed.path or "").lower()
    for fragment, delta in PATH_PENALTIES.items():
        if fragment in path:
            signals.append(Signal("path", delta, f"URL path contains '{fragment}'"))

    # Signal 5: a DOI in the path implies a registered scholarly work.
    if re.search(r"/10\.\d{4,9}/", path):
        signals.append(Signal("doi", 0.10, "URL contains a DOI, suggesting a registered publication"))

    return signals


def _combine_signals(signals: List[Signal]) -> float:
    """
    Fold the signal list into a single number in [0, 1].

    The first signal is treated as the base score (it is always the domain or
    TLD judgment) and every later signal is an additive adjustment. This is a
    crude aggregation — see KNOWN WEAKNESSES.
    """
    if not signals:
        return NEUTRAL_SCORE
    base = signals[0].value
    adjustment = sum(s.value for s in signals[1:])
    return max(0.0, min(1.0, base + adjustment))


# =============================================================================
# LAYER 2 — LLM JUDGMENT
# =============================================================================

_JUDGE_SYSTEM = """You assess the credibility of web sources for a research assistant.

Given a URL, judge how much a careful reader should trust content published there.
Consider: the publisher's editorial standards and reputation, whether the content is
peer reviewed, whether it is self-published, and whether the outlet is satirical.

Score 0.0 (not credible at all) to 1.0 (highly credible). Be skeptical of
self-published platforms and satire. Judge the SOURCE, not the topic. If you do not
recognize the domain, say so and score near 0.5 rather than guessing confidently."""

# Constraining the response to this schema means we never have to parse prose or
# repair malformed JSON — the API guarantees the shape.
_JUDGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number", "description": "Credibility from 0.0 to 1.0"},
        "reason": {"type": "string", "description": "One sentence justifying the score"},
    },
    "required": ["score", "reason"],
    "additionalProperties": False,
}


# =============================================================================
# ⚠️  NOT VERIFIED AGAINST THE LIVE API
# =============================================================================
# The rule-based layer, the tests, and evaluate.py have all been run and pass.
# `llm_opinion()` below has NOT been executed against the real Anthropic API —
# it was written and reviewed against the current SDK signature, but no billed
# request has ever been made with it.
#
# What that means for you: the FIRST time you run with an ANTHROPIC_API_KEY set,
# treat this function as unproven. If it misbehaves, the likely suspects are the
# `output_config` structured-output call and the response parsing right below it.
# Report what you find — fixing it counts toward your grade, and telling the
# class about it is worth more.
# =============================================================================


def llm_opinion(url: str) -> Optional[Signal]:
    """
    Ask Claude to judge the URL. Returns None whenever the call cannot be made.

    Returning None rather than raising is deliberate: a missing API key, a
    network blip, or a safety refusal should degrade the score to rules-only
    instead of taking down the whole app. Effort is set to "low" because this
    is a small judgment and we may be scoring several URLs per question.
    """
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None

    try:
        import anthropic

        client = anthropic.Anthropic()
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=1024,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": f"Rate the credibility of this source: {url}"}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": _JUDGE_SCHEMA}},
        )

        # Claude can decline a request; content is empty or partial when it does.
        if response.stop_reason == "refusal":
            return None

        text = next((b.text for b in response.content if b.type == "text"), "")
        data = json.loads(text)
        score = max(0.0, min(1.0, float(data["score"])))
        return Signal("llm", score, str(data["reason"]))

    except Exception:
        # Any failure falls back to rules-only scoring rather than crashing.
        return None


# =============================================================================
# THE FUNCTION YOU ARE GRADED ON
# =============================================================================

# Scoring the same URL repeatedly in one session is common (a chat may cite the
# same paper on every turn), so results are memoized for the process lifetime.
_CACHE: Dict[Tuple[str, Optional[bool]], Dict[str, Any]] = {}


def score_url(url: str, use_llm: Optional[bool] = None) -> Dict[str, Any]:
    """
    Score the credibility of a source URL.

    :param url:     The URL to evaluate.
    :param use_llm: True forces the Claude judgment, False forces rules-only,
                    None (default) uses the LLM when an API key is available.
    :return:        {"score": float in [0,1], "explanation": str}
    """
    cache_key = (url, use_llm)
    if cache_key in _CACHE:
        return dict(_CACHE[cache_key])

    # Guard clause: anything that is not a usable http(s) URL scores 0.0 with an
    # explanation rather than raising, so one bad link cannot break a whole page.
    if not isinstance(url, str) or not url.strip():
        return {"score": 0.0, "explanation": "No URL was provided."}

    parsed = urlparse(url.strip())
    if parsed.scheme not in ("http", "https") or not _normalize_domain(url):
        return {"score": 0.0, "explanation": f"'{url}' is not a valid http(s) URL."}

    # Layer 1 always runs.
    signals = rule_based_signals(url)
    rule_score = _combine_signals(signals)
    parts = [s.reason for s in signals]

    # Layer 2 runs only when it can. Blend if we got an opinion, otherwise the
    # rule score stands on its own.
    llm = llm_opinion(url) if use_llm is not False else None
    if llm is not None:
        final = RULE_WEIGHT * rule_score + LLM_WEIGHT * llm.value
        parts.append(f"model judgment {llm.value:.2f} — {llm.reason}")
    else:
        final = rule_score

    final = round(max(0.0, min(1.0, final)), 2)
    result = {"score": final, "explanation": "; ".join(parts) + "."}

    _CACHE[cache_key] = dict(result)
    return result


def score_band(score: float) -> Tuple[str, str]:
    """
    Map a score onto a display band. Used by the app to colour the source chips.

    :return: (label, streamlit_colour) — e.g. ("HIGH", "green")
    """
    if score >= 0.70:
        return "HIGH", "green"
    if score >= 0.40:
        return "MEDIUM", "orange"
    return "LOW", "red"


# =============================================================================
# KNOWN WEAKNESSES — YOUR TASK LIST
# =============================================================================
#
# This baseline is deliberately mediocre. Everything below is a real defect.
# You are not expected to fix all of them; pick the ones you can defend in your
# Deliverable 2 report, and measure the change with `python evaluate.py`.
#
#  1. IT NEVER READS THE PAGE. The whole of Layer 1 inspects a string. It cannot
#     tell a rigorous article from a hoax hosted on the same domain. Fetching the
#     page and looking for an author, a date, citations, or a corrections policy
#     is the single biggest available improvement.
#
#  2. THE DOMAIN TABLE IS A HAND-WRITTEN GUESS. ~30 domains, no source, no
#     evidence. Wikipedia's own perennial-sources list and similar published
#     datasets are real, citable alternatives to inventing numbers.
#
#  3. IT CANNOT TELL A PREPRINT FROM A PEER-REVIEWED PAPER. arxiv.org is scored
#     0.75 for every paper on it, whether it is "Attention Is All You Need" or
#     something posted this morning that nobody has read.
#
#  4. IT HAS NEVER HEARD OF RETRACTION. A retracted, discredited paper scores
#     exactly as high as a replicated one. Crossref and OpenAlex both expose
#     retraction status and citation counts over free APIs.
#
#  5. ANY .edu SCORES HIGHLY. Including a student's personal homepage hosted on
#     a university server.
#
#  6. THE AGGREGATION IS ARITHMETIC, NOT STATISTICAL. `_combine_signals` adds
#     numbers that were picked by hand. Nothing here is fitted to data. A
#     regression or a lasso over labelled examples would let you *learn* the
#     weights instead of guessing them — and would let you report which features
#     actually matter (Session 06).
#
#  7. THE SCORE IS NOT CALIBRATED. A 0.7 does not mean "right 70% of the time";
#     it means "some numbers happened to add up to 0.7". A calibration curve or
#     a Brier score would tell you how wrong that is (Session 05).
#
#  8. THERE IS NO UNCERTAINTY. An unrecognized domain and a well-known journal
#     both return a bare point estimate. Bootstrapping a confidence interval is
#     directly on the syllabus (Session 05).
#
#  9. THE TWO LAYERS ARE BLENDED WITH A CONSTANT. RULE_WEIGHT = 0.6 because 0.6
#     looked reasonable. It was never tested against anything.
#
# 10. THE EXPLANATION IS A LIST OF FRAGMENTS JOINED BY SEMICOLONS. It states
#     which rules fired, not why the reader should care. Explanation quality is
#     graded separately from score accuracy.
#
# 11. SUBDOMAINS INHERIT THE PARENT'S REPUTATION IN FULL. `_match_known_domain`
#     suffix-matches, so an opinion blog at blogs.nytimes.com scores exactly as
#     high as the newspaper's reporting, and anything hosted on a subdomain of a
#     trusted publisher is trusted automatically. Whether that inheritance is
#     right depends on the host, and the code never asks.
#
# 12. PENALTIES STACK WITHOUT A FLOOR. A URL matching three PATH_PENALTIES
#     entries takes all three hits additively. Nothing checks whether the
#     combination is meaningful or just the same signal counted three times.
