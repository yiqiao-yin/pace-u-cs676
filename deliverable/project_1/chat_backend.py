"""
chat_backend.py — the parts of the chatbot that are not the user interface.

Two front ends share this module:

    main.py    Streamlit. Run locally with `uv run streamlit run main.py`.
    app.py     Gradio. Run locally with `uv run python app.py`, and it is what
               a Hugging Face Gradio Space serves.

Everything here is UI-agnostic: the model call, the search helper, the citation
extraction, and the optional tracing. Nothing in this file imports streamlit or
gradio, which is the point — a change to how sources are extracted should not
have to be made twice, and a bug fixed in one front end should not still be
present in the other.

You should not need to change much in this file. The part you are graded on
lives in credibility.py.
"""

import os
from typing import Any, Dict, List, Tuple

import anthropic
from dotenv import load_dotenv

load_dotenv()

# Claude Opus 5 is the most capable model. Swap to "claude-sonnet-5" or
# "claude-haiku-4-5" if you want to reduce cost while developing — note which
# one your submitted numbers used.
CHAT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

# Web search can return 20+ results per turn, and every displayed source is
# scored — one API call each when the LLM layer is on. Cap what we show.
MAX_SOURCES = 6

SYSTEM_PROMPT = """You are a research assistant for a graduate data science course.

Answer using the sources available to you and cite them. Be direct and concise.
When the evidence is thin or the sources disagree, say so plainly rather than
smoothing it over. Never invent a source or a URL."""


# -----------------------------------------------------------------------------
# Optional Langfuse tracing
# -----------------------------------------------------------------------------
# Tracing is a nice-to-have, not a requirement. If the Langfuse keys are absent
# we fall back to a no-op decorator so the app still runs on a fresh clone.
# This is why you can start working before configuring anything but the API key.
try:
    from langfuse import get_client, observe

    _langfuse = get_client()
    TRACING_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
except Exception:
    TRACING_ENABLED = False
    _langfuse = None

    def observe(*_args, **_kwargs):  # type: ignore[misc]
        """No-op stand-in for @observe when Langfuse is not configured."""
        def decorator(fn):
            return fn
        return decorator


def search_serpapi(query: str, api_key: str) -> List[Dict[str, Any]]:
    """
    Search Google via SerpAPI and return the organic results.

    This is optional context on top of Claude's own web search — it gives you a
    second, independently-retrieved set of URLs to score, which is useful when
    comparing how your scorer treats different kinds of source.
    """
    from serpapi import GoogleSearch

    search = GoogleSearch({"q": query, "api_key": api_key})
    return search.get_dict().get("organic_results", [])


# ---------------------------------------------------------------------------
# ⚠️  NEEDS YOUR OWN API KEY
# ---------------------------------------------------------------------------
# REQUIRES A KEY. The chat does not work without ANTHROPIC_API_KEY in `.env`;
# both front ends show a red mark when it is missing. Get one at
# https://console.anthropic.com/. Calls are billed to you. The URL scorer, the
# tests, and evaluate.py all work without a key.
#
# VERIFIED LIVE — after a real bug was found here. The first live run returned
# ZERO sources, because this function originally read citations off the text
# blocks. `web_search_20260209` does not put them there: it returns them in
# `web_search_tool_result` blocks, and `block.citations` is None. The code below
# now reads both, and a live run yields six sources including the actual
# arXiv link for "Attention Is All You Need".
#
# The lesson is worth more than the fix: an API that returns an empty list where
# you expected data fails silently. Nothing crashed, no error was logged, the
# app just quietly showed no sources at all.
# ---------------------------------------------------------------------------
@observe()
def ask_claude(messages: List[Dict[str, str]], user: str, email: str, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Send the conversation to Claude and return the answer plus its citations.

    Where the sources come from: the `web_search_20260209` tool returns them in
    `web_search_tool_result` blocks, NOT as citation metadata on the text blocks.
    That is worth knowing — the obvious implementation reads `block.citations`,
    finds it empty, and silently shows no sources at all, which is exactly the
    bug this function was shipped with until it was run against the live API.

    :return: (answer_text, [{"url": ..., "title": ...}, ...])
    """
    client = anthropic.Anthropic()

    response = client.messages.create(
        model=CHAT_MODEL,
        max_tokens=MAX_TOKENS,
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
    )

    # Claude can decline a request. Check before reading content, which is empty
    # or partial on a refusal.
    if response.stop_reason == "refusal":
        return ("I can't help with that request.", [])

    answer = ""
    citations: List[Dict[str, str]] = []
    seen: set = set()

    for block in response.content:
        if block.type == "text":
            answer += block.text
            # Some configurations attach citations directly to text blocks.
            # web_search_20260209 does NOT — see the branch below — but keep this
            # path so the app still works if that changes or you enable document
            # citations.
            for citation in getattr(block, "citations", None) or []:
                url = getattr(citation, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"url": url, "title": getattr(citation, "title", "") or url})

        elif block.type == "web_search_tool_result":
            # This is where the sources actually are. Each successful result block
            # holds a list of `web_search_result` items with .url and .title.
            # On failure `.content` is a single error object rather than a list,
            # so check the type before iterating.
            results = getattr(block, "content", None)
            if not isinstance(results, list):
                continue
            for item in results:
                url = getattr(item, "url", None)
                if url and url not in seen:
                    seen.add(url)
                    citations.append({"url": url, "title": getattr(item, "title", "") or url})

    # A single turn can return twenty-odd results across several searches, and the
    # app scores every one of them — which with the LLM layer on is one API call
    # each. Cap it: the first few are the ones the model actually leaned on.
    citations = citations[:MAX_SOURCES]

    if TRACING_ENABLED and _langfuse is not None:
        _langfuse.update_current_trace(
            input=messages[-1]["content"] if messages else "",
            output=answer,
            user_id=user,
            session_id=session_id,
            tags=["cs676", "project-1"],
            metadata={"email": email, "citations": len(citations)},
        )

    return answer, citations


def merge_sources(citations: List[Dict[str, str]], extra: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Combine Claude's citations with any SerpAPI results, dropping duplicates."""
    sources: List[Dict[str, str]] = []
    seen: set = set()
    for source in citations + extra:
        if source.get("url") and source["url"] not in seen:
            seen.add(source["url"])
            sources.append(source)
    return sources
