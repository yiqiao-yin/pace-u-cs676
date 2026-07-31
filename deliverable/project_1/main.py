"""
CS676 Project 1 — Credibility-scored research chatbot.

A Streamlit chat app that answers questions using Claude, shows the sources it
used, and displays a credibility score beside each one.

Run it with:   streamlit run main.py

You should not need to change much in this file. The part you are graded on
lives in credibility.py.
"""

import os
from typing import Any, Dict, List, Tuple

import anthropic
import streamlit as st
from dotenv import load_dotenv

from credibility import score_band, score_url

load_dotenv()

# Claude Opus 5 is the most capable model. Swap to "claude-sonnet-5" or
# "claude-haiku-4-5" if you want to reduce cost while developing — note which
# one your submitted numbers used.
CHAT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000

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
# ⚠️  NOT VERIFIED AGAINST THE LIVE API
# ---------------------------------------------------------------------------
# `ask_claude()` below has never been run against the real Anthropic API — no
# billed request has been made. The parameters match the current SDK, but the
# citation-extraction loop in particular is unproven against a real response.
# If something breaks on your first keyed run, start here.
# ---------------------------------------------------------------------------
@observe()
def ask_claude(messages: List[Dict[str, str]], user: str, email: str, session_id: str) -> Tuple[str, List[Dict[str, str]]]:
    """
    Send the conversation to Claude and return the answer plus its citations.

    Claude's server-side web_search tool attaches citation metadata to the text
    blocks it returns. We pull those out here so the UI can score them — the
    earlier version of this app dropped them, which meant there was nothing to
    attach a credibility score to.

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
        if block.type != "text":
            continue
        answer += block.text
        # Citations ride along on the text blocks that used a search result.
        for citation in getattr(block, "citations", None) or []:
            url = getattr(citation, "url", None)
            if url and url not in seen:
                seen.add(url)
                citations.append({"url": url, "title": getattr(citation, "title", "") or url})

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


def render_source(index: int, title: str, url: str, snippet: str = "") -> None:
    """
    Render one source as a labelled row with a coloured credibility chip.

    The chip is the visible payoff of your work in credibility.py — a reader
    should be able to judge a source at a glance without reading the URL.
    """
    result = score_url(url)
    label, colour = score_band(result["score"])

    st.markdown(
        f"**{index}. [{title}]({url})** &nbsp; "
        f":{colour}[**● {result['score']:.2f} {label}**]"
    )
    if snippet:
        st.caption(snippet)
    with st.expander("Why this score?"):
        st.write(result["explanation"])


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CS676 — Credibility Chatbot", page_icon="🔍")
st.title("🔍 Credibility-Scored Research Assistant")

with st.sidebar:
    st.subheader("Session")
    user = st.text_input("Name", value="student")
    email = st.text_input("Email", value="student@pace.edu")
    session_id = f"{user}_{email}"

    st.divider()
    use_serpapi = st.checkbox("Also search with SerpAPI", value=False)

    st.divider()
    st.caption("**Status**")
    st.caption(("✅" if os.getenv("ANTHROPIC_API_KEY") else "❌") + " Anthropic API key")
    st.caption(("✅" if os.getenv("SERPAPI_API_KEY") else "⬜") + " SerpAPI key (optional)")
    st.caption(("✅" if TRACING_ENABLED else "⬜") + " Langfuse tracing (optional)")

    st.divider()
    st.caption("Score any URL directly:")
    probe = st.text_input("URL", placeholder="https://arxiv.org/abs/1706.03762")
    if probe:
        probe_result = score_url(probe)
        probe_label, probe_colour = score_band(probe_result["score"])
        st.markdown(f":{probe_colour}[**{probe_result['score']:.2f} — {probe_label}**]")
        st.caption(probe_result["explanation"])

if not os.getenv("ANTHROPIC_API_KEY"):
    st.warning("No ANTHROPIC_API_KEY found. Copy `.env.example` to `.env` and add your key. "
               "The URL scorer in the sidebar still works without one.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Replay the conversation so far.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for i, source in enumerate(message.get("sources", []), 1):
            render_source(i, source["title"], source["url"], source.get("snippet", ""))

if prompt := st.chat_input("Ask a research question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build the request separately from the stored history. Search context is
    # useful for this turn only — writing it back into session_state would
    # re-send it on every later turn and inflate the conversation.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    serp_sources: List[Dict[str, str]] = []

    if use_serpapi and os.getenv("SERPAPI_API_KEY"):
        try:
            results = search_serpapi(prompt, os.getenv("SERPAPI_API_KEY"))[:5]
            if results:
                context = "\n\nSearch results for reference:\n"
                for r in results:
                    title = r.get("title", "Untitled")
                    link = r.get("link", "")
                    snippet = r.get("snippet", "")
                    serp_sources.append({"title": title, "url": link, "snippet": snippet})
                    context += f"- {title} ({link})\n  {snippet}\n"
                api_messages[-1] = {"role": "user", "content": prompt + context}
        except Exception as e:
            st.warning(f"SerpAPI search failed: {e}")

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer, citations = ask_claude(api_messages, user, email, session_id)
            except Exception as e:
                answer, citations = f"Error: {e}", []

        st.markdown(answer)

        # Merge Claude's own citations with any SerpAPI results, dropping dupes.
        sources: List[Dict[str, str]] = []
        seen_urls: set = set()
        for source in citations + serp_sources:
            if source["url"] and source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                sources.append(source)

        if sources:
            st.divider()
            st.caption(f"**{len(sources)} source(s), scored by `credibility.score_url`**")
            for i, source in enumerate(sources, 1):
                render_source(i, source["title"], source["url"], source.get("snippet", ""))

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
