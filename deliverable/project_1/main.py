"""
CS676 Project 1 — Credibility-scored research chatbot.

A Streamlit chat app that answers questions using Claude, shows the sources it
used, and displays a credibility score beside each one.

Run it with:   uv run streamlit run main.py

There is also a Gradio version of this same app in app.py. Both share their
model call and citation handling via chat_backend.py.

You should not need to change much in this file. The part you are graded on
lives in credibility.py.
"""

import os
from typing import Any, Dict, List

import streamlit as st

from chat_backend import (
    MAX_SOURCES,
    TRACING_ENABLED,
    ask_claude,
    merge_sources,
    search_serpapi,
)
from credibility import score_band, score_url


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
