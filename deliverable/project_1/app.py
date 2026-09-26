"""
app.py — the Gradio front end for the credibility-scored research chatbot.

Same app as main.py, different UI framework. Both import their model call,
search helper, and citation extraction from chat_backend.py, so there is exactly
one copy of the logic and a fix in one place reaches both.

    uv run python app.py          # http://localhost:7860

This is the file a Hugging Face **Gradio** Space runs, which is why it is called
`app.py` — Spaces looks for that name by default. Gradio Spaces run free on
ZeroGPU, which is the reason this front end exists alongside the Streamlit one:
a Streamlit app on Spaces needs the Docker SDK, and creating a Docker Space
requires a paid plan. See the bonus section of README.md.

You should not need to change much in this file. The part you are graded on
lives in credibility.py.
"""

import os
from typing import Any, Dict, List, Tuple

import gradio as gr

from chat_backend import MAX_SOURCES, ask_claude, merge_sources, search_serpapi
from credibility import score_band, score_url

# Gradio has no `st.markdown(":green[...]")` equivalent, so the chips are plain
# HTML. These are the three bands from credibility.score_band(), mapped to
# colours that stay legible on both the light and dark Gradio themes.
BAND_COLOURS = {
    "green": "#1a7f37",
    "orange": "#bf8700",
    "red": "#cf222e",
}


def _chip(score: float) -> str:
    """Render one credibility score as a coloured inline badge."""
    label, colour = score_band(score)
    return (
        f'<span style="background:{BAND_COLOURS.get(colour, "#57606a")};color:#fff;'
        f'padding:2px 8px;border-radius:10px;font-size:0.85em;font-weight:600;'
        f'white-space:nowrap">● {score:.2f} {label}</span>'
    )


def render_sources(sources: List[Dict[str, str]]) -> str:
    """
    Turn the scored sources into one HTML block for the panel below the chat.

    This is the visible payoff of your work in credibility.py — a reader should
    be able to judge a source at a glance without reading the URL. Each score
    comes with its explanation, because a number nobody can interrogate is not
    much better than no number at all.
    """
    if not sources:
        return "_No sources for this answer yet._"

    parts = [f"**{len(sources)} source(s), scored by `credibility.score_url`**\n"]
    for i, source in enumerate(sources, 1):
        result = score_url(source["url"])
        title = source.get("title") or source["url"]
        parts.append(
            f'<p style="margin:.6em 0 .2em"><b>{i}.</b> '
            f'<a href="{source["url"]}" target="_blank">{title}</a> &nbsp; '
            f"{_chip(result['score'])}</p>"
        )
        if source.get("snippet"):
            parts.append(
                f'<p style="margin:.1em 0;color:#57606a;font-size:.9em">'
                f'{source["snippet"]}</p>'
            )
        parts.append(
            f"<details><summary>Why this score?</summary>"
            f"<p style='margin:.4em 0'>{result['explanation']}</p></details>"
        )
    return "\n".join(parts)


def score_one_url(url: str) -> str:
    """The standalone URL scorer. Works with no API key, same as in main.py."""
    if not url or not url.strip():
        return ""
    result = score_url(url.strip())
    label, _ = score_band(result["score"])
    return f"{_chip(result['score'])}<p style='margin:.5em 0'>{result['explanation']}</p>"


def status_markdown() -> str:
    """Which keys were found. Mirrors the Streamlit sidebar's status block."""
    from chat_backend import TRACING_ENABLED

    return "\n".join(
        [
            ("✅" if os.getenv("ANTHROPIC_API_KEY") else "❌") + " Anthropic API key",
            ("✅" if os.getenv("SERPAPI_API_KEY") else "⬜") + " SerpAPI key (optional)",
            ("✅" if TRACING_ENABLED else "⬜") + " Langfuse tracing (optional)",
        ]
    )


def respond(
    message: str,
    history: List[Dict[str, str]],
    user: str,
    email: str,
    use_serpapi: bool,
) -> Tuple[List[Dict[str, str]], str, str]:
    """
    Handle one chat turn.

    :param history: Gradio `type="messages"` history — [{"role", "content"}, ...]
    :return: (updated history, sources HTML, cleared textbox)
    """
    if not message or not message.strip():
        return history, "", ""

    history = list(history) + [{"role": "user", "content": message}]

    if not os.getenv("ANTHROPIC_API_KEY"):
        history.append(
            {
                "role": "assistant",
                "content": (
                    "No `ANTHROPIC_API_KEY` found. Copy `.env.example` to `.env` and add "
                    "your key — on a Hugging Face Space, add it under Settings → "
                    "Variables and secrets. The URL scorer below still works without one."
                ),
            }
        )
        return history, "", ""

    # Build the request separately from the displayed history. Search context is
    # useful for this turn only — writing it back into the history would re-send
    # it on every later turn and inflate the conversation.
    api_messages = [{"role": m["role"], "content": m["content"]} for m in history]
    serp_sources: List[Dict[str, str]] = []

    if use_serpapi and os.getenv("SERPAPI_API_KEY"):
        try:
            for r in search_serpapi(message, os.getenv("SERPAPI_API_KEY"))[:5]:
                serp_sources.append(
                    {
                        "title": r.get("title", "Untitled"),
                        "url": r.get("link", ""),
                        "snippet": r.get("snippet", ""),
                    }
                )
            if serp_sources:
                context = "\n\nSearch results for reference:\n" + "".join(
                    f"- {s['title']} ({s['url']})\n  {s['snippet']}\n" for s in serp_sources
                )
                api_messages[-1] = {"role": "user", "content": message + context}
        except Exception as exc:  # noqa: BLE001 — a failed search shouldn't kill the turn
            serp_sources = []
            gr.Warning(f"SerpAPI search failed: {exc}")

    try:
        answer, citations = ask_claude(api_messages, user, email, f"{user}_{email}")
    except Exception as exc:  # noqa: BLE001 — show the error rather than a blank screen
        history.append({"role": "assistant", "content": f"Error: {exc}"})
        return history, "", ""

    sources = merge_sources(citations, serp_sources)
    history.append({"role": "assistant", "content": answer})
    return history, render_sources(sources), ""


with gr.Blocks(title="CS676 — Credibility Chatbot") as demo:
    gr.Markdown("# 🔍 Credibility-Scored Research Assistant")
    gr.Markdown(
        "Ask a research question. The answer cites its sources, and every source "
        "carries a credibility score from `credibility.score_url` — the function "
        "you are graded on."
    )

    with gr.Row():
        with gr.Column(scale=3):
            # Gradio 6 takes [{"role": ..., "content": ...}] natively; the
            # `type="messages"` argument that Gradio 4/5 needed was removed.
            chatbot = gr.Chatbot(height=430, label="Conversation")
            with gr.Row():
                box = gr.Textbox(
                    placeholder="Ask a research question...",
                    show_label=False,
                    scale=8,
                    submit_btn=True,
                )
                clear = gr.Button("Clear", scale=1)
            sources_panel = gr.Markdown("_No sources yet._", label="Sources")

        with gr.Column(scale=1):
            gr.Markdown("### Session")
            user_box = gr.Textbox(value="student", label="Name")
            email_box = gr.Textbox(value="student@pace.edu", label="Email")
            serp_toggle = gr.Checkbox(value=False, label="Also search with SerpAPI")

            gr.Markdown("### Status")
            gr.Markdown(status_markdown())

            gr.Markdown("### Score any URL")
            gr.Markdown("_Works without an API key._")
            probe = gr.Textbox(
                placeholder="https://arxiv.org/abs/1706.03762",
                show_label=False,
                submit_btn=True,
            )
            probe_out = gr.Markdown()

    inputs = [box, chatbot, user_box, email_box, serp_toggle]
    outputs = [chatbot, sources_panel, box]
    box.submit(respond, inputs, outputs)

    probe.submit(score_one_url, probe, probe_out)
    clear.click(lambda: ([], "_No sources yet._", ""), None, outputs)


if __name__ == "__main__":
    # 0.0.0.0:7860 is what Hugging Face Spaces expects. Locally it just means the
    # app is reachable at http://localhost:7860.
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
