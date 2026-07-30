"""
Tests for PersonaAgent — the markdown-plus-model construction.

The important assertions here are about what reaches the model: the persona body
must land in the system prompt, and the transcript must be rewritten from that
agent's point of view.
"""

from personaforge import PersonaSpec, agent_from_spec
from personaforge.conversation import Turn
from personaforge.llm import ScriptedLLM


def make_agent(replies=None):
    spec = PersonaSpec(
        name="Dr. Reyes",
        role="doctor",
        summary="direct, busy",
        body="# Dr. Reyes\n\n## Personality\n- Blunt but not unkind\n- SECRET_MARKER_XYZ",
    )
    llm = ScriptedLLM(replies=replies or ["What brings you in today?"])
    return agent_from_spec(spec, llm), llm


def test_persona_body_becomes_the_system_prompt():
    agent, _ = make_agent()
    prompt = agent.system_prompt()

    # The persona markdown is the brain — if it isn't here, the agent is nobody.
    assert "SECRET_MARKER_XYZ" in prompt
    assert "Dr. Reyes" in prompt


def test_system_prompt_forbids_narration():
    agent, _ = make_agent()
    prompt = agent.system_prompt().lower()
    assert "stage directions" in prompt
    assert "speak only as" in prompt


def test_respond_returns_the_model_reply():
    agent, _ = make_agent(replies=["Take a seat."])
    assert agent.respond([]) == "Take a seat."


def test_empty_transcript_gets_a_user_opener():
    """The API requires the first message to be from the user."""
    agent, llm = make_agent()
    agent.respond([])

    messages = llm.calls[0]["messages"]
    assert messages[0]["role"] == "user"


def test_own_lines_become_assistant_turns():
    agent, llm = make_agent()
    transcript = [
        Turn(speaker="Scene", text="A clinic room."),
        Turn(speaker="Dr. Reyes", text="What brings you in?"),
        Turn(speaker="Maria", text="My back hurts."),
    ]
    agent.respond(transcript)

    roles = [m["role"] for m in llm.calls[0]["messages"]]
    contents = [m["content"] for m in llm.calls[0]["messages"]]

    # The agent's own line is 'assistant'; everyone else is 'user' and is
    # prefixed with the speaker's name so the model can tell them apart.
    assert roles == ["user", "assistant", "user"]
    assert contents[1] == "What brings you in?"
    assert contents[2].startswith("Maria:")


def test_agent_name_comes_from_the_spec():
    agent, _ = make_agent()
    assert agent.name == "Dr. Reyes"
