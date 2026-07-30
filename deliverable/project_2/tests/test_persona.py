"""
Tests for persona serialization and the on-disk format.

None of these touch the network. A persona is a file, so these are file tests.
"""

from personaforge import PersonaSpec, find_persona, generate_persona, list_personas, load_persona, save_persona
from personaforge.llm import ScriptedLLM

SAMPLE_MARKDOWN = """---
name: Maria Delgado
role: patient
summary: 58-year-old with poorly controlled type 2 diabetes
---

# Maria Delgado

## Background
Retired schoolteacher. Diagnosed eight years ago.

## How they speak
- Long sentences, lots of context before the point
"""


def test_slug_is_filesystem_safe():
    spec = PersonaSpec(name="Dr. Anne O'Hara-Smith", role="doctor", summary="x", body="y")
    assert spec.slug() == "dr-anne-o-hara-smith"


def test_slug_never_empty():
    spec = PersonaSpec(name="!!!", role="x", summary="", body="")
    assert spec.slug() == "unnamed"


def test_roundtrip_preserves_fields():
    spec = PersonaSpec(name="Maria Delgado", role="patient", summary="a summary", body="# Body\n\ntext")
    parsed = PersonaSpec.from_markdown(spec.to_markdown())
    assert parsed.name == spec.name
    assert parsed.role == spec.role
    assert parsed.summary == spec.summary
    assert parsed.body == spec.body


def test_parses_frontmatter():
    spec = PersonaSpec.from_markdown(SAMPLE_MARKDOWN)
    assert spec.name == "Maria Delgado"
    assert spec.role == "patient"
    assert "Retired schoolteacher" in spec.body
    # The frontmatter block itself must not leak into the body — it would end up
    # in the system prompt and confuse the model about who it is.
    assert "---" not in spec.body.split("\n")[0]


def test_missing_frontmatter_still_loads():
    spec = PersonaSpec.from_markdown("# Just a heading\n\nSome prose.")
    assert spec.name == "Unnamed"
    assert "Just a heading" in spec.body


def test_save_and_load(tmp_path):
    spec = PersonaSpec(name="Maria Delgado", role="patient", summary="s", body="# M")
    path = save_persona(spec, tmp_path)

    assert path.exists()
    assert path.name == "maria-delgado.md"
    assert load_persona(path).name == "Maria Delgado"


def test_list_personas_empty_dir(tmp_path):
    assert list_personas(tmp_path / "does-not-exist") == []


def test_list_personas_sorted(tmp_path):
    for name in ["Zoe Chen", "Adam Blake"]:
        save_persona(PersonaSpec(name=name, role="x", summary="", body="b"), tmp_path)

    names = [s.name for s in list_personas(tmp_path)]
    assert names == ["Adam Blake", "Zoe Chen"]


def test_find_persona_by_name_slug_and_role(tmp_path):
    save_persona(PersonaSpec(name="Maria Delgado", role="patient", summary="", body="b"), tmp_path)

    assert find_persona("Maria Delgado", tmp_path).role == "patient"
    assert find_persona("maria-delgado", tmp_path).role == "patient"
    assert find_persona("patient", tmp_path).name == "Maria Delgado"
    assert find_persona("nobody at all", tmp_path) is None


def test_generate_persona_parses_model_output():
    llm = ScriptedLLM(replies=[SAMPLE_MARKDOWN])
    spec = generate_persona("a diabetic patient", llm)

    assert spec.name == "Maria Delgado"
    assert spec.role == "patient"
    assert len(llm.calls) == 1


def test_generate_persona_survives_unstructured_output():
    """The model ignoring the format must not produce a nameless persona."""
    llm = ScriptedLLM(replies=["Sure! Here is a persona for you."])
    spec = generate_persona("a nurse", llm)

    assert spec.name != "Unnamed"
    assert spec.summary == "a nurse"
