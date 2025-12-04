"""Tests for extraction prompt templates."""


from research_kb_extraction.prompts import (
    SYSTEM_PROMPT,
    EXTRACTION_PROMPT,
    DEFINITION_EXTRACTION_PROMPT,
    METHOD_RELATIONSHIP_PROMPT,
    QUICK_EXTRACTION_PROMPT,
    format_extraction_prompt,
)


def test_system_prompt_structure():
    """Test SYSTEM_PROMPT is well-formed."""
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 50
    assert "causal inference" in SYSTEM_PROMPT.lower()
    assert "JSON" in SYSTEM_PROMPT or "json" in SYSTEM_PROMPT


def test_system_prompt_mentions_task():
    """Test SYSTEM_PROMPT describes the extraction task."""
    assert "extract" in SYSTEM_PROMPT.lower()
    assert "concepts" in SYSTEM_PROMPT.lower()
    assert "relationships" in SYSTEM_PROMPT.lower()


def test_extraction_prompt_structure():
    """Test EXTRACTION_PROMPT is well-formed."""
    assert isinstance(EXTRACTION_PROMPT, str)
    assert "{chunk}" in EXTRACTION_PROMPT
    assert "JSON" in EXTRACTION_PROMPT or "json" in EXTRACTION_PROMPT


def test_extraction_prompt_defines_concept_types():
    """Test EXTRACTION_PROMPT defines expected concept types."""
    assert "method" in EXTRACTION_PROMPT
    assert "assumption" in EXTRACTION_PROMPT
    assert "problem" in EXTRACTION_PROMPT
    assert "definition" in EXTRACTION_PROMPT
    assert "theorem" in EXTRACTION_PROMPT


def test_extraction_prompt_defines_relationship_types():
    """Test EXTRACTION_PROMPT defines relationship types."""
    relationship_types = [
        "REQUIRES",
        "USES",
        "ADDRESSES",
        "GENERALIZES",
        "SPECIALIZES",
        "ALTERNATIVE_TO",
        "EXTENDS",
    ]

    for rel_type in relationship_types:
        assert rel_type in EXTRACTION_PROMPT


def test_extraction_prompt_json_schema():
    """Test EXTRACTION_PROMPT includes valid JSON schema."""
    assert '{"concepts":' in EXTRACTION_PROMPT or '"concepts": [' in EXTRACTION_PROMPT
    assert '"relationships"' in EXTRACTION_PROMPT
    assert '"name"' in EXTRACTION_PROMPT
    assert '"concept_type"' in EXTRACTION_PROMPT
    assert '"confidence"' in EXTRACTION_PROMPT


def test_definition_extraction_prompt_structure():
    """Test DEFINITION_EXTRACTION_PROMPT is well-formed."""
    assert isinstance(DEFINITION_EXTRACTION_PROMPT, str)
    assert "{chunk}" in DEFINITION_EXTRACTION_PROMPT
    assert "definition" in DEFINITION_EXTRACTION_PROMPT.lower()


def test_definition_extraction_prompt_focus():
    """Test DEFINITION_EXTRACTION_PROMPT focuses on definitions."""
    assert "formal" in DEFINITION_EXTRACTION_PROMPT.lower()
    assert "theorem" in DEFINITION_EXTRACTION_PROMPT.lower()
    assert "assumption" in DEFINITION_EXTRACTION_PROMPT.lower()


def test_method_relationship_prompt_structure():
    """Test METHOD_RELATIONSHIP_PROMPT is well-formed."""
    assert isinstance(METHOD_RELATIONSHIP_PROMPT, str)
    assert "{chunk}" in METHOD_RELATIONSHIP_PROMPT
    assert "relationships" in METHOD_RELATIONSHIP_PROMPT.lower()


def test_method_relationship_prompt_focus():
    """Test METHOD_RELATIONSHIP_PROMPT focuses on method relationships."""
    assert "method" in METHOD_RELATIONSHIP_PROMPT.lower()
    assert "require" in METHOD_RELATIONSHIP_PROMPT.lower()
    assert "assumption" in METHOD_RELATIONSHIP_PROMPT.lower()


def test_quick_extraction_prompt_structure():
    """Test QUICK_EXTRACTION_PROMPT is well-formed."""
    assert isinstance(QUICK_EXTRACTION_PROMPT, str)
    assert "{chunk}" in QUICK_EXTRACTION_PROMPT
    # Quick prompt should be shorter
    assert len(QUICK_EXTRACTION_PROMPT) < len(EXTRACTION_PROMPT)


def test_format_extraction_prompt_full():
    """Test formatting full extraction prompt."""
    chunk = "Instrumental variables address endogeneity bias."

    formatted = format_extraction_prompt(chunk, prompt_type="full")

    assert chunk in formatted
    assert "JSON" in formatted or "json" in formatted
    assert "method" in formatted


def test_format_extraction_prompt_definition():
    """Test formatting definition extraction prompt."""
    chunk = "A valid instrumental variable must satisfy two conditions."

    formatted = format_extraction_prompt(chunk, prompt_type="definition")

    assert chunk in formatted
    assert "definition" in formatted.lower()
    assert "formal" in formatted.lower()


def test_format_extraction_prompt_relationship():
    """Test formatting relationship extraction prompt."""
    chunk = "The DiD method requires the parallel trends assumption."

    formatted = format_extraction_prompt(chunk, prompt_type="relationship")

    assert chunk in formatted
    assert "method" in formatted.lower()
    assert "relationship" in formatted.lower()


def test_format_extraction_prompt_quick():
    """Test formatting quick extraction prompt."""
    chunk = "This is a test chunk."

    formatted = format_extraction_prompt(chunk, prompt_type="quick")

    assert chunk in formatted
    assert len(formatted) < 500  # Quick prompt should be short


def test_format_extraction_prompt_default():
    """Test formatting with default prompt type."""
    chunk = "Test text"

    # Should default to full extraction
    formatted = format_extraction_prompt(chunk)

    assert chunk in formatted
    assert "EXTRACTION_PROMPT" in formatted or "concept" in formatted.lower()


def test_format_extraction_prompt_invalid_type():
    """Test formatting with invalid prompt type falls back to default."""
    chunk = "Test text"

    formatted = format_extraction_prompt(chunk, prompt_type="invalid_type")

    # Should fallback to full extraction
    assert chunk in formatted
    assert "concept" in formatted.lower()


def test_all_prompts_have_chunk_placeholder():
    """Test all prompt templates have {chunk} placeholder."""
    prompts = [
        EXTRACTION_PROMPT,
        DEFINITION_EXTRACTION_PROMPT,
        METHOD_RELATIONSHIP_PROMPT,
        QUICK_EXTRACTION_PROMPT,
    ]

    for prompt in prompts:
        assert "{chunk}" in prompt


def test_prompts_mention_json_output():
    """Test all prompts mention JSON output format."""
    prompts = [
        EXTRACTION_PROMPT,
        DEFINITION_EXTRACTION_PROMPT,
        METHOD_RELATIONSHIP_PROMPT,
        QUICK_EXTRACTION_PROMPT,
    ]

    for prompt in prompts:
        assert "JSON" in prompt or "json" in prompt


def test_extraction_prompt_has_confidence_scores():
    """Test extraction prompt requests confidence scores."""
    assert '"confidence"' in EXTRACTION_PROMPT
    assert "0.0-1.0" in EXTRACTION_PROMPT or "confidence" in EXTRACTION_PROMPT.lower()


def test_extraction_prompt_has_evidence():
    """Test extraction prompt requests evidence for relationships."""
    assert '"evidence"' in EXTRACTION_PROMPT


def test_format_extraction_prompt_preserves_special_characters():
    """Test formatting handles special characters in chunk text."""
    chunk = 'Text with "quotes" and {braces} and $math$'

    formatted = format_extraction_prompt(chunk, prompt_type="full")

    assert chunk in formatted


def test_format_extraction_prompt_multiline_text():
    """Test formatting handles multiline chunk text."""
    chunk = """This is a multiline chunk.
It has several lines.
Including math notation: Y = βX + ε"""

    formatted = format_extraction_prompt(chunk, prompt_type="full")

    assert chunk in formatted
    assert "Y = βX + ε" in formatted


def test_system_prompt_sets_expert_persona():
    """Test SYSTEM_PROMPT establishes expert persona."""
    assert "expert" in SYSTEM_PROMPT.lower()
    assert (
        "causal inference" in SYSTEM_PROMPT.lower()
        or "econometrics" in SYSTEM_PROMPT.lower()
    )


def test_extraction_prompt_provides_guidelines():
    """Test EXTRACTION_PROMPT includes extraction guidelines."""
    assert (
        "GUIDELINES" in EXTRACTION_PROMPT or "guidelines" in EXTRACTION_PROMPT.lower()
    )


def test_prompts_request_conservative_extraction():
    """Test prompts encourage conservative/precise extraction."""
    # At least one prompt should mention being conservative or precise
    all_prompts = SYSTEM_PROMPT + EXTRACTION_PROMPT

    assert "conservative" in all_prompts.lower() or "precise" in all_prompts.lower()


def test_extraction_prompt_covers_domain():
    """Test EXTRACTION_PROMPT covers causal inference domain."""
    domain_terms = ["IV", "DiD", "regression", "matching", "endogen"]

    # Should mention at least some domain-specific examples
    found_terms = sum(1 for term in domain_terms if term in EXTRACTION_PROMPT)
    assert found_terms >= 2


def test_relationship_types_are_uppercase():
    """Test relationship types are consistently uppercase."""
    relationship_types = [
        "REQUIRES",
        "USES",
        "ADDRESSES",
        "GENERALIZES",
        "SPECIALIZES",
        "ALTERNATIVE_TO",
        "EXTENDS",
    ]

    for rel_type in relationship_types:
        # Should appear in uppercase (not lowercase)
        assert rel_type in EXTRACTION_PROMPT
        assert (
            rel_type.lower() not in EXTRACTION_PROMPT or rel_type in EXTRACTION_PROMPT
        )


def test_format_extraction_prompt_empty_chunk():
    """Test formatting with empty chunk."""
    formatted = format_extraction_prompt("", prompt_type="full")

    # Should still be valid prompt, just with empty text section
    assert isinstance(formatted, str)
    assert "JSON" in formatted or "json" in formatted
