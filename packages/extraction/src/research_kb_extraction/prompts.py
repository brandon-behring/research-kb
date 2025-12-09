"""Prompt templates for concept extraction.

These prompts are designed to work with Ollama's JSON mode for
structured output. They guide the LLM to extract concepts and
relationships from causal inference / econometrics text.
"""

# System prompt establishing the extraction task
SYSTEM_PROMPT = """You are an expert in causal inference, econometrics, and machine learning.
Your task is to extract structured knowledge from academic text.

You will identify:
1. CONCEPTS: Methods, assumptions, problems, definitions, and theorems
2. RELATIONSHIPS: How concepts relate to each other

Be precise and conservative - only extract what is clearly stated or strongly implied.
Output must be valid JSON matching the specified schema."""

# Main extraction prompt template
EXTRACTION_PROMPT = """Analyze the following text chunk from an academic paper or textbook.

Extract:
1. All concepts mentioned (methods, assumptions, problems, definitions, theorems)
2. Relationships between concepts

TEXT CHUNK:
---
{chunk}
---

CONCEPT TYPES (USE ONLY THESE):
You MUST use ONLY one of these exact values for concept_type:
1. method: Statistical/econometric methods (e.g., IV, DiD, matching, regression)
2. assumption: Required conditions for validity (e.g., parallel trends, unconfoundedness)
3. problem: Issues methods address (e.g., endogeneity, selection bias, confounding)
4. definition: Formal definitions of terms
5. theorem: Formal mathematical results

CRITICAL: Do NOT use ANY other concept_type values like "concept", "parameter", "technique", "framework", etc.
Only use: method, assumption, problem, definition, theorem

RELATIONSHIP TYPES (USE ONLY THESE):
You MUST use ONLY one of these exact values for relationship_type:
1. REQUIRES: Method requires an assumption to be valid
2. USES: Method uses another technique/concept
3. ADDRESSES: Method solves or mitigates a problem
4. GENERALIZES: Concept is broader than another (parent)
5. SPECIALIZES: Concept is narrower than another (child)
6. ALTERNATIVE_TO: Concepts are competing approaches
7. EXTENDS: Concept builds upon another

CRITICAL: Do NOT use ANY other relationship_type values like "OUTPUTS", "CONTRIBUTES_TO", "DETERMINES", "SPECIFIES", "DRIVES", etc.
Only use: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS

OUTPUT FORMAT (JSON):
{{
  "concepts": [
    {{
      "name": "concept name as in text",
      "concept_type": "method",
      "definition": "brief definition if provided, null otherwise",
      "aliases": ["alternative names", "abbreviations"],
      "confidence": 0.0-1.0
    }}
  ],
  "relationships": [
    {{
      "source_concept": "concept name",
      "target_concept": "concept name",
      "relationship_type": "REQUIRES",
      "evidence": "quote from text supporting this relationship",
      "confidence": 0.0-1.0
    }}
  ]
}}

GUIDELINES:
- Only extract concepts that are substantively discussed, not just mentioned in passing
- Confidence should reflect how clearly the concept/relationship is defined in the text
- Include aliases like abbreviations (IV, DiD, DML) when mentioned
- Relationships should be supported by evidence in the text
- If no concepts or relationships are found, return empty arrays
- ALWAYS verify your concept_type values are one of: method, assumption, problem, definition, theorem
- ALWAYS verify your relationship_type values are one of: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS

Return ONLY valid JSON, no additional text."""

# Prompt for extracting concepts from a definition-heavy section
DEFINITION_EXTRACTION_PROMPT = """Extract formal definitions from this text.

TEXT:
---
{chunk}
---

Focus on:
- Mathematical definitions
- Formal assumptions with notation
- Theorem statements
- Key terms being defined

IMPORTANT CONSTRAINTS:
- concept_type MUST be one of: definition, assumption, theorem (use ONLY these)
- Do NOT use values like "concept", "parameter", "framework", etc.

OUTPUT FORMAT (JSON):
{{
  "concepts": [
    {{
      "name": "term being defined",
      "concept_type": "definition",
      "definition": "the formal definition text",
      "aliases": [],
      "confidence": 0.9
    }}
  ],
  "relationships": []
}}

Return ONLY valid JSON."""

# Prompt for extracting method relationships
METHOD_RELATIONSHIP_PROMPT = """Analyze relationships between methods in this text.

TEXT:
---
{chunk}
---

Focus on:
- Which methods require which assumptions
- How methods relate to each other (generalizations, alternatives, extensions)
- What problems methods address

IMPORTANT CONSTRAINTS:
- relationship_type MUST be one of: REQUIRES, USES, ADDRESSES, GENERALIZES, SPECIALIZES, ALTERNATIVE_TO, EXTENDS
- Do NOT use values like "OUTPUTS", "CONTRIBUTES_TO", "DETERMINES", "SPECIFIES", "DRIVES", etc.

OUTPUT FORMAT (JSON):
{{
  "concepts": [],
  "relationships": [
    {{
      "source_concept": "method name",
      "target_concept": "related concept",
      "relationship_type": "REQUIRES",
      "evidence": "supporting quote",
      "confidence": 0.8
    }}
  ]
}}

Return ONLY valid JSON."""

# Simplified prompt for quick extraction (lower quality but faster)
QUICK_EXTRACTION_PROMPT = """List the main concepts in this text as JSON.

TEXT: {chunk}

{{
  "concepts": [
    {{"name": "...", "concept_type": "method|assumption|problem|definition|theorem", "confidence": 0.7}}
  ],
  "relationships": []
}}"""


def format_extraction_prompt(chunk: str, prompt_type: str = "full") -> str:
    """Format the appropriate prompt with the chunk text.

    Args:
        chunk: Text chunk to analyze
        prompt_type: One of "full", "definition", "relationship", "quick"

    Returns:
        Formatted prompt string
    """
    prompts = {
        "full": EXTRACTION_PROMPT,
        "definition": DEFINITION_EXTRACTION_PROMPT,
        "relationship": METHOD_RELATIONSHIP_PROMPT,
        "quick": QUICK_EXTRACTION_PROMPT,
    }

    template = prompts.get(prompt_type, EXTRACTION_PROMPT)
    return template.format(chunk=chunk)
