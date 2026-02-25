# Assumption Tracking Skill

This skill teaches agents how to extract, validate, and track assumptions in causal inference research.

## When to Use

### Before Implementation
- Identifying required assumptions for a method
- Checking if assumptions are satisfied in your context
- Understanding testability of assumptions

### During Implementation
- Documenting assumption violations
- Implementing robustness checks
- Sensitivity analysis design

### During Review
- Auditing assumption coverage
- Finding how-to-check procedures
- Identifying potential violations

## What Qualifies as an Assumption

### Identification Assumptions

**Core characteristics:**
- Required for causal identification
- Often untestable (maintained by design/argument)
- Mathematically formalized
- Failure → biased estimates

**Examples:**
- Unconfoundedness (CIA)
- Parallel trends
- Exclusion restriction
- Exogeneity
- SUTVA

### Statistical Assumptions

**Core characteristics:**
- Related to estimation/inference
- Often testable
- About data generating process
- Failure → invalid inference

**Examples:**
- Homoskedasticity
- No serial correlation
- Normally distributed errors
- IID observations

## Extraction Patterns

### Recognition Signals

Assumptions appear with phrases like:

**Strong signals:**
- "We assume that..."
- "Under the assumption of..."
- "Requires [assumption] to hold..."
- "If [assumption] is satisfied..."
- "[Method] relies on..."

**Implicit signals:**
- "Given that treatment is as-if random..."
- "Assuming no spillovers..."
- "In the absence of unmeasured confounding..."

### Structured Extraction

```python
from research_kb_extraction import ConceptExtractor
from research_kb_contracts import ConceptType

# Extract assumptions
chunk_text = """
Difference-in-differences requires the parallel trends assumption:
in the absence of treatment, the treatment and control groups would
have followed the same trend over time. This assumption is inherently
untestable but can be assessed through pre-treatment trend tests.
"""

extraction = await extractor.extract_from_chunk(chunk_text)

# Filter for assumptions
assumptions = [
    c for c in extraction.concepts
    if c.concept_type == ConceptType.ASSUMPTION
]

# Result:
# assumptions[0] = ExtractedConcept(
#     name="parallel trends",
#     canonical_name="parallel trends",
#     concept_type="assumption",
#     definition="Treatment and control would follow same trend absent treatment",
#     aliases=["parallel trends assumption", "common trends"],
#     confidence=0.91,
#     metadata={
#         "is_testable": false,
#         "testability_note": "Inherently untestable",
#         "common_tests": ["pre-treatment trend test"],
#     }
# )
```

## Assumption Schema

### Core Fields

```python
from research_kb_storage import ConceptStore, AssumptionStore
from research_kb_contracts import ConceptType

# Create assumption concept
assumption = await ConceptStore.create(
    name="unconfoundedness",
    canonical_name="unconfoundedness",
    concept_type=ConceptType.ASSUMPTION,
    category="identification",
    definition="Potential outcomes independent of treatment conditional on covariates",
    aliases=["CIA", "conditional independence", "ignorability", "selection on observables"],
    confidence_score=0.95,
)

# Create specialized assumption record
await AssumptionStore.create(
    concept_id=assumption.id,
    mathematical_statement="(Y(1), Y(0)) ⊥ D | X",
    is_testable=False,
    common_tests=[],
    violation_consequences="Biased treatment effect estimates; direction depends on confounding structure"
)
```

### Extended Metadata

```python
assumption_metadata = {
    "is_testable": False,
    "testability_note": "Fundamentally untestable; requires no unmeasured confounding",
    "common_tests": [],  # Empty if untestable
    "violation_consequences": "Biased ATE estimates",
    "plausibility_checks": [
        "Balance tests on observables",
        "Sensitivity analysis (e.g., Rosenbaum bounds)",
        "Placebo tests on pre-treatment outcomes"
    ],
    "required_by_methods": [
        "propensity score matching",
        "regression adjustment",
        "inverse probability weighting"
    ],
    "alternative_assumptions": [
        "Conditional independence given propensity score"
    ]
}
```

## How-to-Check Procedures

### Testable Assumptions

For testable assumptions, document **concrete procedures**:

#### Example: Parallel Trends

```python
assumption_metadata = {
    "is_testable": True,
    "testability_note": "Pre-treatment trends can be tested",
    "common_tests": [
        "Event study / dynamic DiD",
        "Pre-treatment trend test (joint F-test)",
        "Placebo test on pre-treatment periods"
    ],
    "how_to_check": {
        "event_study": {
            "procedure": "Estimate dynamic treatment effects for pre-treatment periods",
            "null_hypothesis": "Pre-treatment coefficients jointly equal zero",
            "test_statistic": "F-statistic on pre-treatment leads",
            "interpretation": "Rejection suggests trends differ before treatment",
            "code_example": "did_reg = smf.ols('y ~ treatment * time + ... + pre_leads', data).fit()"
        },
        "visual_check": {
            "procedure": "Plot treatment vs control trends over time",
            "what_to_look_for": "Parallel paths before treatment date",
            "red_flags": "Diverging trends, level shifts before treatment"
        }
    }
}
```

#### Example: Relevance (First Stage)

```python
assumption_metadata = {
    "is_testable": True,
    "testability_note": "First-stage strength is directly testable",
    "common_tests": [
        "F-statistic > 10 (rule of thumb)",
        "Effective F-statistic (Olea-Pflueger, 2013)",
        "First-stage regression"
    ],
    "how_to_check": {
        "first_stage_regression": {
            "procedure": "Regress endogenous variable on instrument + controls",
            "test_statistic": "F-statistic on instrument coefficient",
            "rule_of_thumb": "F > 10 indicates strong instrument",
            "interpretation": "Low F suggests weak instrument problem",
            "code_example": "first_stage = smf.ols('D ~ Z + X', data).fit(); print(first_stage.f_pvalue)"
        }
    }
}
```

### Untestable Assumptions

For untestable assumptions, document **plausibility arguments**:

#### Example: Exclusion Restriction

```python
assumption_metadata = {
    "is_testable": False,
    "testability_note": "Cannot test exclusion; maintained by design",
    "plausibility_checks": [
        "Institutional knowledge of instrument",
        "Placebo tests on auxiliary outcomes",
        "Overidentification tests (multiple instruments)"
    ],
    "how_to_argue": {
        "institutional_argument": {
            "approach": "Show instrument only affects outcome through treatment",
            "example": "Draft lottery number only affects outcomes via military service",
            "required_evidence": "Detailed knowledge of instrument assignment mechanism"
        },
        "placebo_tests": {
            "approach": "Test instrument effect on outcomes it shouldn't affect",
            "example": "Lottery number shouldn't affect outcomes before draft age",
            "interpretation": "Significant placebo effects suggest exclusion violation"
        }
    }
}
```

## Linking Assumptions to Methods

### Requirement Tracking

```python
from research_kb_storage import ConceptStore, RelationshipStore
from research_kb_contracts import RelationshipType

# Get method
iv_method = await ConceptStore.get_by_canonical_name("instrumental variables")

# Find required assumptions
required = await RelationshipStore.list_from_concept(
    iv_method.id,
    relationship_type=RelationshipType.REQUIRES
)

# Result: IV requires relevance, exclusion restriction, exogeneity
for rel in required:
    assumption = await ConceptStore.get_by_id(rel.target_concept_id)
    print(f"IV requires: {assumption.name}")
```

### Reverse Lookup: Find Methods Needing Assumption

```python
# Get assumption
unconf = await ConceptStore.get_by_canonical_name("unconfoundedness")

# Find methods that require it
requiring_methods = await RelationshipStore.list_to_concept(
    unconf.id,
    relationship_type=RelationshipType.REQUIRES
)

# Result: PSM, matching, regression adjustment all require unconfoundedness
for rel in requiring_methods:
    method = await ConceptStore.get_by_id(rel.source_concept_id)
    print(f"{method.name} requires unconfoundedness")
```

## CLI Usage for Assumption Tracking

### Find Assumption by Name

```bash
# Look up specific assumption
research-kb graph concepts "unconfoundedness"

# Output:
# [1] Unconfoundedness
#     Type: assumption
#     Category: identification
#     Aliases: CIA, conditional independence, ignorability
#     Confidence: 0.94
#     Definition: Potential outcomes independent of treatment conditional on X
#     Relationships (5):
#       ← REQUIRES (from propensity score matching)
#       ← REQUIRES (from regression adjustment)
#       ← REQUIRES (from matching)
```

### Find Assumptions for a Method

```bash
# Get method's required assumptions
research-kb graph neighborhood "instrumental variables" --type REQUIRES --hops 1

# Output:
# CENTER: Instrumental Variables (method)
# Connected concepts (3):
#   [1] Relevance (assumption)
#   [2] Exclusion Restriction (assumption)
#   [3] Exogeneity (assumption)
# Relationships (3):
#   Instrumental Variables -[REQUIRES]-> Relevance
#   Instrumental Variables -[REQUIRES]-> Exclusion Restriction
#   Instrumental Variables -[REQUIRES]-> Exogeneity
```

### Find Path Between Method and Problem via Assumptions

```bash
# Understand why method works for problem
research-kb graph path "instrumental variables" "endogeneity"

# Output:
# Path length: 1 hop(s)
# START: Instrumental Variables (method)
#   ↓ [ADDRESSES]
#   Endogeneity (problem)
# END: Endogeneity
```

## Violation Consequences

### Documentation Template

For each assumption, document:

1. **What happens if violated**
   - Direction of bias (if known)
   - Magnitude factors
   - Estimand changes

2. **Severity assessment**
   - Minor violation → small bias
   - Major violation → invalid inference

3. **Detection methods**
   - Tests for violations
   - Diagnostic plots
   - Sensitivity analysis

#### Example: Parallel Trends Violation

```python
violation_documentation = {
    "assumption": "parallel trends",
    "violation_scenario": "Treatment and control have different pre-treatment trends",
    "consequences": {
        "bias_direction": "Depends on trend differential sign",
        "bias_magnitude": "Proportional to trend differential × time since treatment",
        "estimand_change": "DiD captures trend differential, not treatment effect"
    },
    "detection": {
        "tests": ["Event study", "Pre-treatment trend test"],
        "visual": "Plot treatment vs control trends",
        "sensitivity": "Rambachan & Roth (2023) honest DiD"
    },
    "remedies": [
        "Include group-specific time trends",
        "Use alternative control group",
        "Synthetic control methods",
        "Change differences approach"
    ]
}
```

## Best Practices

### 1. Complete Assumption Coverage

For every method in knowledge base:
- Document all required assumptions
- Create REQUIRES edges
- Include testability info
- Add how-to-check procedures

### 2. Assumption Verification Workflow

When implementing a method:
```python
# 1. Find method's assumptions
method = await ConceptStore.get_by_canonical_name("difference-in-differences")
assumptions = await get_required_assumptions(method.id)

# 2. For each assumption, check testability
for assumption in assumptions:
    metadata = assumption.metadata
    if metadata.get("is_testable"):
        # Run tests
        tests = metadata["common_tests"]
        for test in tests:
            print(f"Run: {test}")
    else:
        # Make plausibility argument
        checks = metadata.get("plausibility_checks", [])
        for check in checks:
            print(f"Argue: {check}")

# 3. Document violations and robustness
```

### 3. Sensitivity Analysis Integration

Link assumptions to sensitivity methods:

```python
assumption_metadata = {
    "assumption": "unconfoundedness",
    "sensitivity_methods": [
        {
            "method": "Rosenbaum bounds",
            "reference": "Rosenbaum (2002)",
            "software": "rbounds (R)",
            "purpose": "Assess sensitivity to hidden bias"
        },
        {
            "method": "Oster (2019) coefficient stability",
            "reference": "Oster (2019) JBES",
            "software": "psacalc (Stata)",
            "purpose": "Bound treatment effect under violations"
        }
    ]
}
```

## Python API Reference

### Query Assumptions

```python
from research_kb_storage import ConceptStore, RelationshipStore
from research_kb_contracts import ConceptType, RelationshipType

# Get all assumptions
assumptions = await ConceptStore.list_by_type(ConceptType.ASSUMPTION)

# Get assumptions for specific method
method = await ConceptStore.get_by_canonical_name("propensity score matching")
requirements = await RelationshipStore.list_from_concept(
    method.id,
    relationship_type=RelationshipType.REQUIRES
)

# Get specialized assumption data
from research_kb_storage import AssumptionStore
assumption_details = await AssumptionStore.get_by_concept_id(assumption.id)
print(assumption_details.mathematical_statement)
print(f"Testable: {assumption_details.is_testable}")
print(f"Tests: {assumption_details.common_tests}")
```

### Extract from Text

```python
from research_kb_extraction import ConceptExtractor

# Extract with assumption focus
chunk = """
The synthetic control method requires that the synthetic control unit
closely approximates the treated unit in pre-treatment periods. This
is verified by comparing pre-treatment fit.
"""

extraction = await extractor.extract_from_chunk(chunk)
assumptions = [c for c in extraction.concepts if c.concept_type == "assumption"]

# Check for testability signals
for assumption in assumptions:
    if "verified" in chunk.lower() or "test" in chunk.lower():
        assumption.metadata["is_testable"] = True
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Missing assumptions | Extract from method papers, especially identification sections |
| Unclear testability | Check econometrics textbooks (MHE, Pearl) |
| No how-to-check | Search applied papers for common tests |
| Conflicting info | Prefer canonical sources (Pearl, Angrist/Pischke) |
| Test not documented | Add to specialized `AssumptionStore` record |
| Relationship missing | Create REQUIRES edge from method to assumption |

## Example Workflow

### Complete Assumption Documentation for IV

```python
# 1. Extract or create assumptions
relevance = await ConceptStore.create(
    name="relevance",
    canonical_name="relevance",
    concept_type=ConceptType.ASSUMPTION,
    aliases=["instrument relevance", "first stage strength"],
    definition="Instrument must be correlated with endogenous variable",
)

# 2. Add specialized info
await AssumptionStore.create(
    concept_id=relevance.id,
    mathematical_statement="Cov(Z, D | X) ≠ 0",
    is_testable=True,
    common_tests=["F-statistic > 10", "first stage regression"],
    violation_consequences="Weak instrument bias toward OLS"
)

# 3. Link to method
iv = await ConceptStore.get_by_canonical_name("instrumental variables")
await RelationshipStore.create(
    source_concept_id=iv.id,
    target_concept_id=relevance.id,
    relationship_type=RelationshipType.REQUIRES,
)

# 4. Document how-to-check
relevance.metadata["how_to_check"] = {
    "first_stage": {
        "procedure": "Regress D on Z",
        "test": "F-test on Z coefficient",
        "threshold": "F > 10",
        "code": "first_stage = sm.OLS(D, Z).fit(); print(first_stage.fvalue)"
    }
}
```

## See Also

- **Concept Extraction Skill**: General concept extraction workflow
- **Research Context Retrieval Skill**: Finding assumption documentation
- **PDF Ingestion Skill**: Ensuring good extraction from methods papers
