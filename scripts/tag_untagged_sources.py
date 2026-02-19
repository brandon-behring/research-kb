"""Tag untagged sources with domain metadata.

Classifies 221 sources with null domain tags based on title keywords
and first-chunk content analysis. Dry-run by default; pass --apply to execute.

Also normalizes non-canonical domain names (e.g., "ML" → "machine_learning").

Usage:
    python scripts/tag_untagged_sources.py           # dry run
    python scripts/tag_untagged_sources.py --apply    # apply changes
"""

import asyncio
import re
import sys
from pathlib import Path

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool

# ── Domain normalization: non-canonical → canonical ──────────────────────────

DOMAIN_NORMALIZATION: dict[str, str] = {
    "econometrics/ML": "econometrics",
    "ML": "machine_learning",
    "statistical methods": "statistics",
    "investment analysis": "finance",
    "portfolio management": "portfolio_management",
    "causal discovery": "causal_inference",
    "knowledge representation": "rag_llm",
    "data_mining": "data_science",
    "experimentation": "causal_inference",
}

# ── Title-based classification rules (checked in order) ─────────────────────
# Each rule: (compiled regex on title, domain)
# First match wins.

TITLE_RULES: list[tuple[re.Pattern, str]] = [
    # Causal inference
    (re.compile(r"causal|CUPED|doubly robust|instrumental variable|IV\b|treatment effect|synthetic control|experiment(?:s|ation)|RCT|A/B test|network experiment|confound", re.I), "causal_inference"),
    (re.compile(r"Huntington.Klein|Cunningham.+Mixtape|Brady Neal|Judea Pearl|Rubin.*Bayesian.*Causal|Manning Causal Inference|Wager.*Causal|Chai.*Double M", re.I), "causal_inference"),
    # Time series
    (re.compile(r"time.series|forecast(?:ing)?|volatil|GARCH|ARIMA|Box.Jenkins|unit.root|serial.correlation|HAC|Diebold.Mariano|Tashman|Pesaran.Timmermann|DEPTS|Lag-Llama|TSMixer|Ti-MAE|temporal.+neural", re.I), "time_series"),
    (re.compile(r"Shumway.*Stoffer|Aileen Nielsen|Bergmeir|Andrews.*Bandwidth|Newey.West|Said.Dickey|Harvey.*Adjustment|Goodhart.*Interest Rate|Conformal.*Time Series|Conformal PID", re.I), "time_series"),
    # RAG / LLM
    (re.compile(r"RAG|retriev.+augment|LLM|large language model|GPT|Llama\b|hallucin|tool.?use|text.to.sql|knowledge graph|GraphRAG|chatbot|RAIN.*align|BeaverTails|PandaLM|EduChat|ToolLLM|ART.*multi.step.*reason|Building AI Agents.*LLM|Intro.?To Knowledge Graphs", re.I), "rag_llm"),
    (re.compile(r"Openai.*Technical|Touvron.*Llama|SMART.LLM", re.I), "rag_llm"),
    # Deep learning
    (re.compile(r"deep.learn|neural.network|transformer(?!.*time)|attention.mechanism|self.supervised|diffusion.model|vision|ViT|image|video|Paint Transformer|GNN|graph neural", re.I), "deep_learning"),
    (re.compile(r"Santanu Pattanayak|Dosovitskiy", re.I), "deep_learning"),
    # Machine learning
    (re.compile(r"machine.learn|xgboost|ensemble.method|imbalanced.class|causal.forest|regression.tree|random.forest|genetic.algorithm|reinforcement.learn|conformal.predict(?!.*time)|agent.based", re.I), "machine_learning"),
    (re.compile(r"Shalev.Shwartz|Deisenroth.*Mathematics|optimization.*machine learning|machine learning mastery|machine learning algorithms", re.I), "machine_learning"),
    # Statistics
    (re.compile(r"statistic(?:s|al)|bayesian|biostat|binomial|probability|multilevel|hierarchical.+model|Agresti|Wasserman|openintro", re.I), "statistics"),
    (re.compile(r"computer.age.statistical|probabilistic.graphical|Koller.*Friedman|Lee.*Bayesian|Donovan.*Bayesian", re.I), "statistics"),
    # Finance
    (re.compile(r"option.pricing|Black.Scholes|Hull.*Derivative|Bitcoin|equity.*annuit|investment.guarantee|credit.*model|FinRegLab|fintech", re.I), "finance"),
    (re.compile(r"Boyle.*Annuit|Hardy.*Guarantee|Responsible.+Credit|Wu.*Fintech", re.I), "finance"),
    # Econometrics (specific papers/books not caught by causal)
    (re.compile(r"econometric|The Effect(?!.*COVID)|COVID.*lockdown|vaccination|PM2\.5|air.pollu", re.I), "econometrics"),
    # Software engineering
    (re.compile(r"API.design|web.API|cloud.platform|GitHub.Action|software.developer|software.engineering|ML.pipeline|data.science.on.AWS|TensorFlow|data.structure|algorithm(?:s)?.illuminat|problem.solving.with.algo|programming.with", re.I), "software_engineering"),
    (re.compile(r"Geewax|Chris Fregly|Hannes Hapke|Jay Wengrow|Scopatz.*Physics|ScalaByExample|Skills of a Software|Nick Alteen|Moroney.*Coders|Ameisen.*Building|Ryan.*White.*Distributed|Miller.*Problem Solving|Roughgarden.*Algorithms|Kulikov.*Puzzle", re.I), "software_engineering"),
    # Data science
    (re.compile(r"data.science|getting.started.with.data", re.I), "data_science"),
    (re.compile(r"Murtaza Haider", re.I), "data_science"),
    # Mathematics
    (re.compile(r"linear.algebra|differential.equation|numerical.method|ODE|quantum.mechanic|quantum.field|quantum.theory|classical.dynamic|classical.mechanic|math.programming|multivari.+calculus|complex.analysis|spectral.method", re.I), "mathematics"),
    (re.compile(r"Strang.*Linear|Lay.*Linear|Trefethen|Boyd.*Vandenberghe|Aggarwal.*Linear|Lipschutz|Sakurai|Griffiths.*Schroeter|Goldstein.*Poole|Klauber.*Quantum|Arnold.*ODE|Chicone|Bronson.*Differential|Hubbard.*Schaum|Miller.*Quantum|José.*Saletan|quantitative.economics.*julia", re.I), "mathematics"),
    # Physics (specialized, beyond math)
    (re.compile(r"quantum.field.theory.*gifted|strength.training|locomotion|robotic|grasping", re.I), "physics"),
    # Algorithms (if not caught by SE)
    (re.compile(r"Kochenderfer.*Algorithms|community.detect|Traag.*Algorithm|graph.algorithm", re.I), "algorithms"),
    # Network science / graph algorithms
    (re.compile(r"network.data|Kolaczyk", re.I), "data_science"),
    # Functional programming
    (re.compile(r"Haskell|functional.programming", re.I), "functional_programming"),
    # Conformal prediction (specific)
    (re.compile(r"conformal.+(?!time)", re.I), "statistics"),
]


# ── Manual overrides for sources that defy keyword matching ─────────────────
# source_id → domain
MANUAL_OVERRIDES: dict[str, str] = {
    # All UUIDs verified against database — no guessed suffixes
    # Fitness
    "36a6155d-d3c3-4ead-9fa6-ded7ebe770ea": "fitness",  # Fry/Zatsiorsky Strength Training
    "996a1d70-c08f-4b20-8f42-6bb3b7dab564": "fitness",  # Class/Health Strength Training
    # COVID / epidemiology
    "15112a69-169b-44ab-b847-67d82fd1d2e9": "econometrics",  # COVID-19 certificates
    "dae85cf3-7612-412b-8805-71787a665955": "econometrics",  # Wuhan lockdown
    # Medical / health
    "8ab6dbf0-c6f2-4c32-8b3b-460ed7a5c94e": "statistics",  # Brucellosis incidence
    "760a860c-766a-46a3-afbc-c7c412161016": "statistics",  # Perinatal management
    "b4b5a171-6f6b-44a3-9979-e8b5565af7a1": "statistics",  # Hepatological guidelines
    "b9dbb4ec-7f69-44cf-a049-d5b42912730e": "time_series",  # Heart rate TS
    # Specific books
    "7468defd-1d99-41dd-9e4f-7b32786dc8aa": "data_science",  # Railsback Agent-Based Modeling
    "6cde85f8-6dba-48f9-81dd-243c60e8b973": "data_science",  # Thurner Complex Systems
    "e44e7288-e5e6-491f-a12c-95aeb9e12904": "mathematics",  # MathProgrammingIntro
    "946cdd81-b5d4-4ec3-88f2-3aab1ecf96c9": "mathematics",  # quantitative economics julia
    "4f5c7b2e-4260-4de3-bcab-c0f8585a62d6": "mathematics",  # V.I. Arnold (GTM60)
    "12887d0d-3e3f-40e9-aa33-e98619ab9994": "mathematics",  # QT for Mathematicians
    "fb2c975b-7823-4806-a265-3b0cb5d7be0d": "mathematics",  # QFT Gifted Amateur
    "1d488389-c413-4e46-80da-3be6c5bb4fca": "mathematics",  # Undergrad Lecture Notes Physics
    "01438118-1bb6-4879-912a-2b8afabdc8b8": "mathematics",  # Monographs Math Modeling
    # Robotics / RL
    "bb01bfce-3b21-46ab-add3-680e1f4497d5": "deep_learning",  # Walk in the Park (RL)
    "b7179446-917f-457c-92e5-84a86c41c9a6": "deep_learning",  # Humanoid locomotion RL
    "f0b427f3-cc2a-4af6-810c-ec34cc676071": "deep_learning",  # Transformer grasping
    "3aa7b1b7-7b18-4b44-b330-949f9fca6145": "deep_learning",  # CrystalGPT
    # Security
    "4fcc66d4-7d0a-45ea-94b9-12aa1af1b23b": "software_engineering",  # DL Hybrid Security
    # Fairness / bias
    "3c663f59-7ed0-4cc7-b48e-48c77cfdfad5": "machine_learning",  # Fairness in AI
    "602009db-1948-4839-8d22-71a0f69652df": "machine_learning",  # Beyond bias/discrimination
    "e00a5d88-550a-42ca-9170-f38e3c0679c5": "machine_learning",  # Bias in ML Models
    "12b16683-7f79-4942-932b-5f0adce1cb41": "machine_learning",  # Connecting Dots Trustworthy AI
    # Graph / causal specific
    "adf274fd-a2da-4fae-adfe-7aec0176fe2f": "algorithms",  # Manning Graph Algorithms
    "82f50a9c-530d-44a9-96f6-399ed0823caa": "causal_inference",  # Manning Causal Inference
    "a2e2d729-3730-4120-b155-28dfe5a7d0a4": "econometrics",  # The Effect Huntington-Klein
    "5f5904c8-d5ea-4e74-8c4e-d8dec89e7934": "causal_inference",  # Towards optimal DR
    "ae5d826b-cc26-49d0-8f70-759c9190e750": "causal_inference",  # 2112.08417 Gerhardus
    "ffe2d593-c4ee-4c4a-a12b-d89c2836c602": "causal_inference",  # Etsy Pre In VR 2024
    # Remaining from dry-run
    "7d34c695-3387-4f66-89df-4a1860fd0d8c": "machine_learning",  # ensemble learning algorithms
    "a14c2da8-d578-407e-8900-6cdb02c39eee": "machine_learning",  # Li, Benkai Systems Decision
    # ArXiv papers with opaque first chunks
    "584cb567-1214-4d6b-8fa5-4298c47e5221": "causal_inference",  # 1706.09523 BCF
    "08d4cb2e-d5bf-49bb-96a5-bca8555b7c55": "deep_learning",  # 2210.03629 (RL agent)
    "a049ba0f-d8af-426e-8fa2-0de46eff7c27": "deep_learning",  # 2302.10866 (foundation model)
}


async def identify_arxiv_papers(pool) -> dict[str, str]:
    """Read first chunk of arXiv papers to classify them by content."""
    arxiv_rows = await pool.fetch(
        "SELECT s.id, s.title, "
        "(SELECT c.content FROM chunks c WHERE c.source_id = s.id ORDER BY c.page_start NULLS LAST, c.id LIMIT 1) as first_chunk "
        "FROM sources s "
        "WHERE s.title LIKE 'arXiv:%' AND (s.metadata->>'domain' IS NULL OR s.metadata->>'domain' = 'none') "
        "ORDER BY s.title"
    )

    classifications: dict[str, str] = {}
    for row in arxiv_rows:
        sid = str(row["id"])
        chunk = (row["first_chunk"] or "").lower()
        title = row["title"]

        # Try to classify from chunk content
        if any(kw in chunk for kw in ["causal", "treatment effect", "instrumental", "confound", "potential outcome"]):
            classifications[sid] = "causal_inference"
        elif any(kw in chunk for kw in ["time series", "forecast", "temporal", "volatil", "arima", "lstm"]):
            classifications[sid] = "time_series"
        elif any(kw in chunk for kw in ["retrieval", "augment", "language model", "llm", "gpt", "prompt", "rag"]):
            classifications[sid] = "rag_llm"
        elif any(kw in chunk for kw in ["transformer", "attention", "bert", "pre-train", "self-supervis", "diffusion", "neural network", "deep learn"]):
            classifications[sid] = "deep_learning"
        elif any(kw in chunk for kw in ["reinforcement learn", "robot", "locomot", "grasp"]):
            classifications[sid] = "deep_learning"
        elif any(kw in chunk for kw in ["machine learn", "classif", "regression", "ensemble", "random forest"]):
            classifications[sid] = "machine_learning"
        elif any(kw in chunk for kw in ["conformal", "prediction interval", "coverage"]):
            classifications[sid] = "statistics"
        else:
            # Unknown — leave untagged for now, print for manual review
            preview = chunk[:200].replace("\n", " ").strip()
            print(f"  UNKNOWN arXiv: {title} → {preview[:100]}...")

    return classifications


async def main(apply: bool = False):
    pool = await get_connection_pool(DatabaseConfig())

    # Phase 1: Normalize non-canonical domains
    print("=" * 60)
    print("Phase 1: Domain Normalization")
    print("=" * 60)
    norm_count = 0
    for old, new in DOMAIN_NORMALIZATION.items():
        count = await pool.fetchval(
            "SELECT COUNT(*) FROM sources WHERE metadata->>'domain' = $1", old
        )
        if count > 0:
            print(f"  {old:30s} → {new:25s} ({count} sources)")
            norm_count += count
            if apply:
                await pool.execute(
                    "UPDATE sources SET metadata = jsonb_set(metadata, '{domain}', $1::jsonb) "
                    "WHERE metadata->>'domain' = $2",
                    f'"{new}"', old,
                )

    if norm_count == 0:
        print("  No normalization needed.")
    else:
        print(f"\n  {'Applied' if apply else 'Would apply'} {norm_count} normalization(s).")

    # Phase 2: Classify arXiv papers by content
    print()
    print("=" * 60)
    print("Phase 2: ArXiv Paper Classification (by first chunk)")
    print("=" * 60)
    arxiv_map = await identify_arxiv_papers(pool)
    print(f"  Classified {len(arxiv_map)} arXiv paper(s) by content.")

    # Phase 3: Title-based classification for all untagged
    print()
    print("=" * 60)
    print("Phase 3: Title-Based Classification")
    print("=" * 60)

    untagged = await pool.fetch(
        "SELECT id, title FROM sources "
        "WHERE metadata->>'domain' IS NULL OR metadata->>'domain' = 'none' "
        "ORDER BY title"
    )

    classified: dict[str, str] = {}  # source_id → domain
    unclassified: list[tuple[str, str]] = []  # (source_id, title)

    for row in untagged:
        sid = str(row["id"])
        title = row["title"]

        # Check manual overrides first
        if sid in MANUAL_OVERRIDES:
            classified[sid] = MANUAL_OVERRIDES[sid]
            continue

        # Check arXiv content classification
        if sid in arxiv_map:
            classified[sid] = arxiv_map[sid]
            continue

        # Try title rules
        matched = False
        for pattern, domain in TITLE_RULES:
            if pattern.search(title):
                classified[sid] = domain
                matched = True
                break

        if not matched:
            unclassified.append((sid, title))

    # Summary
    by_domain: dict[str, list[str]] = {}
    for sid, domain in classified.items():
        by_domain.setdefault(domain, []).append(sid)

    for domain in sorted(by_domain.keys()):
        sids = by_domain[domain]
        print(f"  {domain:30s} → {len(sids)} source(s)")

    print(f"\n  Total classified: {len(classified)}")
    print(f"  Unclassified:     {len(unclassified)}")

    if unclassified:
        print("\n  Unclassified sources:")
        for sid, title in unclassified:
            print(f"    {sid[:8]}... | {title[:70]}")

    # Phase 4: Apply updates
    if apply and classified:
        print(f"\n  Applying {len(classified)} domain tag(s)...")
        async with pool.acquire() as conn:
            for sid, domain in classified.items():
                await conn.execute(
                    "UPDATE sources SET metadata = jsonb_set("
                    "  COALESCE(metadata, '{}'::jsonb), '{domain}', $1::jsonb"
                    ") WHERE id = $2",
                    f'"{domain}"', __import__("uuid").UUID(sid),
                )
        print("  Done.")
    elif not apply and classified:
        print(f"\n  DRY RUN: Would apply {len(classified)} domain tag(s).")
        print("  Re-run with --apply to execute.")

    # Final stats
    if apply:
        print()
        remaining = await pool.fetchval(
            "SELECT COUNT(*) FROM sources "
            "WHERE metadata->>'domain' IS NULL OR metadata->>'domain' = 'none'"
        )
        print(f"  Remaining untagged: {remaining}")

    await pool.close()


if __name__ == "__main__":
    do_apply = "--apply" in sys.argv
    asyncio.run(main(apply=do_apply))
