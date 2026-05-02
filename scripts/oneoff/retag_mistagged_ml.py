"""Retag ML-misclassified sources via filesystem-first, title-second, LLM-third.

Surfaces sources currently tagged `machine_learning` whose actual domain
should be derived from (in priority order):

  1. fixtures/library_books/<dir>/  →  LIBRARY_DIR_TO_DOMAIN[dir]   (256 sources)
  2. Title keyword heuristic        →  rich math/physics regex set  (~28 sources)
  3. LLM classification (Ollama)    →  llama3.1:8b on title+path    (~75 sources)

Each retagged source has its original `domain_id` preserved at
`sources.metadata.legacy_domain_id` for auditability.

Usage:
  python scripts/retag_mistagged_ml.py                # dry-run, write /tmp/retag_plan.json
  python scripts/retag_mistagged_ml.py --apply        # write to DB
  python scripts/retag_mistagged_ml.py --skip-llm     # path+title only, fallbacks stay general
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import urllib.error
import urllib.request

_root = Path(__file__).parent.parent
for pkg in ("storage", "contracts", "common"):
    sys.path.insert(0, str(_root / "packages" / pkg / "src"))

from research_kb_storage import DatabaseConfig, get_connection_pool

# ── Tier mapping (mirrors audit_formula_gaps.py) ─────────────────────────────
TIER_1 = {
    "mathematics",
    "linear_algebra",
    "optimization",
    "probability_theory",
    "analysis",
    "topology_geometry",
}
TIER_2 = {
    "statistics",
    "machine_learning",
    "reinforcement_learning",
    "deep_learning",
    "causal_inference",
    "time_series",
    "econometrics",
}
TIER_3 = {
    "physics",
    "engineering",
    "finance",
    "nlp",
    "information_theory",
    "rag_llm",
    "algorithms",
    "computer_science",
    "data_science",
}
TIER_4 = {
    "software_engineering",
    "functional_programming",
    "business",
    "economics",
    "neuroscience",
    "interview_prep",
    "general",
    "fitness",
}
ALL_DOMAINS = TIER_1 | TIER_2 | TIER_3 | TIER_4

# DB has a slightly different canonical set than the audit script. Map
# audit/LLM terms onto the names the `domains` table actually stores.
# Drift between the two taxonomies should be reconciled separately.
DOMAIN_TO_DB = {
    "computer_science": "software_engineering",
    "information_theory": "signal_processing",
    "engineering": "signal_processing",
    "neuroscience": "biology_neuroscience",
    "nlp": "rag_llm",
    "business": "economics",
    "general": "machine_learning",  # last-resort: no 'general' row in DB; flag via metadata.retag_method=fallback
}


def to_db_domain(domain: str) -> str:
    """Translate audit/LLM-canonical name to a name that exists in DB.domains."""
    return DOMAIN_TO_DB.get(domain, domain)


def tier_of(domain: str) -> int:
    if domain in TIER_1:
        return 1
    if domain in TIER_2:
        return 2
    if domain in TIER_3:
        return 3
    if domain in TIER_4:
        return 4
    return 0


# ── Filesystem-first classification ──────────────────────────────────────────
LIBRARY_DIR_TO_DOMAIN = {
    "algebra": "mathematics",
    "algorithms": "algorithms",
    "analysis": "analysis",
    "biology_neuroscience": "neuroscience",
    "causal_inference": "causal_inference",
    "data_science": "data_science",
    "deep_learning": "deep_learning",
    "dynamical_systems": "mathematics",
    "finance": "finance",
    "fitness": "fitness",
    "machine_learning": "machine_learning",
    "mathematics": "mathematics",
    "numerical_methods": "mathematics",
    "optimization": "optimization",
    "physics": "physics",
    "probability_theory": "probability_theory",
    "rag_llm": "rag_llm",
    "reinforcement_learning": "reinforcement_learning",
    "signal_processing": "engineering",
    "software_engineering": "software_engineering",
    "statistics": "statistics",
    "time_series": "time_series",
    "topology_geometry": "topology_geometry",
}

LIBRARY_RE = re.compile(r"/library_books/([^/]+)/")


def classify_by_path(file_path: str | None) -> str | None:
    if not file_path:
        return None
    m = LIBRARY_RE.search(file_path)
    if not m:
        return None
    return LIBRARY_DIR_TO_DOMAIN.get(m.group(1))


# ── Title-keyword heuristic ──────────────────────────────────────────────────
TITLE_RULES: list[tuple[str, str]] = [
    (r"\blinear\s+algebra\b|\bmatrix\s+(theory|analysis)\b|\bmatrices\b", "linear_algebra"),
    (r"\boptim(al|ization|isation)\b|\bconvex\s+optim", "optimization"),
    (
        r"\bprobability\s+(theory|measure|space)\b|\bmeasure\s+theory\b|\bstochastic\s+process|\bmartingale\b",
        "probability_theory",
    ),
    (
        r"\b(real|complex|functional|harmonic|fourier|spectral|operator|hilbert|banach)\s+(analysis|space|theor)|\banalysis\s+(i|ii|iii)\b|\b(banach|hilbert)\s+(space|algebra)\b|\boperator\s+theor|\bergodic\b",
        "analysis",
    ),
    (
        r"\btopolog|\bgeometr|\bmanifold|\blie\s+(group|algebra)\b|\bdifferential\s+geometr|\bhomotopy|\balgebraic\s+(geometr|topolog)|\bhyperbolic\b|\bknot\s+theor|\bcohomology|\bhomology\b|\bsymplectic|\briemannian",
        "topology_geometry",
    ),
    (
        r"\bcalculus\b|\bdifferential\s+equation|\b(ode|pde)\b|\bvariation(al)?\b|\bnumerical\s+(method|analysis)\b|\bnumber\s+theor|\babstract\s+algebra\b|\bcommutative\s+algebra\b|\bcategor(y|ie|ical)|\b(mathematical\s+)?logic\b|\bset\s+theor|\bgalois\b|\bcombinator|\bgraph\s+theor|\bdynamical\s+system|\bchaos\b|\bbifurcation|\bgroup\s+theor|\bring\s+theor|\bproof|\bmath\s+method|\bmathemati",
        "mathematics",
    ),
    (
        r"\bquantum\s+(mechanic|theor|field|electrodynamic|computing|physics)|\bquantum\b|\bclassical\s+(mechanic|dynamic|field)|\bgeneral\s+relativit|\bspecial\s+relativit|\bstatistical\s+(mechanic|physic)|\bthermodynamic|\boptics\b|\belectrodynamic|\bsolid\s+state|\bcondensed\s+matter|\bnuclear\s+physic|\bparticle\s+physic|\bcosmology|\bastrophysic|\bplasma\b|\bastronom|\bfeynman|\bgriffiths|\bjackson\s+electro|\bgoldstein\s+classical|\bsakurai|\blandau\s+lifshitz|\bcohen.tannoudji|\bphysics\b",
        "physics",
    ),
    (
        r"\bcontrol\s+(theory|system|engineering)|\bsignal\s+processing|\binformation\s+theor|\bcoding\s+theor|\bcommunicat\s+system",
        "engineering",
    ),
    (
        r"\bmachine\s+learning\b|\bml\s+for\b|\bdeep\s+learning|\bneural\s+network|\bgradient\s+boost|\brandom\s+forest|\btransformer|\battention\s+(mechanism|is\s+all)|\bgenerative\s+model|\bbayesian\s+(network|reasoning)",
        "machine_learning",
    ),
    (
        r"\breinforcement\s+learning|\bbandit|\bmulti.?armed|\bpolicy\s+gradient|\bq.?learning|\bmarkov\s+decision",
        "reinforcement_learning",
    ),
    (r"\btime.?series|\bforecast|\barima\b|\bbox.?jenkins", "time_series"),
    (r"\beconometric|\bwooldridge|\bgreene\s+economet", "econometrics"),
    (
        r"\bcausal\s+(inference|effect|graph)|\binstrumental\s+variable|\bregression\s+discontinuit|\bsynthetic\s+control|\bdouble.?ml",
        "causal_inference",
    ),
    (
        r"\bstatistic|\bbayesian\b|\bhypothesis\s+test|\bmle\b|\bfrequentist|\bprobability\s+and\s+statist",
        "statistics",
    ),
    (
        r"\bgraphical\s+model|\bkoller\s+friedman|\bpgm\b|\bprobabilistic\s+graphical",
        "machine_learning",
    ),
    (r"\bneuroscienc|\bbrain\b|\bcognitive\s+(scienc|neuro)", "neuroscience"),
]
TITLE_PATTERNS = [(re.compile(p, re.IGNORECASE), d) for p, d in TITLE_RULES]


def classify_by_title(title: str) -> str | None:
    # Normalize: replace underscores with spaces so book titles like
    # "Distributed_Machine_Learning_Patterns" match \bmachine\s+learning\b.
    norm = title.replace("_", " ")
    for pat, dom in TITLE_PATTERNS:
        if pat.search(norm):
            return dom
    return None


# ── LLM fallback (Ollama) ────────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"


LLM_PROMPT_TEMPLATE = """You are a librarian classifying a research textbook or paper into ONE canonical domain.

Title: {title}
File path: {path}

Choose EXACTLY ONE domain from this list. Reply with the domain string only — no other words, no punctuation, no explanation:

{domains}

Choose the most specific applicable domain. If unclear, choose 'general'.
"""


def llm_classify(title: str, path: str) -> tuple[str, str]:
    """Classify via Ollama. Returns (domain, raw_response). 'general' on any failure."""
    domains_block = "\n".join(f"- {d}" for d in sorted(ALL_DOMAINS))
    prompt = LLM_PROMPT_TEMPLATE.format(title=title, path=path or "(none)", domains=domains_block)
    body = json.dumps(
        {
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 32},
        }
    ).encode()

    req = urllib.request.Request(
        OLLAMA_URL, data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        return ("general", f"<llm_error: {e}>")

    raw = (data.get("response") or "").strip().lower()
    # Pick first valid domain mentioned
    for d in ALL_DOMAINS:
        if d in raw:
            return (d, raw)
    return ("general", raw)


# ── Main retag flow ──────────────────────────────────────────────────────────
@dataclass
class Suggestion:
    source_id: str
    title: str
    file_path: str | None
    current_domain: str
    suggested_domain: str
    suggested_tier: int
    method: str  # "path" | "title" | "llm" | "fallback"
    llm_raw: str | None = None


async def fetch_ml_sources(pool) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch("""
            SELECT id, title, file_path::text AS file_path
            FROM sources WHERE domain_id = 'machine_learning'
            ORDER BY title
        """)
    return [dict(r) for r in rows]


def classify_one(row: dict, use_llm: bool) -> Suggestion:
    path = classify_by_path(row.get("file_path"))
    if path:
        return Suggestion(
            str(row["id"]),
            row["title"],
            row.get("file_path"),
            "machine_learning",
            path,
            tier_of(path),
            "path",
        )
    title = classify_by_title(row["title"])
    if title:
        return Suggestion(
            str(row["id"]),
            row["title"],
            row.get("file_path"),
            "machine_learning",
            title,
            tier_of(title),
            "title",
        )
    if use_llm:
        dom, raw = llm_classify(row["title"], row.get("file_path") or "")
        return Suggestion(
            str(row["id"]),
            row["title"],
            row.get("file_path"),
            "machine_learning",
            dom,
            tier_of(dom),
            "llm" if dom != "general" else "fallback",
            llm_raw=raw,
        )
    return Suggestion(
        str(row["id"]),
        row["title"],
        row.get("file_path"),
        "machine_learning",
        "general",
        tier_of("general"),
        "fallback",
    )


async def apply_retag(pool, suggestions: list[Suggestion]) -> dict:
    """Apply retag to DB. Backs up old domain_id to metadata.legacy_domain_id.

    Translates suggested_domain → DB-canonical via DOMAIN_TO_DB before write.
    Also stores the original (audit-canonical) suggestion in metadata.audit_domain_suggested.
    """
    async with pool.acquire() as conn:
        await conn.set_type_codec(
            "jsonb", encoder=json.dumps, decoder=json.loads, schema="pg_catalog"
        )
        applied = 0
        skipped = 0
        for s in suggestions:
            db_target = to_db_domain(s.suggested_domain)
            if db_target == s.current_domain:
                skipped += 1
                continue
            await conn.execute(
                """UPDATE sources SET
                       domain_id = $1,
                       metadata = jsonb_set(
                           jsonb_set(
                               jsonb_set(COALESCE(metadata, '{}'::jsonb),
                                         '{legacy_domain_id}', to_jsonb($2::text), true),
                               '{retag_method}', to_jsonb($3::text), true),
                           '{audit_domain_suggested}', to_jsonb($5::text), true)
                   WHERE id = $4""",
                db_target,
                s.current_domain,
                s.method,
                s.source_id,
                s.suggested_domain,
            )
            applied += 1
        return {"applied": applied, "skipped": skipped}


async def main():
    parser = argparse.ArgumentParser(description="Retag ML-misclassified sources.")
    parser.add_argument("--apply", action="store_true", help="Write to DB (default: dry-run only)")
    parser.add_argument(
        "--skip-llm", action="store_true", help="Skip LLM, use fallback for unmatched"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/tmp/retag_plan.json"),
        help="Where to write the retag plan JSON (default: /tmp/retag_plan.json)",
    )
    args = parser.parse_args()

    pool = await get_connection_pool(DatabaseConfig())
    rows = await fetch_ml_sources(pool)
    print(f"Total ML-tagged sources: {len(rows)}")

    suggestions: list[Suggestion] = []
    use_llm = not args.skip_llm
    if use_llm:
        # Stage 1: path + title (cheap, deterministic)
        for r in rows:
            path = classify_by_path(r.get("file_path"))
            if path:
                suggestions.append(
                    Suggestion(
                        str(r["id"]),
                        r["title"],
                        r.get("file_path"),
                        "machine_learning",
                        path,
                        tier_of(path),
                        "path",
                    )
                )
                continue
            title = classify_by_title(r["title"])
            if title:
                suggestions.append(
                    Suggestion(
                        str(r["id"]),
                        r["title"],
                        r.get("file_path"),
                        "machine_learning",
                        title,
                        tier_of(title),
                        "title",
                    )
                )
                continue
            suggestions.append(
                Suggestion(
                    str(r["id"]),
                    r["title"],
                    r.get("file_path"),
                    "machine_learning",
                    "__pending_llm__",
                    0,
                    "pending_llm",
                )
            )
        # Stage 2: LLM for pending
        pending = [s for s in suggestions if s.method == "pending_llm"]
        print(f"\nLLM-classifying {len(pending)} sources via {OLLAMA_MODEL}...")
        for i, s in enumerate(pending, 1):
            dom, raw = llm_classify(s.title, s.file_path or "")
            s.suggested_domain = dom
            s.suggested_tier = tier_of(dom)
            s.method = "llm" if dom != "general" else "fallback"
            s.llm_raw = raw
            if i % 10 == 0 or i == len(pending):
                print(f"  [{i}/{len(pending)}] last → {dom} (raw: {raw[:60]})")
    else:
        for r in rows:
            suggestions.append(classify_one(r, use_llm=False))

    # Distributions
    method_dist = Counter(s.method for s in suggestions)
    domain_dist = Counter(s.suggested_domain for s in suggestions)
    tier_dist = Counter(s.suggested_tier for s in suggestions)
    print(f"\nMethod distribution:")
    for m, n in method_dist.most_common():
        print(f"  {m:12s}: {n}")
    print(f"\nSuggested domain distribution:")
    for d, n in domain_dist.most_common():
        print(f"  {d:25s} (Tier {tier_of(d)}): {n}")
    print(f"\nSuggested tier distribution:")
    for t in sorted(tier_dist):
        print(f"  Tier {t}: {tier_dist[t]}")

    # Save plan
    args.out.write_text(
        json.dumps(
            {
                "total": len(rows),
                "method_dist": dict(method_dist),
                "domain_dist": dict(domain_dist),
                "tier_dist": {str(k): v for k, v in tier_dist.items()},
                "suggestions": [vars(s) for s in suggestions],
            },
            indent=2,
        )
    )
    print(f"\nRetag plan: {args.out}")

    if args.apply:
        print("\nApplying retag to DB...")
        result = await apply_retag(pool, suggestions)
        print(f"  Applied: {result['applied']}")
        print(f"  Skipped (no change): {result['skipped']}")
    else:
        print("\nDry-run only. Re-run with --apply to write to DB.")


if __name__ == "__main__":
    asyncio.run(main())
