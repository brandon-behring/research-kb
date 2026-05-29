#!/usr/bin/env python3
"""Phase 2 — ingest the research_cache into research-kb.

Two paths:
  * arXiv / true-PDF  -> PDFDispatcher.ingest_pdf (Docling+GROBID). Uses
    skip_embedding=True to avoid Docling/embed-daemon VRAM contention on the
    8GB GPU; embeddings backfilled separately (scripts/embed_missing.py).
  * web / HTML        -> trafilatura clean -> chunk_text -> SourceStore.create
    + ChunkStore.batch_create (embeds inline; only the daemon touches the GPU).

Idempotent on file_hash (cache blob sha256). Tags source_class + a 3-date /
half-life temporal model. Registers the agents / ml_security domains.
NO DB writes under --dry-run. D2 settled: admit frontier web tagged.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import hashlib
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

RKB = Path("/home/brandon_behring/Claude/research-kb")
for _pkg in ("pdf-tools", "storage", "contracts", "common"):
    sys.path.insert(0, str(RKB / "packages" / _pkg / "src"))

import trafilatura  # noqa: E402
from research_kb_contracts import SourceType  # noqa: E402
from research_kb_pdf import run_post_ingest_hooks  # noqa: E402
from research_kb_pdf.chunker import count_tokens, split_paragraphs, split_sentences  # noqa: E402
from research_kb_pdf.embedding_client import EmbeddingClient  # noqa: E402
from research_kb_pdf.grobid_client import GrobidClient, parse_tei_xml  # noqa: E402
from research_kb_storage import (  # noqa: E402
    ChunkStore,
    CitationStore,
    DatabaseConfig,
    DomainStore,
    SourceStore,
    close_connection_pool,
    get_connection_pool,
)

import json  # noqa: E402

# arxiv id -> fetched-PDF sha256 (the source file_hash), from the Phase 1a batch
ARXIV_SHA: dict[str, str] = {}
for _ln in (
    open("/var/tmp/arxiv_pdf_staging/results.jsonl", errors="replace")
    if Path("/var/tmp/arxiv_pdf_staging/results.jsonl").exists()
    else []
):
    try:
        _r = json.loads(_ln)
        if _r.get("status") == "ok" and _r.get("sha256"):
            ARXIV_SHA[_r["id"]] = _r["sha256"]
    except Exception:
        pass

CACHE = Path("/home/brandon_behring/Claude/research_cache")
ARXIV_PDF = Path("/var/tmp/arxiv_pdf_staging/pdf")
ARXIV_TEI = Path("/var/tmp/arxiv_pdf_staging/tei")

# --- new domains to register (idempotent) ---
NEW_DOMAINS = {
    "agents": (
        "Agents & Harness Engineering",
        "LLM agent design, harnesses, MCP, tool use, context engineering",
    ),
    "ml_security": (
        "ML Security & Prompt Injection",
        "Prompt injection, detectors, agentic security, red-teaming, LLM risks",
    ),
}


def domain_for(path: str) -> str:
    p = path.lower()
    if "research_agent_" in p or "agentic" in p:
        return "agents"
    if any(
        k in p
        for k in (
            "prompt-injection",
            "detector",
            "pi_bench",
            "rag-injection",
            "direct-vs-indirect",
            "training-and-eval",
        )
    ):
        return "ml_security"
    if any(k in p for k in ("julia_", "dynamical_systems", "vortex", "epidemic")):
        return "mathematics"
    if any(
        k in p
        for k in (
            "causal",
            "incrementality",
            "pricing",
            "offline_rl",
            "experimentation",
            "recommendation",
        )
    ):
        return "causal_inference"
    if any(k in p for k in ("rlhf", "peft", "transformer", "pretraining", "post_training")):
        return "deep_learning"
    if any(
        k in p
        for k in (
            "calibration",
            "eval_methodology",
            "eval_drift",
            "optimization_theory",
            "statistical_inference",
            "knowledge_graphs",
        )
    ):
        return "machine_learning"
    return "machine_learning"  # safe default


def classify(url: str) -> str:
    u = url.lower()
    if "example.com" in u or "example.invalid" in u:
        return "noise"
    if "api.crossref.org" in u:
        return "crossref_api"
    if "arxiv.org/abs" in u or "arxiv.org/pdf" in u:
        return "arxiv"
    if u.endswith(".pdf"):
        return "pdf"
    if any(
        d in u
        for d in (
            "doi.org",
            "springer",
            "acm.org",
            "ieee",
            "wiley",
            "jstor",
            "oup",
            "tandfonline",
            "mlr.press",
            "aeaweb",
            "ametsoc",
            "ncbi",
            "sciencedirect",
            "nature",
            "cambridge",
            "aclanthology",
            "openreview",
            "semanticscholar",
            "hal.science",
            "princeton.edu",
            "routledge",
            "eric.ed.gov",
            "berkeley.edu",
        )
    ):
        return "journal"
    if any(
        d in u
        for d in (
            "anthropic.com",
            "claude.com",
            "modelcontextprotocol",
            "owasp.org",
            "lakera.ai",
            "nist.gov",
            "mitre",
        )
    ):
        return "web_vendor"
    if any(d in u for d in ("github.com", "huggingface.co", "gitlab")):
        return "web_code"
    return "web_practitioner"


# ---------- helper 1: clean_html (trafilatura) ----------
def clean_html(raw: str) -> str:
    """Return clean prose, or '' if unrecoverable (<80 words)."""
    txt = (
        trafilatura.extract(raw, include_comments=False, include_tables=True, favor_recall=True)
        or ""
    )
    return txt if len(txt.split()) >= 80 else ""


def web_title(raw: str, url: str) -> str:
    m = re.search(r"<title[^>]*>([^<]{3,150})</title>", raw, re.I)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    return urlparse(url).netloc + urlparse(url).path[:40]


def web_authored_date(raw: str) -> str | None:
    try:
        md = trafilatura.metadata.extract_metadata(raw)
        return getattr(md, "date", None) if md else None
    except Exception:
        return None


# ---------- helper 2: chunk_text (raw string -> ~300-token chunks) ----------
def chunk_text(text: str, max_tokens: int = 300) -> list[str]:
    chunks: list[str] = []
    cur: list[str] = []
    cur_tok = 0

    def flush():
        nonlocal cur, cur_tok
        if cur:
            chunks.append("\n\n".join(cur))
            cur, cur_tok = [], 0

    for para in split_paragraphs(text):
        ptok = count_tokens(para)
        if ptok > max_tokens:
            flush()
            sent, stok = [], 0
            for s in split_sentences(para):
                stk = count_tokens(s)
                if stok + stk > max_tokens and sent:
                    chunks.append(" ".join(sent))
                    sent, stok = [], 0
                sent.append(s)
                stok += stk
            if sent:
                chunks.append(" ".join(sent))
        elif cur_tok + ptok > max_tokens:
            flush()
            cur, cur_tok = [para], ptok
        else:
            cur.append(para)
            cur_tok += ptok
    flush()
    return [c for c in chunks if c.strip()]


# ---------- work-list ----------
def build_worklist() -> dict[str, dict]:
    """sha256 -> {url, fetched_at, domain, tier}. Deduped across all manifests."""
    seen: dict[str, dict] = {}
    for mf in glob.glob(str(Path.home() / "Claude" / "**" / "cache_manifest.yml"), recursive=True):
        dom = domain_for(mf)
        cur: dict = {}
        for ln in open(mf, errors="replace"):
            m = re.match(r"\s*-?\s*(source_url|sha256|fetched_at):\s*(\S+)", ln)
            if not m:
                continue
            k, v = m.groups()
            if k == "source_url" and cur.get("source_url"):
                if cur.get("sha256") and cur["sha256"] not in seen:
                    seen[cur["sha256"]] = {
                        "url": cur["source_url"],
                        "fetched_at": cur.get("fetched_at"),
                        "domain": dom,
                        "tier": classify(cur["source_url"]),
                    }
                cur = {}
            cur[k] = v
        if cur.get("sha256") and cur.get("source_url") and cur["sha256"] not in seen:
            seen[cur["sha256"]] = {
                "url": cur["source_url"],
                "fetched_at": cur.get("fetched_at"),
                "domain": dom,
                "tier": classify(cur["source_url"]),
            }
    return seen


# ---------- web ingest (mirrors dispatcher orchestration) ----------
async def ingest_web(
    sha: str, info: dict, embed: EmbeddingClient, skip_embedding: bool, dry: bool
) -> str:
    if await SourceStore.get_by_file_hash(sha):
        return "skip:exists"
    blob = CACHE / "blobs" / "sha256" / sha
    if not blob.exists():
        return "skip:no-blob"
    raw = blob.read_bytes().decode("utf-8", errors="replace")
    text = clean_html(raw)
    if not text:
        return "skip:unrecoverable(<80w)"
    pieces = chunk_text(text)
    if dry:
        return f"would-ingest: {len(pieces)} chunks, {len(text.split())}w"

    sub = info["tier"]
    stale = 365 if sub == "web_vendor" else 180
    src = await SourceStore.create(
        source_type=SourceType.WEB,
        title=web_title(raw, info["url"]),
        file_hash=sha,
        domain_id=info["domain"],
        metadata={
            "url": info["url"],
            "source_class": sub,
            "retrieved_at": info.get("fetched_at"),
            "authored_date": web_authored_date(raw),
            "stale_after_days": stale,
            "freshness_tier": "web",
            "ingest": "cache_web_2026-05-28",
        },
    )
    data = []
    for i, content in enumerate(pieces):
        content = content.replace("\x00", "").replace("�", "")
        data.append(
            {
                "source_id": src.id,
                "content": content,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "page_start": None,
                "page_end": None,
                "embedding": None if skip_embedding else embed.embed(content),
                "domain_id": info["domain"],  # else batch_create defaults to causal_inference
                "metadata": {"chunk_index": i, "chunking_method": "text_splitter"},
            }
        )
    await ChunkStore.batch_create(data)
    return f"ok: {len(data)} chunks"


# ---------- arxiv ingest (TEI-based: CPU parse of staged GROBID TEI + daemon embed) ----------
# Docling/dispatcher path OOMs on the 8GB GPU (embed daemon resident); the 691
# GROBID TEIs already exist from Phase 1a, so parse those instead — no Docling/OCR.
async def ingest_arxiv(
    sha: str, info: dict, embed: EmbeddingClient, skip_embedding: bool, dry: bool
) -> tuple[str, object]:
    m = re.search(r"arxiv\.org/abs/(\S+)", info["url"])
    aid = re.sub(r"v\d+$", "", m.group(1)) if m else None
    if not aid:
        return "skip:no-id", None
    pdf_sha = ARXIV_SHA.get(aid)
    if not pdf_sha:
        return "skip:not-fetched", None  # fetch failed / not in batch
    if await SourceStore.get_by_file_hash(pdf_sha):
        return "skip:exists", None  # already in KB (e.g. Mamba) — idempotent
    tei_f = ARXIV_TEI / f"{aid.replace('/', '_')}.tei.xml"
    if not tei_f.exists():
        return "skip:no-tei", None
    paper = parse_tei_xml(tei_f.read_text(errors="replace"))
    year = paper.metadata.year or (2000 + int(aid[:2]) if re.match(r"\d{4}\.", aid) else None)
    pieces = chunk_text(paper.raw_text or "")
    if not pieces:
        return "skip:no-text", None
    if dry:
        return (
            f"would: {(paper.metadata.title or aid)[:40]} {len(pieces)}ch {len(paper.citations)}cit",
            None,
        )

    src = await SourceStore.create(
        source_type=SourceType.PAPER,
        title=(paper.metadata.title or f"arXiv:{aid}").strip()[:480],
        file_hash=pdf_sha,
        domain_id=info["domain"],
        authors=paper.metadata.authors or [],
        year=year,
        metadata={
            "arxiv_id": aid,
            "source_class": "paper",
            "authored_date": f"20{aid[:2]}-{aid[2:4]}" if re.match(r"\d{4}\.", aid) else None,
            "retrieved_at": info.get("fetched_at"),
            "stale_after_days": 1825,
            "freshness_tier": "paper",
            "extraction_method": "grobid_tei",
            "ingest": "cache_arxiv_2026-05-28",
        },
    )
    data = []
    for i, content in enumerate(pieces):
        content = content.replace("\x00", "").replace("�", "")
        data.append(
            {
                "source_id": src.id,
                "content": content,
                "content_hash": hashlib.sha256(content.encode()).hexdigest(),
                "page_start": None,
                "page_end": None,
                "embedding": None if skip_embedding else embed.embed(content),
                "domain_id": info["domain"],
                "metadata": {"chunk_index": i, "chunking_method": "tei_text_splitter"},
            }
        )
    await ChunkStore.batch_create(data)
    if paper.citations:
        cdata = [
            {
                "source_id": src.id,
                "authors": c.authors,
                "title": c.title,
                "year": c.year,
                "venue": c.venue,
                "doi": c.doi,
                "arxiv_id": c.arxiv_id,
                "raw_string": c.raw_string,
                "bibtex": None,
                "extraction_method": "grobid",
                "confidence_score": None,
                "metadata": {},
            }
            for c in paper.citations
        ]
        await CitationStore.batch_create(cdata)
    return f"ok: {len(data)}ch {len(paper.citations)}cit", src.id


# ---------- journal ingest (HTML landing/full-text -> PAPER if recoverable) ----------
async def ingest_journal(
    sha: str, info: dict, embed: EmbeddingClient, skip_embedding: bool, dry: bool
) -> str:
    if await SourceStore.get_by_file_hash(sha):
        return "skip:exists"
    blob = CACHE / "blobs" / "sha256" / sha
    if not blob.exists():
        return "skip:no-blob"
    raw = blob.read_bytes().decode("utf-8", errors="replace")
    text = clean_html(raw)
    if not text:
        return "skip:paywall/landing(<80w)"
    pieces = chunk_text(text)
    if dry:
        return f"would: {len(pieces)}ch, {len(text.split())}w"
    src = await SourceStore.create(
        source_type=SourceType.PAPER,
        title=web_title(raw, info["url"]),
        file_hash=sha,
        domain_id=info["domain"],
        metadata={
            "url": info["url"],
            "source_class": "journal",
            "authored_date": web_authored_date(raw),
            "retrieved_at": info.get("fetched_at"),
            "stale_after_days": 1825,
            "freshness_tier": "paper",
            "ingest": "cache_journal_2026-05-28",
        },
    )
    data = [
        {
            "source_id": src.id,
            "content": (c := p.replace("\x00", "").replace("�", "")),
            "content_hash": hashlib.sha256(c.encode()).hexdigest(),
            "page_start": None,
            "page_end": None,
            "embedding": None if skip_embedding else embed.embed(c),
            "domain_id": info["domain"],
            "metadata": {"chunk_index": i, "chunking_method": "text_splitter"},
        }
        for i, p in enumerate(pieces)
    ]
    await ChunkStore.batch_create(data)
    return f"ok: {len(data)}ch"


# ---------- pdf-blob ingest (true cache PDFs -> GROBID on the fly -> PAPER) ----------
async def ingest_pdf_blob(
    sha: str,
    info: dict,
    grobid: GrobidClient,
    embed: EmbeddingClient,
    skip_embedding: bool,
    dry: bool,
) -> tuple[str, object]:
    if await SourceStore.get_by_file_hash(sha):
        return "skip:exists", None
    blob = CACHE / "blobs" / "sha256" / sha
    if not blob.exists():
        return "skip:no-blob", None
    if dry:
        return "would: GROBID+ingest", None
    try:
        paper = grobid.process_pdf(str(blob))
    except Exception as e:
        return f"skip:grobid-fail({type(e).__name__})", None
    pieces = chunk_text(paper.raw_text or "")
    if not pieces:
        return "skip:no-text", None
    src = await SourceStore.create(
        source_type=SourceType.PAPER,
        title=(paper.metadata.title or f"cache-pdf:{sha[:12]}").strip()[:480],
        file_hash=sha,
        domain_id=info["domain"],
        authors=paper.metadata.authors or [],
        year=paper.metadata.year,
        metadata={
            "url": info["url"],
            "source_class": "pdf",
            "retrieved_at": info.get("fetched_at"),
            "stale_after_days": 1825,
            "freshness_tier": "paper",
            "extraction_method": "grobid",
            "ingest": "cache_pdf_2026-05-28",
        },
    )
    data = [
        {
            "source_id": src.id,
            "content": (c := p.replace("\x00", "").replace("�", "")),
            "content_hash": hashlib.sha256(c.encode()).hexdigest(),
            "page_start": None,
            "page_end": None,
            "embedding": None if skip_embedding else embed.embed(c),
            "domain_id": info["domain"],
            "metadata": {"chunk_index": i, "chunking_method": "grobid_text_splitter"},
        }
        for i, p in enumerate(pieces)
    ]
    await ChunkStore.batch_create(data)
    if paper.citations:
        await CitationStore.batch_create(
            [
                {
                    "source_id": src.id,
                    "authors": c.authors,
                    "title": c.title,
                    "year": c.year,
                    "venue": c.venue,
                    "doi": c.doi,
                    "arxiv_id": c.arxiv_id,
                    "raw_string": c.raw_string,
                    "bibtex": None,
                    "extraction_method": "grobid",
                    "confidence_score": None,
                    "metadata": {},
                }
                for c in paper.citations
            ]
        )
    return f"ok: {len(data)}ch {len(paper.citations)}cit", src.id


async def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest research_cache into research-kb")
    ap.add_argument(
        "--source", choices=["web", "arxiv", "journal", "pdf", "rest", "all"], default="web"
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--skip-embedding", action="store_true", help="web path only")
    ap.add_argument("--build-citations", action="store_true", help="arxiv: build graph after")
    args = ap.parse_args()

    await get_connection_pool(DatabaseConfig())
    try:
        if not args.dry_run:
            for did, (name, desc) in NEW_DOMAINS.items():
                try:
                    if not await DomainStore.get_by_id(did):
                        await DomainStore.create(domain_id=did, name=name, description=desc)
                        print(f"[domain] registered {did}")
                except Exception as e:
                    print(f"[domain] {did}: {e}")

        work = build_worklist()
        tiers = {}
        for v in work.values():
            tiers[v["tier"]] = tiers.get(v["tier"], 0) + 1
        print(f"work-list: {len(work)} unique sources | tiers: {tiers}")

        web_tiers = {"web_vendor", "web_practitioner", "web_code"}
        want = {
            "web": web_tiers,
            "arxiv": {"arxiv"},
            "journal": {"journal"},
            "pdf": {"pdf"},
            "rest": {"journal", "pdf"},
            "all": {"arxiv", "journal", "pdf"} | web_tiers,
        }[args.source]
        embed = EmbeddingClient()
        grobid = GrobidClient() if want & {"pdf"} else None
        counts: dict[str, int] = {}
        src_ids = []

        items = list(work.items())
        n = 0
        for sha, info in items:
            t = info["tier"]
            if t not in want:
                continue
            if args.limit and n >= args.limit:
                break
            n += 1
            try:
                if t in web_tiers:
                    r = await ingest_web(sha, info, embed, args.skip_embedding, args.dry_run)
                elif t == "arxiv":
                    r, sid = await ingest_arxiv(sha, info, embed, args.skip_embedding, args.dry_run)
                    if sid:
                        src_ids.append(sid)
                elif t == "journal":
                    r = await ingest_journal(sha, info, embed, args.skip_embedding, args.dry_run)
                elif t == "pdf":
                    r, sid = await ingest_pdf_blob(
                        sha, info, grobid, embed, args.skip_embedding, args.dry_run
                    )
                    if sid:
                        src_ids.append(sid)
                else:
                    r = f"skip:tier={t}"
            except Exception as e:
                r = f"ERROR: {type(e).__name__}: {str(e)[:80]}"
            key = r.split(":")[0].split(" ")[0]
            counts[key] = counts.get(key, 0) + 1
            if n <= 25 or n % 50 == 0:
                print(f"  [{n}] {info['tier']:16} {info['url'][:46]:46} -> {r}")

        print(f"\nsummary ({args.source}): {counts}")

        # Reconcile chunk domains to their source domain (the dispatcher/arxiv path
        # leaves chunks at the causal_inference default; idempotent safety net).
        if not args.dry_run:
            pool = await get_connection_pool(DatabaseConfig())
            async with pool.acquire() as conn:
                res = await conn.execute(
                    "UPDATE chunks SET domain_id = s.domain_id FROM sources s "
                    "WHERE chunks.source_id = s.id AND chunks.domain_id <> s.domain_id "
                    "AND s.metadata->>'ingest' LIKE 'cache_%'"
                )
            print(f"reconciled chunk domains: {res}")

        if src_ids and args.build_citations and not args.dry_run:
            print(f"building citation graph for {len(src_ids)} sources...")
            s = await run_post_ingest_hooks(src_ids)
            print(f"  citations: {s.get('citations', {})}")
    finally:
        await close_connection_pool()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
