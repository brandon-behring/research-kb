"""Tests for the curated programming ingest manifest builder."""

from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path

import pytest

from catalog_library import build_programming_ingest_manifest as builder


def write_file(path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def write_catalog(path: Path, entries: list[dict]) -> None:
    path.write_text(json.dumps(entries, indent=2) + "\n")


def base_entry(
    *,
    title: str,
    filename: str,
    domain: str,
    priority_score: int,
    full_path: str | None = None,
    year: int | None = None,
) -> dict:
    return {
        "title": title,
        "filename": filename,
        "domain": domain,
        "priority_score": priority_score,
        "full_path": full_path or f"/external/{filename}",
        "file_size_mb": 1.0,
        "is_book": True,
        "authors": ["Author Name"],
        "year": year,
        "publisher": "Test Publisher",
        "r2_isbns": [],
    }


def build_outputs(
    tmp_path: Path,
    *,
    catalog_entries: list[dict],
    download_files: list[str],
    library_files: list[str] | None = None,
) -> tuple[list[dict], list[dict], Path]:
    catalog_path = tmp_path / "catalog.json"
    downloads_dir = tmp_path / "downloads"
    library_root = tmp_path / "library_books"

    downloads_dir.mkdir(parents=True, exist_ok=True)
    library_root.mkdir(parents=True, exist_ok=True)
    write_catalog(catalog_path, catalog_entries)

    for filename in library_files or []:
        write_file(library_root / filename)

    for filename in download_files:
        write_file(downloads_dir / filename)

    manifest, review_rows = builder.build_programming_manifest(
        catalog_path=catalog_path,
        downloads_dir=downloads_dir,
        library_root=library_root,
    )
    return manifest, review_rows, library_root


@pytest.mark.scripts
def test_builder_deduplicates_fluent_effective_and_refactoring(tmp_path):
    manifest, review_rows, _library_root = build_outputs(
        tmp_path,
        catalog_entries=[
            base_entry(
                title="Fluent Python",
                filename="Luciano Ramalho - Fluent Python_ Clear Concise and Effective Programming-OReilly Media 2015.pdf",
                domain="software_engineering",
                priority_score=50,
                year=2015,
            )
        ],
        library_files=[
            "Luciano Ramalho - Fluent Python_ Clear Concise and Effective Programming-OReilly Media 2015.pdf"
        ],
        download_files=[
            "Fluent Python_ Clear, Concise, and Effective Programming, -- Luciano Ramalho -- 2nd, 2022 -- Beijing _ O'Reilly Media, Inc -- 9781492056355 -- 6b8f1e751c6d6b82a49cc155099f9949 -- Anna’s Archive.pdf",
            "Effective Python_ 125 Specific Ways to Write Better Python, -- Brett Slatkin -- 2024 -- Addison-Wesley -- 9780138172398 -- ee704457f028e6afc9bff04e2c5e6786 -- Anna’s Archive.pdf",
            "Effective Python_ 125 Specific Ways to Write Better Python_ -- Brett Slatkin -- 3, 2024 nov 30 -- Pearson Education, Limited -- 9780138172183 -- 859e2b0eab74efb9626b6b79a353f72a -- Anna’s Archive.pdf",
            "Refactoring_ Improving the Design of Existing Code -- Martin Fowler, Kent Beck -- Addison-Wesley Signature Series (Fowler) Ser, 2nd ed, -- 9780134757599 -- f4fc75eb1eee5537400908e6f59db631 -- Anna’s Archive.pdf",
            "Refactoring_ Improving the Design of Existing Code -- Martin Fowler, Kent Beck -- Addison-Wesley Signature Series (Fowler) Ser, 2nd ed, -- 9780134757599 -- f4fc75eb1eee5537400908e6f59db631 -- Anna’s Archive (1).pdf",
        ],
    )

    manifest_titles = {entry["title"] for entry in manifest}
    assert any("Fluent Python" in title for title in manifest_titles)
    assert sum("Effective Python" in entry["title"] for entry in manifest) == 1
    assert sum("Refactoring" in entry["title"] for entry in manifest) == 1

    fluent_entry = next(entry for entry in manifest if "Fluent Python" in entry["title"])
    assert "2022" in fluent_entry["full_path"]
    assert fluent_entry["source_kind"] == "downloads"
    assert fluent_entry["catalog_domain"] == "software_engineering"
    assert fluent_entry["catalog_priority"] == 50

    refactoring_entry = next(entry for entry in manifest if "Refactoring" in entry["title"])
    assert "(1)" not in refactoring_entry["filename"]

    duplicate_rows = [row for row in review_rows if row["status"] == "duplicate"]
    assert any("Refactoring" in row["title"] for row in duplicate_rows)
    assert any("Fluent Python" in row["duplicate_of"] for row in duplicate_rows)


@pytest.mark.scripts
def test_builder_filters_meap_cheatsheets_interview_and_cert_guides(tmp_path):
    manifest, review_rows, _library_root = build_outputs(
        tmp_path,
        catalog_entries=[
            base_entry(
                title="Python Concurrency with asyncio",
                filename="Python_Concurrency_with_asyncio_v10_MEAP.pdf",
                domain="software_engineering",
                priority_score=40,
            ),
            base_entry(
                title="Elements of Programming Interviews in Python",
                filename="Elements of Programming Interviews in Python.pdf",
                domain="algorithms",
                priority_score=50,
            ),
            base_entry(
                title="AWS Certified Data Engineering Study Guide",
                filename="AWS Certified Data Engineering Study Guide.pdf",
                domain="ml_engineering",
                priority_score=50,
            ),
            base_entry(
                title="Deep Learning with Python",
                filename="Deep_Learning_with_Python.pdf",
                domain="deep_learning",
                priority_score=70,
            ),
        ],
        download_files=[
            "SQL cookbook _ query solutions and techniques for all SQL -- Anthony Molinaro -- 2020 -- O'Reilly -- 9781492077442 -- abcdefabcdefabcdefabcdefabcdefab -- Anna’s Archive.pdf",
            "sql-cheat-sheet.pdf",
        ],
    )

    manifest_titles = {entry["title"] for entry in manifest}
    assert any("SQL cookbook" in title or "SQL cookbook" in title for title in manifest_titles)
    assert all("MEAP" not in entry["filename"] for entry in manifest)
    assert all("Interview" not in entry["title"] for entry in manifest)
    assert all("Certified" not in entry["title"] for entry in manifest)
    assert all("Deep Learning with Python" not in entry["title"] for entry in manifest)

    status_by_title = {row["title"]: row["status"] for row in review_rows}
    assert status_by_title["Python Concurrency with asyncio"] == "filtered_early_access"
    assert status_by_title["Elements of Programming Interviews in Python"] == "filtered_excluded"
    assert status_by_title["AWS Certified Data Engineering Study Guide"] == "filtered_excluded"
    assert status_by_title["sql-cheat-sheet"] == "filtered_excluded"
    assert "Deep Learning with Python" not in status_by_title


@pytest.mark.scripts
def test_builder_defers_epub_and_keeps_it_out_of_manifest(tmp_path):
    manifest, review_rows, _library_root = build_outputs(
        tmp_path,
        catalog_entries=[],
        download_files=[
            "Designing Data-Intensive Applications_ The Big Ideas Behind -- Martin Kleppmann, Chris Riccomini -- 2nd, 2026 -- O'Reilly Media, Incorporated -- 9781098119003 -- ddbefdc2066789df0b8ac0d23da945ea -- Anna’s Archive.epub"
        ],
    )

    assert manifest == []
    deferred = [row for row in review_rows if row["status"] == "deferred_non_pdf"]
    assert len(deferred) == 1
    assert "Designing Data-Intensive Applications" in deferred[0]["title"]


def import_mass_ingest_catalog_with_stubs(monkeypatch) -> types.ModuleType:
    sys.modules.pop("mass_ingest_catalog", None)

    stub_ingest = types.ModuleType("ingest_missing_textbooks")
    stub_ingest.compute_file_hash = lambda _path: "hash"

    async def ingest_textbook(*_args, **_kwargs):
        return ("source-id", 0, 0)

    stub_ingest.ingest_textbook = ingest_textbook
    monkeypatch.setitem(sys.modules, "ingest_missing_textbooks", stub_ingest)

    stub_common = types.ModuleType("research_kb_common")

    class DummyError(Exception):
        pass

    class DummyLogger:
        def info(self, *_args, **_kwargs):
            return None

        def warning(self, *_args, **_kwargs):
            return None

        def error(self, *_args, **_kwargs):
            return None

    stub_common.EmbeddingError = DummyError
    stub_common.StorageError = DummyError
    stub_common.configure_logging = lambda **_kwargs: None
    stub_common.get_logger = lambda _name: DummyLogger()
    monkeypatch.setitem(sys.modules, "research_kb_common", stub_common)

    stub_storage = types.ModuleType("research_kb_storage")
    stub_storage.DatabaseConfig = object
    stub_storage.SourceStore = object
    stub_storage.get_connection_pool = lambda *_args, **_kwargs: None
    monkeypatch.setitem(sys.modules, "research_kb_storage", stub_storage)

    return importlib.import_module("mass_ingest_catalog")


@pytest.mark.scripts
def test_generated_manifest_is_readable_by_mass_ingest_dry_run(tmp_path, monkeypatch, capsys):
    manifest, review_rows, library_root = build_outputs(
        tmp_path,
        catalog_entries=[
            base_entry(
                title="Unit Testing Principles, Practices, and Patterns",
                filename="Vladimir Khorikov - Unit Testing Principles, Practices, and Patterns-Manning Publications (2019).pdf",
                domain="software_engineering",
                priority_score=65,
            )
        ],
        library_files=[
            "Vladimir Khorikov - Unit Testing Principles, Practices, and Patterns-Manning Publications (2019).pdf"
        ],
        download_files=[],
    )
    manifest_path = tmp_path / "programming_ingest_manifest.json"
    review_path = tmp_path / "programming_ingest_review.csv"
    builder.write_json(manifest_path, manifest)
    builder.write_csv(review_path, review_rows)

    mass_ingest_catalog = import_mass_ingest_catalog_with_stubs(monkeypatch)
    entries = mass_ingest_catalog.load_catalog(manifest_path)
    mass_ingest_catalog.print_dry_run(entries, library_root)
    output = capsys.readouterr().out

    assert len(entries) == 1
    assert "Would ingest 1 books" in output
    assert "software_engineering: 1" in output
