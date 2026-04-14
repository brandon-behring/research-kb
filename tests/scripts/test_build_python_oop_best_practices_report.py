"""Tests for the Python OOP best-practices report builder."""

from __future__ import annotations

from pathlib import Path

import pytest

from catalog_library import build_python_oop_best_practices_report as builder


def write_file(path: Path, size: int = 1024) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)


def write_note_tree(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("# Notes\n")


def build_report(
    tmp_path: Path,
    *,
    library_files: list[str] | None = None,
    acquisition_files: list[str] | None = None,
    top_download_files: list[str] | None = None,
    manning_files: list[str] | None = None,
    note_dirs: list[str] | None = None,
) -> tuple[str, list[dict[str, str]]]:
    library_root = tmp_path / "library_books"
    downloads_root = tmp_path / "Downloads"
    acquisitions_root = downloads_root / "acqusition_books" / "acqusition_books"
    manning_root = tmp_path / "Manning_Books_research_kb"
    course_learning_root = tmp_path / "course_learning"

    library_root.mkdir(parents=True, exist_ok=True)
    acquisitions_root.mkdir(parents=True, exist_ok=True)
    manning_root.mkdir(parents=True, exist_ok=True)
    course_learning_root.mkdir(parents=True, exist_ok=True)

    for filename in library_files or []:
        write_file(library_root / filename)
    for filename in acquisition_files or []:
        write_file(acquisitions_root / filename)
    for filename in top_download_files or []:
        write_file(downloads_root / filename)
    for filename in manning_files or []:
        write_file(manning_root / filename)
    for dirname in note_dirs or []:
        write_note_tree(course_learning_root / dirname)

    return builder.build_python_oop_report(
        library_root=library_root,
        downloads_root=downloads_root,
        manning_root=manning_root,
        course_learning_root=course_learning_root,
    )


@pytest.mark.scripts
def test_primary_report_contains_expected_books_and_note_trees(tmp_path):
    markdown, rows = build_report(
        tmp_path,
        library_files=[
            "JJ Geewax - API Design Patterns-Manning Publications 2021.pdf",
            "Vladimir Khorikov - Unit Testing Principles, Practices, and Patterns-Manning Publications (2019).pdf",
            "David Thomas, Andrew Hunt - The Pragmatic Programmer_ Your Journey to Mastery-Addison-Wesley Professional (2019).pdf",
        ],
        acquisition_files=[
            "Fluent Python_ Clear, Concise, and Effective Programming, -- Luciano Ramalho -- 2nd, 2022 -- OReilly -- 9781492056355.pdf",
            "Effective Python_ 125 Specific Ways to Write Better Python, -- Brett Slatkin -- 2024 -- Addison-Wesley -- 9780138172398.pdf",
            "Refactoring_ Improving the Design of Existing Code -- Martin Fowler, Kent Beck -- 9780134757599.pdf",
        ],
        manning_files=[
            "Software Design for Python Programmers/Software_Design_for_Python_Programmers.pdf",
            "Practices of the Python Pro/Practices_of_the_Python_Pro.pdf",
            "The Well-Grounded Python Developer/The_Well-Grounded_Python_Developer.pdf",
        ],
        note_dirs=[
            "manning_software_design_python",
            "manning_practices_python_pro",
            "manning_well_grounded_python_dev",
            "manning_publishing_python_packages",
        ],
    )

    primary_rows = [row for row in rows if row["tier"] != "appendix"]
    titles = {row["title"] for row in primary_rows}

    assert "## Core Python OOP" in markdown
    assert "## Structured Notes" in markdown
    assert "## Supporting Classics" in markdown
    assert "Fluent Python" in titles
    assert "Effective Python" in titles
    assert "Software Design for Python Programmers" in titles
    assert "Practices of the Python Pro" in titles
    assert "The Well-Grounded Python Developer" in titles
    assert "Software Design for Python Programmers Notes" in titles
    assert "Practices of the Python Pro Notes" in titles
    assert "The Well-Grounded Python Developer Notes" in titles
    assert "Publishing Python Packages Notes" in titles
    assert "Refactoring" in titles
    assert "The Pragmatic Programmer" in titles
    assert "API Design Patterns" in titles
    assert "Unit Testing Principles, Practices, and Patterns" in titles


@pytest.mark.scripts
def test_duplicate_resolution_prefers_newer_final_pdfs(tmp_path):
    _markdown, rows = build_report(
        tmp_path,
        library_files=[
            "Luciano Ramalho - Fluent Python_ Clear Concise and Effective Programming-OReilly Media 2015.pdf"
        ],
        acquisition_files=[
            "Fluent Python_ Clear, Concise, and Effective Programming, -- Luciano Ramalho -- 2nd, 2022 -- OReilly -- 9781492056355.pdf",
            "Effective Python_ 125 Specific Ways to Write Better Python, -- Brett Slatkin -- 2024 -- Addison-Wesley -- 9780138172398.pdf",
            "Effective Python_ 125 Specific Ways to Write Better Python_ -- Brett Slatkin -- 3, 2024 nov 30 -- Pearson -- 9780138172183.pdf",
        ],
        top_download_files=[
            "Fluent Python _ clear, concise, and effective programming -- Luciano Ramalho -- Second edition, 2022 -- OReilly -- 9781492056355.epub",
            "Effective Python 90 Specific Ways to Write Better Python 2nd -- Brett Slatkin -- 9780134853987.epub",
        ],
        manning_files=[
            "Software Design for Python Programmers/Software_Design_for_Python_Programmers.pdf",
            "Software Design for Python Programmers/Software_Design_for_Python_Programmers_v8_MEAP.pdf",
            "Software Design for Python Programmers/Software_Design_for_Python_Programmers_v7_MEAP.epub",
        ],
    )

    by_title = {row["title"]: row for row in rows if row["tier"] != "appendix"}
    fluent = by_title["Fluent Python"]
    effective = by_title["Effective Python"]
    software_design = by_title["Software Design for Python Programmers"]

    assert "2022" in fluent["current_path"]
    assert fluent["current_path"].endswith(".pdf")
    assert fluent["status"] == "move_from_downloads"
    assert "2024" in effective["current_path"]
    assert effective["current_path"].endswith(".pdf")
    assert software_design["current_path"].endswith("Software_Design_for_Python_Programmers.pdf")
    assert "MEAP" not in software_design["current_path"]

    appendix_rows = [row for row in rows if row["tier"] == "appendix"]
    assert any(
        row["title"] == "Fluent Python"
        and row["status"] == "duplicate"
        and row["current_path"].endswith(".epub")
        for row in appendix_rows
    )
    assert any(
        row["title"] == "Software Design for Python Programmers" and "MEAP" in row["current_path"]
        for row in appendix_rows
    )


@pytest.mark.scripts
def test_appendix_captures_generic_oop_file_and_meap_variants(tmp_path):
    markdown, rows = build_report(
        tmp_path,
        library_files=["OOP.pdf"],
        manning_files=[
            "Software Design in Python/Software_Design_in_Python_v6_MEAP.pdf",
        ],
    )

    appendix_rows = [row for row in rows if row["tier"] == "appendix"]
    assert "## Appendix" in markdown
    assert any(
        row["title"] == "OOP.pdf"
        and row["status"] == "already_in_library"
        and "low_confidence_generic_title" in row["notes"]
        for row in appendix_rows
    )
    assert any(
        row["title"] == "Software Design for Python Programmers" and row["status"] == "meap_only"
        for row in appendix_rows
    )


@pytest.mark.scripts
def test_excludes_unrelated_python_and_design_material(tmp_path):
    _markdown, rows = build_report(
        tmp_path,
        top_download_files=[
            "Python_Concurrency_with_asyncio.pdf",
            "Agentic Design Patterns.pdf",
            "Deep Learning Design Patterns Primer.pdf",
        ],
    )

    assert rows == []
