"""BibTeX generation from extracted citations and sources.

Generates valid BibTeX entries for:
1. Source documents (papers, textbooks)
2. Citations extracted from source documents

Master Plan Reference: Line 584
"""

from typing import Optional

from research_kb_contracts import Citation, Source, SourceType


def escape_bibtex(text: str) -> str:
    """Escape special BibTeX characters.

    Args:
        text: Raw text to escape

    Returns:
        BibTeX-safe string with special chars escaped

    Example:
        >>> escape_bibtex("O'Connor & Jones")
        "O'Connor \\& Jones"
    """
    # Escape special chars (order matters)
    replacements = [
        ("\\", "\\\\"),  # Must be first
        ("&", "\\&"),
        ("%", "\\%"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def generate_bibtex_key(
    first_author: str,
    year: Optional[int],
    title: str,
) -> str:
    """Generate a standardized BibTeX citation key.

    Format: lastnameyearfirstword
    Example: pearl2009causality

    Args:
        first_author: First author's name
        year: Publication year
        title: Document title

    Returns:
        BibTeX citation key string
    """
    # Extract last name
    last_name = first_author.split()[-1].lower() if first_author else "unknown"
    # Remove non-alphanumeric
    last_name = "".join(c for c in last_name if c.isalnum())

    year_str = str(year) if year else "0000"

    # First significant word from title
    first_word = ""
    for word in title.split():
        word_clean = "".join(c for c in word.lower() if c.isalnum())
        if word_clean and word_clean not in (
            "the",
            "a",
            "an",
            "on",
            "of",
            "for",
            "and",
        ):
            first_word = word_clean
            break

    if not first_word:
        first_word = "untitled"

    return f"{last_name}{year_str}{first_word}"


def citation_to_bibtex(citation: Citation) -> str:
    """Convert a Citation to BibTeX entry string.

    Args:
        citation: Citation model from GROBID extraction

    Returns:
        Formatted BibTeX entry string

    Example:
        >>> cit = Citation(authors=["Judea Pearl"], title="Causality", year=2009, ...)
        >>> print(citation_to_bibtex(cit))
        @article{pearl2009causality,
          author = {Pearl, Judea},
          title = {Causality},
          year = {2009},
        }
    """
    key = citation.to_bibtex_key()

    # Determine entry type
    entry_type = "article" if citation.venue else "misc"

    lines = [f"@{entry_type}{{{key},"]

    # Authors (convert "First Last" to "Last, First")
    if citation.authors:
        formatted_authors = []
        for author in citation.authors:
            parts = author.split()
            if len(parts) >= 2:
                formatted_authors.append(f"{parts[-1]}, {' '.join(parts[:-1])}")
            else:
                formatted_authors.append(author)
        lines.append(
            f"  author = {{{escape_bibtex(' and '.join(formatted_authors))}}},"
        )

    # Title
    lines.append(f"  title = {{{escape_bibtex(citation.title)}}},")

    # Year
    if citation.year:
        lines.append(f"  year = {{{citation.year}}},")

    # Venue
    if citation.venue:
        lines.append(f"  journal = {{{escape_bibtex(citation.venue)}}},")

    # DOI
    if citation.doi:
        lines.append(f"  doi = {{{citation.doi}}},")

    # arXiv
    if citation.arxiv_id:
        lines.append(f"  eprint = {{{citation.arxiv_id}}},")
        lines.append("  archiveprefix = {arXiv},")

    lines.append("}")

    return "\n".join(lines)


def source_to_bibtex(source: Source) -> str:
    """Convert a Source document to BibTeX entry.

    Args:
        source: Source model from database

    Returns:
        Formatted BibTeX entry string

    Example:
        >>> src = Source(title="Causality", authors=["Pearl, Judea"], year=2009, ...)
        >>> print(source_to_bibtex(src))
        @book{pearl2009causality,
          author = {Pearl, Judea},
          title = {Causality},
          year = {2009},
        }
    """
    first_author = source.authors[0] if source.authors else "Unknown"
    key = generate_bibtex_key(first_author, source.year, source.title)

    # Determine entry type
    if source.source_type == SourceType.TEXTBOOK:
        entry_type = "book"
    elif source.source_type == SourceType.PAPER:
        entry_type = "article"
    else:
        entry_type = "misc"

    lines = [f"@{entry_type}{{{key},"]

    # Authors
    if source.authors:
        lines.append(f"  author = {{{escape_bibtex(' and '.join(source.authors))}}},")

    # Title
    lines.append(f"  title = {{{escape_bibtex(source.title)}}},")

    # Year
    if source.year:
        lines.append(f"  year = {{{source.year}}},")

    # Publisher from metadata
    publisher = source.metadata.get("publisher")
    if publisher:
        lines.append(f"  publisher = {{{escape_bibtex(publisher)}}},")

    # DOI from metadata
    doi = source.metadata.get("doi")
    if doi:
        lines.append(f"  doi = {{{doi}}},")

    # arXiv from metadata
    arxiv_id = source.metadata.get("arxiv_id")
    if arxiv_id:
        lines.append(f"  eprint = {{{arxiv_id}}},")
        lines.append("  archiveprefix = {arXiv},")

    lines.append("}")

    return "\n".join(lines)


def generate_bibliography(
    sources: list[Source],
    citations: Optional[list[Citation]] = None,
) -> str:
    """Generate complete BibTeX bibliography.

    Args:
        sources: List of Source documents
        citations: Optional list of Citations from documents

    Returns:
        Complete .bib file content

    Example:
        >>> bibtex = generate_bibliography(sources, citations)
        >>> with open("references.bib", "w") as f:
        ...     f.write(bibtex)
    """
    entries = []

    # Add source entries
    for source in sources:
        entries.append(source_to_bibtex(source))

    # Add citation entries
    if citations:
        entries.append("\n% Citations extracted from sources")
        for citation in citations:
            entries.append(citation_to_bibtex(citation))

    header = """\
% BibTeX bibliography generated by research-kb
% See: https://github.com/anthropic/research-kb
%
% Generated entries follow standard BibTeX format.
% Citation keys use format: lastnameyearfirstword

"""

    return header + "\n\n".join(entries)
