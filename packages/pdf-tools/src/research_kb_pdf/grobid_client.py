"""GROBID client for academic paper extraction.

Sends PDFs to GROBID service for structure-aware extraction.
Parses TEI-XML responses into structured paper content (IMRAD format).
Extracts citations from reference sections for BibTeX generation.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import requests
from lxml import etree

from research_kb_common import get_logger
from research_kb_contracts import Citation

logger = get_logger(__name__)

# GROBID configuration
DEFAULT_GROBID_URL = "http://localhost:8070"
PROCESS_FULLTEXT_ENDPOINT = "/api/processFulltextDocument"
REQUEST_TIMEOUT = 60  # seconds


@dataclass
class PaperSection:
    """A section from an academic paper."""

    heading: str
    content: str
    level: int  # 1 for main sections, 2 for subsections


@dataclass
class PaperMetadata:
    """Metadata extracted from academic paper."""

    title: str
    authors: list[str]
    abstract: Optional[str]
    year: Optional[int]


@dataclass
class ExtractedPaper:
    """Complete extracted academic paper with structure."""

    metadata: PaperMetadata
    sections: list[PaperSection]
    citations: list[Citation] = field(default_factory=list)  # Extracted from <listBibl>
    raw_text: str = ""  # Full text for fallback
    tei_xml: str = ""  # Original TEI-XML for debugging


class GrobidClient:
    """Client for GROBID academic paper processing service."""

    def __init__(self, grobid_url: str = DEFAULT_GROBID_URL):
        """Initialize GROBID client.

        Args:
            grobid_url: Base URL for GROBID service

        Example:
            >>> client = GrobidClient()
            >>> paper = client.process_pdf("paper.pdf")
        """
        self.grobid_url = grobid_url.rstrip("/")
        self.process_url = self.grobid_url + PROCESS_FULLTEXT_ENDPOINT

    def is_alive(self) -> bool:
        """Check if GROBID service is running.

        Returns:
            True if service is responsive

        Example:
            >>> client = GrobidClient()
            >>> if client.is_alive():
            ...     print("GROBID ready")
        """
        try:
            response = requests.get(f"{self.grobid_url}/api/isalive", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def process_pdf(self, pdf_path: str | Path) -> ExtractedPaper:
        """Process PDF with GROBID to extract structured content.

        Args:
            pdf_path: Path to PDF file

        Returns:
            ExtractedPaper with metadata and structured sections

        Raises:
            FileNotFoundError: If PDF doesn't exist
            ConnectionError: If GROBID service unavailable
            ValueError: If GROBID processing fails

        Example:
            >>> client = GrobidClient()
            >>> paper = client.process_pdf("paper.pdf")
            >>> print(paper.metadata.title)
            >>> for section in paper.sections:
            ...     print(f"{section.heading}: {len(section.content)} chars")
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        if not self.is_alive():
            raise ConnectionError(
                f"GROBID service not available at {self.grobid_url}. "
                f"Start with: docker-compose up grobid"
            )

        logger.info("processing_pdf_grobid", path=str(pdf_path))

        try:
            with open(pdf_path, "rb") as pdf_file:
                files = {"input": pdf_file}
                response = requests.post(
                    self.process_url,
                    files=files,
                    timeout=REQUEST_TIMEOUT,
                )
                response.raise_for_status()

        except requests.Timeout:
            raise ValueError(f"GROBID processing timeout for: {pdf_path}")
        except requests.RequestException as e:
            raise ValueError(f"GROBID processing failed: {e}")

        tei_xml = response.text

        # Parse TEI-XML
        paper = parse_tei_xml(tei_xml)

        logger.info(
            "grobid_processed",
            path=str(pdf_path),
            sections=len(paper.sections),
            chars=len(paper.raw_text),
        )

        return paper


def parse_tei_xml(tei_xml: str) -> ExtractedPaper:
    """Parse GROBID TEI-XML into structured paper.

    Args:
        tei_xml: TEI-XML string from GROBID

    Returns:
        ExtractedPaper with parsed content including citations

    Example:
        >>> with open("paper.tei.xml") as f:
        ...     paper = parse_tei_xml(f.read())
        >>> print(f"Found {len(paper.citations)} citations")
    """
    # Parse XML
    root = etree.fromstring(tei_xml.encode("utf-8"))

    # Define namespaces
    ns = {"tei": "http://www.tei-c.org/ns/1.0"}

    # Extract metadata
    metadata = _extract_metadata(root, ns)

    # Extract sections
    sections = _extract_sections(root, ns)

    # Extract citations from bibliography
    citations = _extract_citations(root, ns)

    # Extract full text
    raw_text = _extract_full_text(root, ns)

    return ExtractedPaper(
        metadata=metadata,
        sections=sections,
        citations=citations,
        raw_text=raw_text,
        tei_xml=tei_xml,
    )


def _extract_metadata(root, ns: dict) -> PaperMetadata:
    """Extract paper metadata from TEI-XML."""
    # Title
    title_elem = root.find(".//tei:titleStmt/tei:title[@type='main']", ns)
    title = title_elem.text if title_elem is not None else "Untitled"

    # Authors
    authors = []
    for author in root.findall(".//tei:sourceDesc//tei:author", ns):
        forename = author.find(".//tei:forename", ns)
        surname = author.find(".//tei:surname", ns)
        if forename is not None and surname is not None:
            authors.append(f"{forename.text} {surname.text}")

    # Abstract
    abstract_elem = root.find(".//tei:profileDesc//tei:abstract", ns)
    abstract = None
    if abstract_elem is not None:
        abstract = _get_text_content(abstract_elem)

    # Year
    year = None
    date_elem = root.find(".//tei:publicationStmt//tei:date[@type='published']", ns)
    if date_elem is not None and "when" in date_elem.attrib:
        try:
            year = int(date_elem.attrib["when"][:4])
        except (ValueError, IndexError):
            pass

    return PaperMetadata(
        title=title.strip(),
        authors=authors,
        abstract=abstract.strip() if abstract else None,
        year=year,
    )


def _extract_sections(root, ns: dict) -> list[PaperSection]:
    """Extract paper sections from TEI-XML."""
    sections = []

    # Find all div elements (sections)
    for div in root.findall(".//tei:text//tei:div", ns):
        # Get heading
        head = div.find("tei:head", ns)
        if head is None:
            continue

        heading = _get_text_content(head).strip()
        if not heading:
            continue

        # Get section content (all paragraphs)
        paragraphs = []
        for p in div.findall(".//tei:p", ns):
            para_text = _get_text_content(p).strip()
            if para_text:
                paragraphs.append(para_text)

        if paragraphs:
            content = "\n\n".join(paragraphs)
            # Determine level by checking nesting
            level = len(list(div.iterancestors("{http://www.tei-c.org/ns/1.0}div"))) + 1

            sections.append(PaperSection(heading=heading, content=content, level=level))

    return sections


def _extract_full_text(root, ns: dict) -> str:
    """Extract all text content from paper."""
    text_elem = root.find(".//tei:text", ns)
    if text_elem is None:
        return ""

    return _get_text_content(text_elem).strip()


def _get_text_content(element) -> str:
    """Get all text content from element recursively."""
    return "".join(element.itertext())


def _extract_citations(root, ns: dict) -> list[Citation]:
    """Extract citations from TEI-XML bibliography.

    Parses <listBibl> element containing <biblStruct> entries.
    Each entry becomes a Citation with extracted metadata.
    """
    citations = []

    # Find all bibliography entries
    for bibl in root.findall(".//tei:listBibl/tei:biblStruct", ns):
        try:
            # Extract authors
            authors = []
            for author in bibl.findall(".//tei:author", ns):
                forename = author.find(".//tei:forename", ns)
                surname = author.find(".//tei:surname", ns)
                if surname is not None:
                    name_parts = []
                    if forename is not None and forename.text:
                        name_parts.append(forename.text.strip())
                    if surname.text:
                        name_parts.append(surname.text.strip())
                    if name_parts:
                        authors.append(" ".join(name_parts))

            # Extract title (analytic for article, monogr for book)
            title = None
            title_elem = bibl.find(".//tei:analytic/tei:title", ns)
            if title_elem is None:
                title_elem = bibl.find(".//tei:monogr/tei:title", ns)
            if title_elem is not None and title_elem.text:
                title = title_elem.text.strip()

            if not title:
                continue  # Skip entries without title

            # Extract year
            year = None
            date_elem = bibl.find(".//tei:imprint/tei:date[@when]", ns)
            if date_elem is not None:
                try:
                    year = int(date_elem.attrib["when"][:4])
                except (ValueError, KeyError):
                    pass

            # Extract venue (journal or conference)
            venue = None
            venue_elem = bibl.find(".//tei:monogr/tei:title[@level='j']", ns)  # Journal
            if venue_elem is None:
                venue_elem = bibl.find(
                    ".//tei:monogr/tei:title[@level='m']", ns
                )  # Book/proceedings
            if venue_elem is not None and venue_elem.text:
                venue = venue_elem.text.strip()

            # Extract DOI
            doi = None
            doi_elem = bibl.find(".//tei:idno[@type='DOI']", ns)
            if doi_elem is not None and doi_elem.text:
                doi = doi_elem.text.strip()

            # Extract arXiv ID
            arxiv_id = None
            arxiv_elem = bibl.find(".//tei:idno[@type='arXiv']", ns)
            if arxiv_elem is not None and arxiv_elem.text:
                arxiv_id = arxiv_elem.text.strip()

            # Build raw string from all text content
            raw_string = _get_text_content(bibl).strip()

            citations.append(
                Citation(
                    authors=authors,
                    title=title,
                    year=year,
                    venue=venue,
                    doi=doi,
                    arxiv_id=arxiv_id,
                    raw_string=raw_string or title,
                )
            )

        except Exception as e:
            logger.warning("citation_extraction_failed", error=str(e))
            continue

    logger.info("citations_extracted", count=len(citations))
    return citations
