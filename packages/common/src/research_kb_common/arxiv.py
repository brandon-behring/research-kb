"""arXiv metadata fetcher (research-kb#20).

Resolve real ``title`` / ``authors`` / ``year`` for a paper from its arXiv ID via
the public arXiv Atom API, so PDFs ingested by filename don't store junk metadata
(``title="<arxiv_id>"``, ``authors=[]``, a guessed ``year``).

Stdlib-only (``urllib`` + ``ElementTree``) and **synchronous** by design, so it
slots into the existing sync ``get_metadata_for_pdf`` ingest path without forcing
async to propagate through every caller. The same arXiv endpoint already backs
the website's out-of-band enrich script, so KB-side and site-side metadata agree.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
import urllib.request
from xml.etree import ElementTree as ET

from research_kb_common.logging_config import get_logger

logger = get_logger(__name__)

_ARXIV_FILENAME_RE = re.compile(r"^(\d{4}\.\d{4,5})(v\d+)?\.pdf$")
_ARXIV_API = "http://export.arxiv.org/api/query?id_list={id}&max_results=1"
_ATOM = "{http://www.w3.org/2005/Atom}"
_USER_AGENT = "research-kb/1.0 (+https://github.com/brandon-behring; arxiv metadata)"


def arxiv_id_from_filename(filename: str) -> str | None:
    """Return the bare arXiv id for a ``<id>.pdf`` filename, else ``None``.

    Accepts an optional version suffix (``1312.5602v2.pdf`` -> ``1312.5602``).
    Non-arXiv filenames (``Kalman_1960.pdf``) return ``None``.
    """
    match = _ARXIV_FILENAME_RE.match(filename)
    return match.group(1) if match else None


def fetch_arxiv_metadata(arxiv_id: str, *, timeout: float = 10.0) -> dict | None:
    """Fetch ``{title, authors, year}`` for an arXiv id, or ``None`` on any failure.

    Returns ``None`` (never raises) so callers can fall back to filename parsing
    when arXiv is unreachable or the id is unknown. ``year`` may be ``None`` if the
    feed omits a parseable ``published`` date.

    Only the modern numeric arXiv id form is accepted; anything else (e.g. a
    comma-joined id from an unvalidated list) is rejected before any request, and
    the returned entry's id must match what was requested — so we can never accept
    a different paper's metadata (research-kb#20 review).
    """
    if not re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", arxiv_id):
        logger.warning("arxiv_id_malformed", arxiv_id=arxiv_id)
        return None

    url = _ARXIV_API.format(id=urllib.parse.quote(arxiv_id))
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("arxiv_fetch_failed", arxiv_id=arxiv_id, error=str(exc))
        return None

    try:
        root = ET.fromstring(body)
    except ET.ParseError as exc:
        logger.warning("arxiv_parse_failed", arxiv_id=arxiv_id, error=str(exc))
        return None

    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return None

    entry_id = entry.findtext(f"{_ATOM}id", default="")
    # arXiv returns a sentinel "Error" entry (id under /api/errors) for unknown ids.
    if "/api/errors" in entry_id:
        return None
    # Guard against the API returning a different paper than requested (alias,
    # redirect, or entry ordering): the returned abs id, version-stripped, must
    # equal the requested id (also version-stripped).
    returned_id = re.sub(r"v\d+$", "", entry_id.rstrip("/").rsplit("/", 1)[-1])
    if returned_id != re.sub(r"v\d+$", "", arxiv_id):
        logger.warning("arxiv_id_mismatch", requested=arxiv_id, returned=returned_id)
        return None

    title_raw = entry.findtext(f"{_ATOM}title", default="").strip()
    if not title_raw or title_raw.lower() == "error":
        return None
    title = " ".join(title_raw.split())  # collapse arXiv's embedded newlines

    authors: list[str] = []
    for author in entry.findall(f"{_ATOM}author"):
        name = author.findtext(f"{_ATOM}name", default="").strip()
        if name:
            authors.append(" ".join(name.split()))

    published = entry.findtext(f"{_ATOM}published", default="")
    year = int(published[:4]) if published[:4].isdigit() else None

    return {"title": title, "authors": authors, "year": year}
