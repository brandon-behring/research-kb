"""Pydantic models for Semantic Scholar API responses.

These models map directly to the S2 Academic Graph API schema.
See: https://api.semanticscholar.org/api-docs/graph
"""

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class OpenAccessPdf(BaseModel):
    """Open access PDF information."""

    url: str | None = None
    status: str | None = None  # "GREEN", "BRONZE", "GOLD", etc.

    model_config = ConfigDict(extra="ignore")


class S2Author(BaseModel):
    """Semantic Scholar author information.

    Note: Some fields are only available with explicit field requests.
    """

    author_id: str | None = Field(None, alias="authorId")
    external_ids: dict[str, Any] | None = Field(None, alias="externalIds")
    name: str | None = None
    url: str | None = None
    affiliations: list[str] | None = None
    paper_count: int | None = Field(None, alias="paperCount")
    citation_count: int | None = Field(None, alias="citationCount")
    h_index: int | None = Field(None, alias="hIndex")

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class S2Paper(BaseModel):
    """Semantic Scholar paper information.

    Core model for paper metadata from the Academic Graph API.
    All fields are optional because API returns vary by requested fields.
    """

    # Identifiers
    paper_id: str | None = Field(None, alias="paperId")
    corpus_id: int | None = Field(None, alias="corpusId")
    external_ids: dict[str, Any] | None = Field(None, alias="externalIds")

    # Bibliographic
    title: str | None = None
    abstract: str | None = None
    venue: str | None = None
    publication_venue: dict[str, Any] | None = Field(None, alias="publicationVenue")
    year: int | None = None
    publication_date: str | None = Field(None, alias="publicationDate")

    # Authors
    authors: list[S2Author] | None = None

    # Metrics
    reference_count: int | None = Field(None, alias="referenceCount")
    citation_count: int | None = Field(None, alias="citationCount")
    influential_citation_count: int | None = Field(None, alias="influentialCitationCount")
    s2_fields_of_study: list[dict[str, Any]] | None = Field(None, alias="s2FieldsOfStudy")
    publication_types: list[str] | None = Field(None, alias="publicationTypes")

    # Access
    is_open_access: bool | None = Field(None, alias="isOpenAccess")
    open_access_pdf: OpenAccessPdf | None = Field(None, alias="openAccessPdf")
    url: str | None = None

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @property
    def doi(self) -> str | None:
        """Extract DOI from external IDs."""
        if self.external_ids:
            return self.external_ids.get("DOI")
        return None

    @property
    def arxiv_id(self) -> str | None:
        """Extract arXiv ID from external IDs."""
        if self.external_ids:
            return self.external_ids.get("ArXiv")
        return None

    @property
    def first_author_name(self) -> str | None:
        """Get first author's name (for filename generation)."""
        if self.authors and len(self.authors) > 0:
            return self.authors[0].name
        return None

    def to_metadata_dict(self) -> dict[str, Any]:
        """Convert to dict suitable for storing in sources.metadata JSONB.

        Returns a flattened dict with s2_ prefix for clarity.
        """
        return {
            "s2_paper_id": self.paper_id,
            "s2_corpus_id": self.corpus_id,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "citation_count": self.citation_count,
            "influential_citation_count": self.influential_citation_count,
            "is_open_access": self.is_open_access,
            "fields_of_study": [f.get("category") for f in (self.s2_fields_of_study or [])],
            "s2_enriched_at": datetime.now(timezone.utc).isoformat(),
        }


class S2SearchResult(BaseModel):
    """Result from paper search endpoint.

    Contains papers plus pagination info.
    """

    total: int = 0
    offset: int = 0
    next_offset: int | None = Field(None, alias="next")
    data: list[S2Paper] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class S2AuthorPapersResult(BaseModel):
    """Result from author papers endpoint."""

    total: int = 0
    offset: int = 0
    next_offset: int | None = Field(None, alias="next")
    data: list[S2Paper] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", populate_by_name=True)


class S2CitationsResult(BaseModel):
    """Result from paper citations endpoint.

    Each item wraps the citing paper in {"citingPaper": {...}}.
    """

    offset: int = 0
    next_offset: int | None = Field(None, alias="next")
    data: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @property
    def papers(self) -> list[S2Paper]:
        """Extract S2Paper objects from wrapped response."""
        return [
            S2Paper(**item["citingPaper"])
            for item in self.data
            if item.get("citingPaper")
        ]


class S2ReferencesResult(BaseModel):
    """Result from paper references endpoint.

    Each item wraps the cited paper in {"citedPaper": {...}}.
    """

    offset: int = 0
    next_offset: int | None = Field(None, alias="next")
    data: list[dict[str, Any]] = Field(default_factory=list)

    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    @property
    def papers(self) -> list[S2Paper]:
        """Extract S2Paper objects from wrapped response."""
        return [
            S2Paper(**item["citedPaper"])
            for item in self.data
            if item.get("citedPaper")
        ]
