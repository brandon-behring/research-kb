# S2 Client

Async Python client for the Semantic Scholar Academic Graph API.

## Features

- **Rate Limiting**: Configurable RPS with automatic backoff
- **Response Caching**: SQLite-based cache with 7-day TTL
- **Type Safety**: Pydantic models for all API responses
- **Retry Logic**: Automatic retry on transient errors
- **Paper Acquisition**: Discover and download open-access papers
- **Citation Enrichment**: Match extracted citations to S2 papers

## Installation

```bash
pip install -e packages/s2-client
```

## Quick Start

```python
from s2_client import S2Client, S2Paper

async with S2Client() as client:
    # Search for papers
    papers = await client.search_papers("causal forest", limit=10)
    for paper in papers:
        print(f"{paper.title} ({paper.year}) - {paper.citation_count} citations")

    # Get paper details
    paper = await client.get_paper("DOI:10.1234/example")

    # Get citations
    citations = await client.get_citations(paper.paper_id, limit=100)
```

## CLI Commands

The s2-client integrates with the research-kb CLI:

```bash
# Search Semantic Scholar
research-kb discover search "double machine learning"

# Browse by topic
research-kb discover topics

# Find papers by author
research-kb discover author "Chernozhukov"

# Enrich corpus with S2 metadata
research-kb enrich citations

# Show enrichment status
research-kb enrich status
```

## Core Components

| Module | Purpose |
|--------|---------|
| `S2Client` | Main async API client |
| `S2Paper` | Paper data model |
| `S2Author` | Author data model |
| `SearchFilters` | Search parameters |
| `TopicDiscovery` | Topic-based paper discovery |
| `PaperAcquisition` | Open-access paper download |
| `match_citation` | Citation-to-paper matching |
| `S2Cache` | Response caching |
| `RateLimiter` | Rate limit management |

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `S2_API_KEY` | None | Semantic Scholar API key (optional, higher limits) |
| `S2_CACHE_DIR` | `~/.cache/s2` | Cache directory |
| `S2_RPS` | 3 | Requests per second |

## Rate Limits

Without API key:
- 100 requests per 5 minutes

With API key:
- Higher limits (contact S2 for details)

The client automatically handles rate limiting with exponential backoff.

## Testing

```bash
pytest packages/s2-client/tests/ -v
```

## API Reference

See the [Semantic Scholar API documentation](https://api.semanticscholar.org/api-docs/) for full endpoint details.
