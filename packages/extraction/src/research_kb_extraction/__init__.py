"""Research KB Extraction - Concept extraction using Ollama LLM.

This package provides:
- OllamaClient: GPU-accelerated LLM wrapper for structured output
- ConceptExtractor: Extract concepts and relationships from text chunks
- Deduplicator: Canonical name normalization and embedding-based deduplication
- GraphSyncService: Sync concepts to Neo4j graph database
"""

from research_kb_extraction.models import (
    ChunkExtraction,
    ConceptMatch,
    ExtractedConcept,
    ExtractedRelationship,
    StoredConcept,
)
from research_kb_extraction.ollama_client import OllamaClient, OllamaError
from research_kb_extraction.concept_extractor import ConceptExtractor
from research_kb_extraction.deduplicator import Deduplicator, ABBREVIATION_MAP
from research_kb_extraction.graph_sync import GraphSyncService, GraphSyncError

__all__ = [
    # Models
    "ExtractedConcept",
    "ExtractedRelationship",
    "ChunkExtraction",
    "ConceptMatch",
    "StoredConcept",
    # Clients
    "OllamaClient",
    "OllamaError",
    "ConceptExtractor",
    "Deduplicator",
    "ABBREVIATION_MAP",
    "GraphSyncService",
    "GraphSyncError",
]
