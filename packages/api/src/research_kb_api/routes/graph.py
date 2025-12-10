"""Graph exploration endpoints.

Provides knowledge graph traversal capabilities for concept relationships.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from research_kb_api import schemas
from research_kb_api import service

router = APIRouter()


@router.get("/neighborhood/{concept_name}", response_model=schemas.GraphNeighborhood)
async def get_neighborhood(
    concept_name: str,
    hops: int = Query(2, ge=1, le=5, description="Number of hops from center"),
    limit: int = Query(50, ge=1, le=200, description="Maximum nodes to return"),
) -> schemas.GraphNeighborhood:
    """Get the neighborhood of a concept in the knowledge graph.

    Traverses relationships up to N hops from the center concept.

    Parameters
    ----------
    concept_name : str
        Name of the center concept
    hops : int
        Number of relationship hops to traverse (1-5)
    limit : int
        Maximum nodes to include in result

    Returns
    -------
    GraphNeighborhood
        Center node with all connected nodes and edges within hop distance.

    Raises
    ------
    HTTPException
        404 if concept not found
    """
    result = await service.get_graph_neighborhood(
        concept_name=concept_name,
        hops=hops,
        limit=limit,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    center_data = result.get("center", {})
    center = schemas.GraphNode(
        id=center_data.get("id", ""),
        name=center_data.get("name", concept_name),
        type=center_data.get("type"),
    )

    nodes = [
        schemas.GraphNode(
            id=n.get("id", ""),
            name=n.get("name", ""),
            type=n.get("type"),
        )
        for n in result.get("nodes", [])
    ]

    edges = [
        schemas.GraphEdge(
            source=e.get("source", ""),
            target=e.get("target", ""),
            type=e.get("type"),
        )
        for e in result.get("edges", [])
    ]

    return schemas.GraphNeighborhood(
        center=center,
        nodes=nodes,
        edges=edges,
    )


@router.get("/path/{concept_a}/{concept_b}", response_model=schemas.GraphPath)
async def get_path(
    concept_a: str,
    concept_b: str,
) -> schemas.GraphPath:
    """Find the shortest path between two concepts.

    Uses breadth-first search to find the shortest relationship path.

    Parameters
    ----------
    concept_a : str
        Name of the starting concept
    concept_b : str
        Name of the target concept

    Returns
    -------
    GraphPath
        Sequence of concepts forming the shortest path.

    Raises
    ------
    HTTPException
        404 if either concept not found or no path exists
    """
    result = await service.get_graph_path(concept_a, concept_b)

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    path_nodes = result.get("path", [])

    # Convert path to GraphNode list
    nodes = []
    for item in path_nodes:
        if isinstance(item, dict):
            nodes.append(schemas.GraphNode(
                id=item.get("id", ""),
                name=item.get("name", ""),
                type=item.get("type"),
            ))
        elif isinstance(item, str):
            # Simple string path - just names
            nodes.append(schemas.GraphNode(
                id="",
                name=item,
                type=None,
            ))

    return schemas.GraphPath(
        from_concept=concept_a,
        to_concept=concept_b,
        path=nodes,
        path_length=len(nodes) - 1 if nodes else 0,
    )
