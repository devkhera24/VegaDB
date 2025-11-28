#!/usr/bin/env python3
"""
api_v2.py - Generic Vector+Graph Hybrid Database API
Works with ANY data - no hardcoding
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os
import math
from datetime import datetime

# Core imports
from core.embeddings import Embeddings

app = FastAPI(
    title="Vector+Graph Hybrid Database API",
    description="Generic CRUD + Hybrid Search - Works with ANY data",
    version="2.0"
)

# ==================== INITIALIZATION ====================
MOCK_MODE = os.environ.get("MOCK_EMBEDDINGS", "0") == "1"
embedder = Embeddings(mock_mode=MOCK_MODE)

# In-memory stores (completely generic)
metadata_store = {}  # {node_id: {text, title, metadata, embedding, created_at}}
graph_edges = {}     # {source_id: [(target_id, type, weight), ...]}
reverse_edges = {}   # {target_id: [(source_id, type, weight), ...]}

# ==================== PYDANTIC MODELS ====================
class NodeCreate(BaseModel):
    id: Optional[str] = None
    text: str
    title: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    embedding: Optional[List[float]] = None

class NodeUpdate(BaseModel):
    text: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    regen_embedding: bool = False

class EdgeCreate(BaseModel):
    source: str
    target: str
    type: str
    weight: float = 1.0

class VectorSearchRequest(BaseModel):
    query_text: str
    query_embedding: Optional[List[float]] = None
    top_k: int = 5

class HybridSearchRequest(BaseModel):
    query_text: str
    query_embedding: Optional[List[float]] = None
    vector_weight: float = 0.6
    graph_weight: float = 0.4
    top_k: int = 5

# ==================== PURE GENERIC HELPER FUNCTIONS ====================
def cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Generic cosine similarity - works for any vectors"""
    if len(v1) != len(v2):
        raise ValueError("Vectors must have same dimension")
    
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def bfs_traverse(start_id: str, max_depth: int) -> List[Dict[str, Any]]:
    """Generic BFS - works with any graph structure"""
    visited = set()
    queue = [(start_id, 0, None, None)]
    results = []
    
    while queue:
        node_id, depth, edge_type, weight = queue.pop(0)
        
        if node_id in visited:
            continue
        
        visited.add(node_id)
        
        if depth > 0:  # Skip start node
            results.append({
                "id": node_id,
                "hop": depth,
                "edge": edge_type,
                "weight": weight
            })
        
        if depth < max_depth:
            # Forward edges
            for target_id, rel_type, rel_weight in graph_edges.get(node_id, []):
                if target_id not in visited:
                    queue.append((target_id, depth + 1, rel_type, rel_weight))
            
            # Reverse edges (undirected traversal)
            for source_id, rel_type, rel_weight in reverse_edges.get(node_id, []):
                if source_id not in visited:
                    queue.append((source_id, depth + 1, rel_type, rel_weight))
    
    return results

def get_shortest_hop(from_id: str, to_id: str, max_depth: int = 5) -> int:
    """Generic shortest path - works with any graph"""
    if from_id == to_id:
        return 0
    
    visited = set()
    queue = [(from_id, 0)]
    
    while queue:
        node_id, depth = queue.pop(0)
        
        if node_id == to_id:
            return depth
        
        if node_id in visited or depth >= max_depth:
            continue
        
        visited.add(node_id)
        
        # Check all neighbors
        for target_id, _, _ in graph_edges.get(node_id, []):
            if target_id not in visited:
                queue.append((target_id, depth + 1))
        
        for source_id, _, _ in reverse_edges.get(node_id, []):
            if source_id not in visited:
                queue.append((source_id, depth + 1))
    
    return -1  # Unreachable

# ==================== HEALTH ====================
@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "mock_mode": MOCK_MODE,
        "nodes": len(metadata_store),
        "edges": sum(len(v) for v in graph_edges.values())
    }

# ==================== NODE CRUD ====================
@app.post("/nodes")
async def create_node(node: NodeCreate):
    """Create node - works with ANY text"""
    node_id = node.id or f"node_{len(metadata_store) + 1}"
    
    # Generate or use provided embedding
    if node.embedding:
        embedding = node.embedding
    else:
        embedding = embedder.embed(node.text)
    
    # Store metadata (generic)
    metadata_store[node_id] = {
        "id": node_id,
        "text": node.text,
        "title": node.title or node_id,
        "metadata": node.metadata or {},
        "embedding": embedding,
        "created_at": datetime.utcnow().isoformat() + "Z"
    }
    
    return {
        "status": "created",
        "id": node_id,
        "created_at": metadata_store[node_id]["created_at"]
    }

@app.get("/nodes/{node_id}")
async def get_node(node_id: str):
    """Get node - generic retrieval"""
    if node_id not in metadata_store:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = metadata_store[node_id]
    
    # Get all edges
    edges = [
        {"type": et, "target_id": tid, "weight": w}
        for tid, et, w in graph_edges.get(node_id, [])
    ]
    
    return {
        "id": node_id,
        "title": node["title"],
        "text": node["text"],
        "metadata": node["metadata"],
        "embedding": node["embedding"],
        "edges": edges
    }

@app.put("/nodes/{node_id}")
async def update_node(node_id: str, update: NodeUpdate):
    """Update node - generic logic"""
    if node_id not in metadata_store:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node = metadata_store[node_id]
    old_embedding = node["embedding"]
    
    if update.text:
        node["text"] = update.text
        
        if update.regen_embedding:
            new_embedding = embedder.embed(update.text)
            node["embedding"] = new_embedding
            
            similarity = cosine_similarity(old_embedding, new_embedding)
            
            return {
                "status": "updated",
                "id": node_id,
                "embedding_changed": True,
                "similarity": round(similarity, 8)
            }
    
    if update.metadata:
        node["metadata"].update(update.metadata)
    
    return {"status": "updated", "id": node_id}

@app.delete("/nodes/{node_id}")
async def delete_node(node_id: str):
    """Delete node - generic cascading"""
    if node_id not in metadata_store:
        raise HTTPException(status_code=404, detail="Node not found")
    
    del metadata_store[node_id]
    
    edges_deleted = 0
    
    if node_id in graph_edges:
        edges_deleted += len(graph_edges[node_id])
        del graph_edges[node_id]
    
    if node_id in reverse_edges:
        edges_deleted += len(reverse_edges[node_id])
        del reverse_edges[node_id]
    
    # Clean up all references
    for source in list(graph_edges.keys()):
        graph_edges[source] = [(t, et, w) for t, et, w in graph_edges[source] if t != node_id]
    
    for target in list(reverse_edges.keys()):
        reverse_edges[target] = [(s, et, w) for s, et, w in reverse_edges[target] if s != node_id]
    
    return {"status": "deleted", "id": node_id, "removed_edges_count": edges_deleted}

# ==================== EDGE CRUD ====================
@app.post("/edges")
async def create_edge(edge: EdgeCreate):
    """Create edge - generic graph building"""
    if edge.source not in metadata_store:
        raise HTTPException(status_code=404, detail="Source node not found")
    if edge.target not in metadata_store:
        raise HTTPException(status_code=404, detail="Target node not found")
    
    # Add to forward edges
    if edge.source not in graph_edges:
        graph_edges[edge.source] = []
    graph_edges[edge.source].append((edge.target, edge.type, edge.weight))
    
    # Add to reverse edges
    if edge.target not in reverse_edges:
        reverse_edges[edge.target] = []
    reverse_edges[edge.target].append((edge.source, edge.type, edge.weight))
    
    return {
        "status": "created",
        "source": edge.source,
        "target": edge.target
    }

# ==================== VECTOR SEARCH ====================
@app.post("/search/vector")
async def vector_search(request: VectorSearchRequest):
    """
    Vector-only search - COMPLETELY GENERIC
    Works with ANY embeddings, ANY text
    """
    # Generate query embedding (or use provided)
    if request.query_embedding:
        query_vec = request.query_embedding
    else:
        query_vec = embedder.embed(request.query_text)
    
    # Calculate similarities for ALL nodes (generic)
    results = []
    for node_id, node in metadata_store.items():
        score = cosine_similarity(query_vec, node["embedding"])
        results.append({
            "id": node_id,
            "title": node["title"],
            "vector_score": round(score, 8)
        })
    
    # Sort by score (generic ranking)
    results.sort(key=lambda x: x["vector_score"], reverse=True)
    
    return {
        "query_text": request.query_text,
        "results": results[:request.top_k]
    }

# ==================== GRAPH TRAVERSAL ====================
@app.get("/search/graph")
async def graph_traversal(
    start_id: str = Query(..., description="Starting node ID"),
    depth: int = Query(1, description="Maximum traversal depth")
):
    """
    Graph traversal - COMPLETELY GENERIC BFS
    Works with ANY graph structure
    """
    if start_id not in metadata_store:
        raise HTTPException(status_code=404, detail="Start node not found")
    
    nodes = bfs_traverse(start_id, depth)
    
    return {
        "start_id": start_id,
        "depth": depth,
        "nodes": nodes
    }

# ==================== HYBRID SEARCH ====================
@app.post("/search/hybrid")
async def hybrid_search(request: HybridSearchRequest):
    """
    Hybrid search - COMPLETELY GENERIC
    Combines vector similarity + graph proximity for ANY data
    
    Algorithm:
    1. Calculate vector scores for all nodes
    2. Find anchor (highest vector score)
    3. Calculate graph proximity to anchor
    4. Combine: final_score = vector_weight * vector_score + graph_weight * graph_score
    """
    # Get query embedding
    if request.query_embedding:
        query_vec = request.query_embedding
    else:
        query_vec = embedder.embed(request.query_text)
    
    # Calculate vector scores (generic)
    anchor_id = None
    max_vector_score = -1
    vector_scores = {}
    
    for node_id, node in metadata_store.items():
        score = cosine_similarity(query_vec, node["embedding"])
        vector_scores[node_id] = score
        if score > max_vector_score:
            max_vector_score = score
            anchor_id = node_id
    
    if not anchor_id:
        return {"query_text": request.query_text, "results": []}
    
    # Calculate hybrid scores (generic formula)
    results = []
    for node_id, node in metadata_store.items():
        vector_score = vector_scores[node_id]
        
        # Graph proximity: 1/(1+hops)
        if node_id == anchor_id:
            graph_score = 1.0
            hop = 0
        else:
            hop = get_shortest_hop(anchor_id, node_id)
            if hop == -1:
                graph_score = 0.0
                hop = None
            else:
                graph_score = 1.0 / (1.0 + hop)
        
        # Generic weighted combination
        final_score = (
            request.vector_weight * vector_score +
            request.graph_weight * graph_score
        )
        
        # Get edge info if directly connected
        edge_info = None
        edge_weight = None
        if hop == 1:
            for target_id, edge_type, weight in graph_edges.get(anchor_id, []):
                if target_id == node_id:
                    edge_info = edge_type
                    edge_weight = weight
                    break
            
            if not edge_info:
                for source_id, edge_type, weight in reverse_edges.get(anchor_id, []):
                    if source_id == node_id:
                        edge_info = edge_type
                        edge_weight = weight
                        break
        
        result = {
            "id": node_id,
            "title": node["title"],
            "vector_score": round(vector_score, 8),
            "graph_score": round(graph_score, 8),
            "final_score": round(final_score, 8),
            "info": {"hop": hop}
        }
        
        if edge_info:
            result["info"]["edge"] = edge_info
            result["info"]["edge_weight"] = edge_weight
        
        results.append(result)
    
    # Generic sorting
    results.sort(key=lambda x: x["final_score"], reverse=True)
    
    return {
        "query_text": request.query_text,
        "vector_weight": request.vector_weight,
        "graph_weight": request.graph_weight,
        "results": results[:request.top_k]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)