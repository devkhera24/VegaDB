#!/usr/bin/env python3
"""
test_canonical_examples.py - Test exact examples from problem statement
Uses mock embeddings for deterministic results
"""
import requests
import json
import os
import math

# Set mock mode
os.environ["MOCK_EMBEDDINGS"] = "1"

API_BASE = "http://localhost:8000"

# Canonical mock embeddings from problem statement
MOCK_EMBEDDINGS = {
    "doc1": [0.90, 0.10, 0.00, 0.00, 0.00, 0.00],
    "doc2": [0.70, 0.10, 0.60, 0.00, 0.00, 0.00],
    "doc3": [0.10, 0.05, 0.00, 0.90, 0.00, 0.00],
    "doc4": [0.80, 0.15, 0.00, 0.00, 0.00, 0.00],
    "doc5": [0.05, 0.00, 0.90, 0.10, 0.00, 0.00],
    "doc6": [0.60, 0.05, 0.50, 0.00, 0.10, 0.00],
}

# Canonical documents
DOCS = {
    "doc1": "Redis became the default choice for caching mostly because people like avoiding slow databases. There are the usual headaches: eviction policies like LRU vs LFU, memory pressure, and when someone forgets to set TTLs and wonders why servers fall over.",
    "doc2": "The RedisGraph module promises a weird marriage: pretend your cache is also a graph database. Honestly, it works better than expected. You can store relationships like user -> viewed -> product and then still query it with cypher-like syntax.",
    "doc3": "Distributed systems are basically long-distance relationships. Nodes drift apart, messages get lost, and during network partitions everyone blames everyone else. Leader election decides who gets boss privileges until the next heartbeat timeout.",
    "doc4": "A short note on cache invalidation: you think you understand it until your application grows. Patterns like write-through, write-behind, and cache-aside all behave differently under load. Versioned keys help, but someone will always ship code that forgets to update them.",
    "doc5": "Graph algorithms show up in real life more than people notice. Social feeds rely on BFS for exploring connections, recommendations rely on random walks, and PageRank still refuses to die. Even your team's on-call rotation effectively forms a directed cycle.",
    "doc6": "README draft: to combine Redis with a graph database, you start by defining nodes for each entity, like articles, users, or configuration snippets. Then you create edges describing interactions: mentions, references, imports. The magic happens when semantic search embeddings overlay this structure.",
}

TITLES = {
    "doc1": "Redis caching strategies",
    "doc2": "RedisGraph module",
    "doc3": "Distributed systems",
    "doc4": "Cache invalidation note",
    "doc5": "Graph algorithms",
    "doc6": "README: Redis+Graph",
}

def setup_canonical_data():
    """Ingest canonical test documents"""
    print("Setting up canonical test data...")
    
    for doc_id, text in DOCS.items():
        response = requests.post(
            f"{API_BASE}/nodes",
            json={
                "id": doc_id,
                "text": text,
                "title": TITLES[doc_id],
                "embedding": MOCK_EMBEDDINGS[doc_id]
            }
        )
        if response.status_code == 200:
            print(f"✓ Created {doc_id}")
        else:
            print(f"✗ Failed to create {doc_id}: {response.text}")
    
    # Create canonical edges
    edges = [
        ("doc1", "doc4", "related_to", 0.8),
        ("doc2", "doc6", "mentions", 0.9),
        ("doc6", "doc1", "references", 0.6),
        ("doc3", "doc5", "related_to", 0.5),
        ("doc2", "doc5", "example_of", 0.3),
    ]
    
    for source, target, edge_type, weight in edges:
        response = requests.post(
            f"{API_BASE}/edges",
            json={
                "source": source,
                "target": target,
                "type": edge_type,
                "weight": weight
            }
        )
        if response.status_code == 200:
            print(f"✓ Created edge {source}->{target}")

def test_vector_only():
    """Test vector-only search from problem statement"""
    print("\n" + "="*70)
    print("TEST: Vector-Only Search (Expected: doc1, doc4, doc2, doc6, doc5)")
    print("="*70)
    
    query_embedding = [0.88, 0.12, 0.02, 0, 0, 0]
    
    response = requests.post(
        f"{API_BASE}/search/vector",
        json={
            "query_text": "redis caching",
            "query_embedding": query_embedding,
            "top_k": 5
        }
    )
    
    data = response.json()
    print("\nResults:")
    for i, result in enumerate(data["results"], 1):
        print(f"{i}. {result['id']} (score: {result['vector_score']:.8f})")
    
    # Expected order: doc1, doc4, doc2, doc6, doc5
    expected = ["doc1", "doc4", "doc2", "doc6", "doc5"]
    actual = [r["id"] for r in data["results"]]
    
    if actual == expected:
        print("\n✓ PASS: Order matches expected")
    else:
        print(f"\n✗ FAIL: Expected {expected}, got {actual}")

def test_graph_only():
    """Test graph-only traversal from problem statement"""
    print("\n" + "="*70)
    print("TEST: Graph-Only Traversal from doc6 (depth=2)")
    print("="*70)
    
    response = requests.get(
        f"{API_BASE}/search/graph",
        params={"start_id": "doc6", "depth": 2}
    )
    
    data = response.json()
    print("\nResults:")
    for node in data["nodes"]:
        print(f"  - {node['id']} (hop={node['hop']}, edge={node['edge']}, weight={node['weight']})")
    
    # Should return doc2 (depth 1), doc1 (depth 1), doc4 (depth 2)
    node_ids = {n["id"] for n in data["nodes"]}
    expected = {"doc2", "doc1", "doc4"}
    
    if expected.issubset(node_ids):
        print("\n✓ PASS: Expected nodes found")
    else:
        print(f"\n✗ FAIL: Expected {expected}, found {node_ids}")

def test_hybrid():
    """Test hybrid search from problem statement"""
    print("\n" + "="*70)
    print("TEST: Hybrid Search (vector_weight=0.6, graph_weight=0.4)")
    print("="*70)
    
    query_embedding = [0.88, 0.12, 0.02, 0, 0, 0]
    
    response = requests.post(
        f"{API_BASE}/search/hybrid",
        json={
            "query_text": "redis caching",
            "query_embedding": query_embedding,
            "vector_weight": 0.6,
            "graph_weight": 0.4,
            "top_k": 5
        }
    )
    
    data = response.json()
    print("\nResults:")
    for result in data["results"]:
        print(f"  {result['id']}: vec={result['vector_score']:.4f}, "
              f"graph={result['graph_score']:.4f}, "
              f"final={result['final_score']:.4f}, hop={result['info']['hop']}")
    
    # doc1 should remain top due to both high vector and graph score
    if data["results"][0]["id"] == "doc1":
        print("\n✓ PASS: doc1 ranks first in hybrid mode")
    else:
        print(f"\n✗ FAIL: Expected doc1 first, got {data['results'][0]['id']}")

def main():
    print("\n" + "="*70)
    print("CANONICAL TEST EXAMPLES (Problem Statement)")
    print("="*70 + "\n")
    
    setup_canonical_data()
    test_vector_only()
    test_graph_only()
    test_hybrid()
    
    print("\n" + "="*70)
    print("CANONICAL TESTS COMPLETE")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()