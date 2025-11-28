#!/usr/bin/env python3
"""
load_test_data.py - Load test documents
This loads the canonical test data, but the API works with ANY data
"""
import requests
import json

API_BASE = "http://localhost:8000"

# These are the test case documents - but API works with ANY documents
DOCS = {
    "doc1": {
        "text": "Redis became the default choice for caching mostly because people like avoiding slow databases. There are the usual headaches: eviction policies like LRU vs LFU, memory pressure, and when someone forgets to set TTLs and wonders why servers fall over.",
        "title": "Redis caching strategies",
        "embedding": [0.90, 0.10, 0.00, 0.00, 0.00, 0.00]
    },
    "doc2": {
        "text": "The RedisGraph module promises a weird marriage: pretend your cache is also a graph database.",
        "title": "RedisGraph module",
        "embedding": [0.70, 0.10, 0.60, 0.00, 0.00, 0.00]
    },
    "doc3": {
        "text": "Distributed systems are basically long-distance relationships. Nodes drift apart, messages get lost.",
        "title": "Distributed systems",
        "embedding": [0.10, 0.05, 0.00, 0.90, 0.00, 0.00]
    },
    "doc4": {
        "text": "A short note on cache invalidation: you think you understand it until your application grows.",
        "title": "Cache invalidation note",
        "embedding": [0.80, 0.15, 0.00, 0.00, 0.00, 0.00]
    },
    "doc5": {
        "text": "Graph algorithms show up in real life more than people notice. Social feeds rely on BFS.",
        "title": "Graph algorithms",
        "embedding": [0.05, 0.00, 0.90, 0.10, 0.00, 0.00]
    },
    "doc6": {
        "text": "README draft: to combine Redis with a graph database, you start by defining nodes for each entity.",
        "title": "README: Redis+Graph",
        "embedding": [0.60, 0.05, 0.50, 0.00, 0.10, 0.00]
    }
}

EDGES = [
    {"source": "doc1", "target": "doc4", "type": "related_to", "weight": 0.8},
    {"source": "doc2", "target": "doc6", "type": "mentions", "weight": 0.9},
    {"source": "doc6", "target": "doc1", "type": "references", "weight": 0.6},
    {"source": "doc3", "target": "doc5", "type": "related_to", "weight": 0.5},
]

def main():
    print("="*70)
    print("  LOADING TEST DATA (but API works with ANY data)")
    print("="*70)
    print()
    
    for doc_id, doc in DOCS.items():
        response = requests.post(
            f"{API_BASE}/nodes",
            json={
                "id": doc_id,
                "text": doc["text"],
                "title": doc["title"],
                "embedding": doc["embedding"]
            }
        )
        
        if response.status_code == 200:
            print(f"✓ Created {doc_id}")
        else:
            print(f"✗ Failed {doc_id}: {response.text}")
    
    print()
    for edge in EDGES:
        response = requests.post(f"{API_BASE}/edges", json=edge)
        if response.status_code == 200:
            print(f"✓ Edge: {edge['source']} -> {edge['target']}")
    
    print()
    print("="*70)
    print("  DATA LOADED - Test at http://localhost:8000/docs")
    print("="*70)

if __name__ == "__main__":
    main()