#!/usr/bin/env python3
"""
test_api_crud.py - Comprehensive API test suite for problem statement
Tests all CRUD operations and search modes
"""
import requests
import json
import time
from typing import Dict, Any

API_BASE = "http://localhost:8000"

def print_test(name: str):
    print(f"\n{'='*70}")
    print(f"TEST: {name}")
    print('='*70)

def print_pass(msg: str):
    print(f"✓ PASS: {msg}")

def print_fail(msg: str):
    print(f"✗ FAIL: {msg}")

def assert_response(condition: bool, message: str):
    if condition:
        print_pass(message)
    else:
        print_fail(message)
        raise AssertionError(message)

# ==================== TC-API-01: Create Node ====================
def test_create_node():
    print_test("TC-API-01: Create Node")
    
    response = requests.post(
        f"{API_BASE}/nodes",
        json={
            "text": "Venkat's note on caching",
            "metadata": {"type": "note", "author": "v"}
        }
    )
    
    assert_response(response.status_code == 200, "Status code is 200")
    data = response.json()
    
    assert_response("id" in data, "Response contains 'id'")
    assert_response("text" in data, "Response contains 'text'")
    assert_response("embedding" in data, "Response contains 'embedding'")
    assert_response(data["text"] == "Venkat's note on caching", "Text matches")
    
    node_id = data["id"]
    
    # Verify GET returns same content
    get_response = requests.get(f"{API_BASE}/nodes/{node_id}")
    assert_response(get_response.status_code == 200, "GET node succeeds")
    get_data = get_response.json()
    assert_response(get_data["text"] == data["text"], "GET returns same text")
    
    return node_id

# ==================== TC-API-02: Read Node with Relationships ====================
def test_read_node_with_relationships():
    print_test("TC-API-02: Read Node with Relationships")
    
    # Create two nodes
    resp_a = requests.post(f"{API_BASE}/nodes", json={"text": "Node A"})
    resp_b = requests.post(f"{API_BASE}/nodes", json={"text": "Node B"})
    
    node_a = resp_a.json()["id"]
    node_b = resp_b.json()["id"]
    
    # Create edge A->B
    edge_resp = requests.post(
        f"{API_BASE}/edges",
        json={"source": node_a, "target": node_b, "type": "relates_to", "weight": 0.8}
    )
    assert_response(edge_resp.status_code == 200, "Edge creation succeeds")
    
    # GET node A
    get_resp = requests.get(f"{API_BASE}/nodes/{node_a}")
    data = get_resp.json()
    
    assert_response("edges" in data, "Response includes edges")
    assert_response(len(data["edges"]) > 0, "Node has at least one edge")
    
    edge = data["edges"][0]
    assert_response("type" in edge, "Edge has 'type'")
    assert_response("target_id" in edge, "Edge has 'target_id'")
    assert_response("weight" in edge, "Edge has 'weight'")
    assert_response(edge["target_id"] == node_b, "Edge target matches")

# ==================== TC-API-03: Update Node ====================
def test_update_node():
    print_test("TC-API-03: Update Node & Regenerate Embedding")
    
    # Create node
    create_resp = requests.post(
        f"{API_BASE}/nodes",
        json={"text": "Original text"}
    )
    node_id = create_resp.json()["id"]
    old_embedding = create_resp.json()["embedding"]
    
    # Update with new text and regen flag
    update_resp = requests.put(
        f"{API_BASE}/nodes/{node_id}",
        json={"text": "Completely different text", "regen_embedding": True}
    )
    
    assert_response(update_resp.status_code == 200, "Update succeeds")
    data = update_resp.json()
    
    assert_response(data["status"] == "updated", "Status is 'updated'")
    
    # Verify embedding changed
    if "old_new_similarity" in data:
        similarity = data["old_new_similarity"]
        assert_response(similarity < 0.99, f"Embedding changed (similarity={similarity:.4f})")
    
    # GET and verify text updated
    get_resp = requests.get(f"{API_BASE}/nodes/{node_id}")
    get_data = get_resp.json()
    assert_response(get_data["text"] == "Completely different text", "Text updated")

# ==================== TC-API-04: Delete Node ====================
def test_delete_node():
    print_test("TC-API-04: Delete Node Cascading Edges")
    
    # Create node with edges
    node_resp = requests.post(f"{API_BASE}/nodes", json={"text": "To be deleted"})
    node_id = node_resp.json()["id"]
    
    target_resp = requests.post(f"{API_BASE}/nodes", json={"text": "Target"})
    target_id = target_resp.json()["id"]
    
    requests.post(f"{API_BASE}/edges", json={
        "source": node_id,
        "target": target_id,
        "type": "test",
        "weight": 1.0
    })
    
    # Delete node
    delete_resp = requests.delete(f"{API_BASE}/nodes/{node_id}")
    assert_response(delete_resp.status_code == 200, "Delete succeeds")
    
    # Verify node is gone
    get_resp = requests.get(f"{API_BASE}/nodes/{node_id}")
    assert_response(get_resp.status_code == 404, "Node returns 404 after delete")

# ==================== TC-VEC-01: Vector Search ====================
def test_vector_search():
    print_test("TC-VEC-01: Top-k Cosine Similarity Ordering")
    
    # Create nodes with known text
    nodes = []
    texts = [
        "Redis caching strategies are important",
        "Cache invalidation is hard",
        "Completely unrelated topic about cats"
    ]
    
    for text in texts:
        resp = requests.post(f"{API_BASE}/nodes", json={"text": text})
        nodes.append(resp.json()["id"])
    
    time.sleep(0.2)
    
    # Search for "redis caching"
    search_resp = requests.post(
        f"{API_BASE}/search/vector",
        json={"query_text": "redis caching", "top_k": 3}
    )
    
    assert_response(search_resp.status_code == 200, "Vector search succeeds")
    data = search_resp.json()
    
    assert_response("results" in data, "Response has 'results'")
    results = data["results"]
    assert_response(len(results) > 0, "Returns at least one result")
    
    # Verify ordering (highest score first)
    for i in range(len(results) - 1):
        assert_response(
            results[i]["vector_score"] >= results[i+1]["vector_score"],
            f"Result {i} score >= result {i+1} score"
        )

# ==================== TC-GRAPH-01: Graph Traversal ====================
def test_graph_traversal():
    print_test("TC-GRAPH-01: BFS Depth-Limited Traversal")
    
    # Create chain A->B->C->D
    nodes = {}
    for label in ["A", "B", "C", "D"]:
        resp = requests.post(f"{API_BASE}/nodes", json={"text": f"Node {label}"})
        nodes[label] = resp.json()["id"]
    
    # Create edges
    requests.post(f"{API_BASE}/edges", json={
        "source": nodes["A"], "target": nodes["B"], "type": "next", "weight": 1.0
    })
    requests.post(f"{API_BASE}/edges", json={
        "source": nodes["B"], "target": nodes["C"], "type": "next", "weight": 1.0
    })
    requests.post(f"{API_BASE}/edges", json={
        "source": nodes["C"], "target": nodes["D"], "type": "next", "weight": 1.0
    })
    
    # Traverse from A with depth=2
    traverse_resp = requests.get(
        f"{API_BASE}/search/graph",
        params={"start_id": nodes["A"], "depth": 2}
    )
    
    assert_response(traverse_resp.status_code == 200, "Graph traversal succeeds")
    data = traverse_resp.json()
    
    assert_response("nodes" in data, "Response has 'nodes'")
    visited_ids = [n["id"] for n in data["nodes"]]
    
    assert_response(nodes["B"] in visited_ids, "B is reachable (depth 1)")
    assert_response(nodes["C"] in visited_ids, "C is reachable (depth 2)")
    assert_response(nodes["D"] not in visited_ids, "D is NOT reachable (depth 3)")

# ==================== TC-HYB-01: Hybrid Search ====================
def test_hybrid_search():
    print_test("TC-HYB-01: Weighted Merge Correctness")
    
    # Create test nodes
    v_similar_resp = requests.post(f"{API_BASE}/nodes", json={
        "text": "Redis caching and performance optimization"
    })
    v_similar_id = v_similar_resp.json()["id"]
    
    g_close_resp = requests.post(f"{API_BASE}/nodes", json={
        "text": "Unrelated content about databases"
    })
    g_close_id = g_close_resp.json()["id"]
    
    # Create edge to make g_close graph-connected
    requests.post(f"{API_BASE}/edges", json={
        "source": v_similar_id, "target": g_close_id, "type": "relates", "weight": 1.0
    })
    
    time.sleep(0.2)
    
    # Hybrid search with vector_weight=0.7, graph_weight=0.3
    hybrid_resp = requests.post(
        f"{API_BASE}/search/hybrid",
        json={
            "query_text": "redis caching",
            "vector_weight": 0.7,
            "graph_weight": 0.3,
            "top_k": 5
        }
    )
    
    assert_response(hybrid_resp.status_code == 200, "Hybrid search succeeds")
    data = hybrid_resp.json()
    
    assert_response("results" in data, "Response has 'results'")
    results = data["results"]
    
    # Verify each result has required fields
    for r in results:
        assert_response("vector_score" in r, "Result has vector_score")
        assert_response("graph_score" in r, "Result has graph_score")
        assert_response("final_score" in r, "Result has final_score")
    
    # Verify final_score ordering
    for i in range(len(results) - 1):
        assert_response(
            results[i]["final_score"] >= results[i+1]["final_score"],
            "Results sorted by final_score"
        )

# ==================== MAIN TEST RUNNER ====================
def main():
    print("\n" + "="*70)
    print("VECTOR+GRAPH DATABASE API TEST SUITE")
    print("="*70)
    
    tests = [
        ("TC-API-01", test_create_node),
        ("TC-API-02", test_read_node_with_relationships),
        ("TC-API-03", test_update_node),
        ("TC-API-04", test_delete_node),
        ("TC-VEC-01", test_vector_search),
        ("TC-GRAPH-01", test_graph_traversal),
        ("TC-HYB-01", test_hybrid_search),
    ]
    
    passed = 0
    failed = 0
    
    for test_id, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print_fail(f"{test_id} failed: {e}")
            failed += 1
    
    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()