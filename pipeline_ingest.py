#!/usr/bin/env python3
"""
pipeline_ingest.py - Complete ingestion with structured storage
Extracts important info and stores in vector + graph DBs
"""
import uuid
import spacy
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer

# Load models
nlp = spacy.load("en_core_web_sm", disable=["parser"])
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# ==================== CHUNKING ====================
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Dict]:
    """
    Split text into overlapping chunks
    Each chunk is a dict with: {chunk_id, text, start_char, end_char}
    """
    chunks = []
    i = 0
    text_len = len(text)
    
    while i < text_len:
        end = min(i + chunk_size, text_len)
        chunk_text_content = text[i:end].strip()
        
        if chunk_text_content:
            chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text_content,
                "start_char": i,
                "end_char": end
            })
        
        i = end - overlap
        if i <= 0:
            break
    
    return chunks

# ==================== ENTITY EXTRACTION ====================
def extract_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Extract named entities (PERSON, ORG, GPE, DATE, etc.)
    Returns: {"PERSON": [...], "ORG": [...], ...}
    """
    doc = nlp(text)
    entities_by_type = {}
    
    for ent in doc.ents:
        entity_dict = {
            "text": ent.text,
            "label": ent.label_,
            "start": ent.start_char,
            "end": ent.end_char
        }
        
        if ent.label_ not in entities_by_type:
            entities_by_type[ent.label_] = []
        
        entities_by_type[ent.label_].append(entity_dict)
    
    return entities_by_type

# ==================== KEYWORD EXTRACTION ====================
def extract_keywords(text: str) -> List[str]:
    """
    Extract important keywords (nouns, proper nouns)
    """
    doc = nlp(text)
    keywords = []
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
            keywords.append(token.text.lower())
    
    # Remove duplicates, keep top 10
    keywords = list(set(keywords))[:10]
    return keywords

# ==================== EMBEDDING GENERATION ====================
def generate_embeddings(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for list of texts
    Returns: List of 384-dim vectors
    """
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    return [emb.tolist() for emb in embeddings]

# ==================== MAIN INGESTION FUNCTION ====================
def ingest_file(file_path: str) -> Dict[str, Any]:
    """
    Complete ingestion pipeline:
    1. Read file
    2. Chunk text
    3. Extract entities & keywords
    4. Generate embeddings
    5. Create structured document
    6. Store in databases
    
    Returns: Structured document JSON
    """
    # Read file
    text = Path(file_path).read_text(encoding='utf-8')
    
    # Generate document ID
    doc_id = f"doc_{uuid.uuid4().hex[:8]}"
    
    # Chunk text
    chunks = chunk_text(text)
    
    # Extract entities from full text
    entities = extract_entities(text)
    
    # Extract keywords from full text
    keywords = extract_keywords(text)
    
    # Generate embeddings for each chunk
    chunk_texts = [c["text"] for c in chunks]
    embeddings = generate_embeddings(chunk_texts)
    
    # Attach embeddings to chunks
    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding
        chunk["entities"] = extract_entities(chunk["text"])
        chunk["keywords"] = extract_keywords(chunk["text"])
    
    # Create structured document
    doc_json = {
        "doc_id": doc_id,
        "source": file_path,
        "title": Path(file_path).stem,  # filename without extension
        "created_at": datetime.utcnow().isoformat() + "Z",
        "full_text": text,
        "summary": text[:200] + "..." if len(text) > 200 else text,
        "entities": entities,
        "keywords": keywords,
        "chunk_count": len(chunks),
        "chunks": chunks
    }
    
    return doc_json

# ==================== STORAGE FUNCTIONS ====================
def store_to_api(doc_json: Dict[str, Any], api_base: str = "http://localhost:8000"):
    """
    Store document via API
    Creates one node per chunk for vector search
    """
    import requests
    
    # Store full document as a node
    response = requests.post(
        f"{api_base}/nodes",
        json={
            "id": doc_json["doc_id"],
            "text": doc_json["full_text"],
            "title": doc_json["title"],
            "metadata": {
                "source": doc_json["source"],
                "entities": doc_json["entities"],
                "keywords": doc_json["keywords"],
                "chunk_count": doc_json["chunk_count"]
            }
        }
    )
    
    if response.status_code != 200:
        print(f"Failed to create document node: {response.text}")
        return
    
    print(f"✓ Created document: {doc_json['doc_id']}")
    
    # Store each chunk as a separate node
    for i, chunk in enumerate(doc_json["chunks"]):
        chunk_response = requests.post(
            f"{api_base}/nodes",
            json={
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "title": f"{doc_json['title']} - Chunk {i+1}",
                "embedding": chunk["embedding"],
                "metadata": {
                    "doc_id": doc_json["doc_id"],
                    "chunk_index": i,
                    "entities": chunk["entities"],
                    "keywords": chunk["keywords"]
                }
            }
        )
        
        if chunk_response.status_code == 200:
            print(f"  ✓ Created chunk {i+1}/{len(doc_json['chunks'])}")
        
        # Create edge: document -> chunk
        requests.post(
            f"{api_base}/edges",
            json={
                "source": doc_json["doc_id"],
                "target": chunk["chunk_id"],
                "type": "HAS_CHUNK",
                "weight": 1.0
            }
        )

# ==================== CLI ====================
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline_ingest.py <file_path>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    print("="*70)
    print(f"  INGESTING: {file_path}")
    print("="*70)
    print()
    
    # Ingest file
    doc_json = ingest_file(file_path)
    
    # Print structured output
    print("\n📄 STRUCTURED DOCUMENT:")
    print(f"  Doc ID: {doc_json['doc_id']}")
    print(f"  Title: {doc_json['title']}")
    print(f"  Chunks: {doc_json['chunk_count']}")
    print(f"  Keywords: {', '.join(doc_json['keywords'][:5])}")
    print(f"  Entities: {sum(len(v) for v in doc_json['entities'].values())} found")
    print()
    
    # Store to API
    print("💾 STORING TO DATABASE...")
    store_to_api(doc_json)
    
    print()
    print("="*70)
    print("  ✅ INGESTION COMPLETE")
    print("="*70)
    print()
    print(f"Test queries:")
    print(f'  POST /search/vector: {{"query_text": "caching strategies", "top_k": 5}}')
    print(f'  GET  /search/graph?start_id={doc_json["doc_id"]}&depth=1')