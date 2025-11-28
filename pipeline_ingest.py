#!/usr/bin/env python3
"""
enhanced_ingest.py - Full NLP pipeline with entity extraction
Extracts: PERSON, ORG, GPE, DATE, CARDINAL, ORDINAL, etc.
"""
import sys
import json
import requests
import spacy
from typing import Dict, Any, List
from collections import defaultdict

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
    print("✓ spaCy model loaded")
except:
    print("✗ spaCy model not found. Installing...")
    import subprocess
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

API_BASE = "http://localhost:8000"

def extract_entities(text: str) -> Dict[str, List[Dict]]:
    """
    Extract named entities from text
    Returns: {
        "PERSON": [...],
        "ORG": [...],
        "GPE": [...],
        "DATE": [...],
        "CARDINAL": [...],
        "ORDINAL": [...],
        ...
    }
    """
    doc = nlp(text)
    entities_by_type = defaultdict(list)
    
    for ent in doc.ents:
        entity_dict = {
            "text": ent.text,
            "label": ent.label_,
            "start_char": ent.start_char,
            "end_char": ent.end_char
        }
        entities_by_type[ent.label_].append(entity_dict)
    
    return dict(entities_by_type)

def extract_keywords(text: str) -> List[str]:
    """Extract important keywords (nouns, proper nouns)"""
    doc = nlp(text)
    keywords = []
    
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 3:
            if not token.is_stop:
                keywords.append(token.text.lower())
    
    # Remove duplicates, keep top 15
    keywords = list(dict.fromkeys(keywords))[:15]
    return keywords

def extract_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents]

def analyze_text(text: str) -> Dict[str, Any]:
    """
    Complete NLP analysis
    Returns structured metadata with all extractions
    """
    # Extract entities
    entities = extract_entities(text)
    
    # Extract keywords
    keywords = extract_keywords(text)
    
    # Extract sentences
    sentences = extract_sentences(text)
    
    # Count statistics
    doc = nlp(text)
    stats = {
        "char_count": len(text),
        "word_count": len([t for t in doc if not t.is_space]),
        "sentence_count": len(sentences),
        "entity_count": sum(len(v) for v in entities.values()),
        "unique_entity_types": len(entities.keys())
    }
    
    return {
        "entities": entities,
        "keywords": keywords,
        "sentences": sentences[:5],  # First 5 sentences as preview
        "stats": stats
    }

def create_node_with_analysis(text: str, title: str = None) -> Dict[str, Any]:
    """
    Create node with full NLP analysis
    Returns complete response with embeddings and metadata
    """
    print("\n" + "=" * 70)
    print("  ANALYZING TEXT...")
    print("=" * 70)
    
    # Analyze text
    analysis = analyze_text(text)
    
    # Print analysis
    print("\n📊 EXTRACTION RESULTS:")
    print("-" * 70)
    
    if analysis["entities"]:
        print("\n🏷️  ENTITIES:")
        for entity_type, entities in sorted(analysis["entities"].items()):
            print(f"\n  {entity_type}:")
            for ent in entities[:5]:  # Show first 5 of each type
                print(f"    • {ent['text']}")
            if len(entities) > 5:
                print(f"    ... and {len(entities) - 5} more")
    else:
        print("\n🏷️  No entities found")
    
    if analysis["keywords"]:
        print(f"\n🔑 KEYWORDS:")
        print(f"  {', '.join(analysis['keywords'][:10])}")
    
    print(f"\n📈 STATISTICS:")
    for key, value in analysis["stats"].items():
        print(f"  {key}: {value}")
    
    # Create node via API
    print("\n" + "-" * 70)
    print("  CREATING NODE IN DATABASE...")
    print("-" * 70)
    
    try:
        response = requests.post(
            f"{API_BASE}/nodes",
            json={
                "text": text,
                "title": title or "Document",
                "metadata": {
                    "entities": analysis["entities"],
                    "keywords": analysis["keywords"],
                    "stats": analysis["stats"],
                    "source": "enhanced_ingest"
                }
            }
        )
        response.raise_for_status()
        result = response.json()
        
        print("\n✓ NODE CREATED SUCCESSFULLY")
        print(f"  Node ID: {result['id']}")
        print(f"  Embedding dimension: {len(result['embedding'])}")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Failed to create node: {e}")
        return None

def read_multiline_input():
    """Read text from user (multi-line)"""
    print("\n" + "=" * 70)
    print("  PASTE YOUR TEXT")
    print("=" * 70)
    print("Press Enter on empty line to finish, or type 'END'\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == "END" or (not line.strip() and lines):
                break
            if line.strip():
                lines.append(line)
        except EOFError:
            break
    
    return "\n".join(lines).strip()

def main():
    print("\n" + "=" * 70)
    print("  ENHANCED NLP INGESTION PIPELINE")
    print("  Extracts: PERSON, ORG, GPE, DATE, CARDINAL, ORDINAL, etc.")
    print("=" * 70)
    
    # Get input
    if len(sys.argv) > 1:
        # Read from file
        file_path = sys.argv[1]
        print(f"\nReading from file: {file_path}")
        with open(file_path, 'r') as f:
            text = f.read()
        title = file_path
    else:
        # Interactive input
        text = read_multiline_input()
        title = "CLI Document"
    
    if not text:
        print("\n✗ No text provided")
        sys.exit(1)
    
    # Process and create node
    result = create_node_with_analysis(text, title)
    
    if result:
        # Save detailed output
        output_file = "ingestion_output.json"
        with open(output_file, 'w') as f:
            json.dump(result, indent=2, fp=f)
        
        print("\n" + "=" * 70)
        print("  ✅ COMPLETE!")
        print("=" * 70)
        print(f"\n📄 Full output saved to: {output_file}")
        print(f"\n🔍 View in API docs: http://localhost:8000/docs")
        print(f"   GET /nodes/{result['id']}")
        
        # Show sample of JSON output
        print("\n" + "-" * 70)
        print("  SAMPLE OUTPUT (metadata.entities):")
        print("-" * 70)
        if result.get('metadata', {}).get('entities'):
            entities_preview = {}
            for k, v in result['metadata']['entities'].items():
                entities_preview[k] = v[:2]  # Show first 2 of each
            print(json.dumps(entities_preview, indent=2))

if __name__ == "__main__":
    main()