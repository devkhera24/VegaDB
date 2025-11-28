from sentence_transformers import SentenceTransformer
from typing import List
import hashlib
import os

class Embeddings:
    """
    Wrapper for text → vector using SentenceTransformers.
    Default model: all-MiniLM-L6-v2 (384 dimensions).
    
    Supports mock mode for deterministic testing (hash-based, no hardcoding).
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", mock_mode: bool = None):
        self.mock_mode = mock_mode if mock_mode is not None else os.environ.get("MOCK_EMBEDDINGS", "0") == "1"
        
        if self.mock_mode:
            print("[EMBEDDINGS] Running in MOCK MODE - deterministic hash-based vectors")
            self.model = None
            self.vector_size = 6
        else:
            self.model = SentenceTransformer(model_name)
            self.vector_size = 384

    def embed(self, text: str) -> List[float]:
        """Generate embedding vector for text"""
        if self.mock_mode:
            return self._hash_based_embed(text)
        
        vec = self.model.encode(text)
        return [float(x) for x in vec]
    
    def _hash_based_embed(self, text: str) -> List[float]:
        """
        Generate deterministic embeddings using hash function.
        This works for ANY text, not just test cases.
        
        Method:
        1. Hash the text using SHA256
        2. Convert hash bytes to 6 floats in range [0, 1]
        3. Normalize to unit vector
        """
        # Create hash
        hash_bytes = hashlib.sha256(text.encode('utf-8')).digest()
        
        # Convert first 24 bytes to 6 floats (4 bytes per float)
        raw_values = []
        for i in range(6):
            # Take 4 bytes and convert to int
            byte_slice = hash_bytes[i*4:(i+1)*4]
            int_val = int.from_bytes(byte_slice, byteorder='big')
            # Normalize to [0, 1]
            float_val = int_val / (2**32 - 1)
            raw_values.append(float_val)
        
        # Normalize to unit vector (for cosine similarity)
        magnitude = sum(x**2 for x in raw_values) ** 0.5
        if magnitude > 0:
            normalized = [x / magnitude for x in raw_values]
        else:
            normalized = raw_values
        
        return normalized
    
    def get_dimension(self) -> int:
        """Return embedding dimension"""
        return self.vector_size