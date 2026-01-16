from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np


class LegalVectorStore:
    """
    Handles FAISS index creation, storage, and loading
    for legal clause embeddings.
    """

    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)

        self.index_file = self.index_dir / "clauses.index"
        self.meta_file = self.index_dir / "metadata.npy"

        self.index = None
        self.metadata = None

    def create_index(self, embeddings: np.ndarray, documents: List[Dict]):
        """
        Creates and saves a FAISS index from embeddings.
        Replaces any existing index.
        """
        dimension = embeddings.shape[1]

        # Cosine similarity via inner product (vectors already normalized)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        # Save index
        faiss.write_index(self.index, str(self.index_file))

        # Save metadata separately
        self.metadata = documents
        np.save(self.meta_file, self.metadata, allow_pickle=True)
    
    def clear_index(self):
        """
        Deletes existing index and metadata files.
        Called before creating new index to ensure clean slate.
        """
        if self.index_file.exists():
            self.index_file.unlink()
            print(f"   ✓ Deleted old index file: {self.index_file.name}", flush=True)
        
        if self.meta_file.exists():
            self.meta_file.unlink()
            print(f"   ✓ Deleted old metadata file: {self.meta_file.name}", flush=True)
        
        self.index = None
        self.metadata = None

    def load_index(self):
        """
        Loads FAISS index and metadata from disk.
        """
        if not self.index_file.exists() or not self.meta_file.exists():
            raise FileNotFoundError("FAISS index or metadata not found")

        self.index = faiss.read_index(str(self.index_file))
        self.metadata = np.load(self.meta_file, allow_pickle=True).tolist()

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Searches the FAISS index for similar clauses.
        """
        if self.index is None or self.metadata is None:
            raise RuntimeError("Index not loaded")

        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            results.append({
                "score": float(score),
                "text": self.metadata[idx]["text"],
                "metadata": self.metadata[idx]["metadata"]
            })

        return results
