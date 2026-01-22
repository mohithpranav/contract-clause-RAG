from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np

class LegalEmbedder:
  

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[Dict]) -> Dict:
        texts = [doc["text"] for doc in documents]

        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return {
            "embeddings": embeddings,
            "documents": documents
        }
