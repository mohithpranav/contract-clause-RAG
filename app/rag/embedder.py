from typing import List, Dict

from sentence_transformers import SentenceTransformer
import numpy as np

class LegalEmbedder:
    """
    Creates embeddings for legal text chunks
    using Sentence Transformers.
    """

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5"):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, documents: List[Dict]) -> Dict:
        """
        Generates embeddings for chunked legal documents.

        Input:
            documents: Output from LegalTextSplitter

        Output:
            {
                "embeddings": np.ndarray,
                "documents": original documents with metadata
            }
        """
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
