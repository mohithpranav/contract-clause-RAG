from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class LegalTextSplitter:

    def __init__(
        self,
        chunk_size: int = 400,  # Reduced for better precision (200-400 tokens optimal)
        chunk_overlap: int = 50  # 40-60 tokens overlap recommended
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n## ",      # Section headers (markdown-style)
                "\n# ",       # Main headers
                "\n\n",       # Paragraph/clause breaks (most important)
                "\n",         # Line breaks
                ". ",         # Sentence end with space
                " ",          # Word boundary
                ""
            ],
            length_function=len,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        chunked_docs = []

        for doc in documents:
            text_chunks = self.splitter.split_text(doc["text"])

            for idx, chunk in enumerate(text_chunks):
                chunked_docs.append({
                    "text": chunk,
                    "metadata": {
                        **doc["metadata"],
                        "chunk_id": idx
                    }
                })

        return chunked_docs
