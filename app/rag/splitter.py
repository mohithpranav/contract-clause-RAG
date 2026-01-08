from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter


class LegalTextSplitter:
    """
    Splits legal text into semantically meaningful chunks
    suitable for embedding and retrieval.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",      # Clause breaks
                "\n",        # Line breaks
                ".",         # Sentence end
                " ",         # Word boundary
                ""
            ]
        )

    def split_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Splits loaded legal documents into chunks.

        Input:
            documents -> output from LegalDocumentLoader

        Output:
            List of chunked documents with metadata                 
        """
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
