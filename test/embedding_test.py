import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from app.rag.embedder import LegalEmbedder
from app.rag.loader import LegalDocumentLoader
from app.rag.splitter import LegalTextSplitter

loader = LegalDocumentLoader("data/contracts")
docs = loader.load_pdfs()

splitter = LegalTextSplitter()
chunks = splitter.split_documents(docs)

embedder = LegalEmbedder()
result = embedder.embed_documents(chunks)

print(result["embeddings"].shape)
