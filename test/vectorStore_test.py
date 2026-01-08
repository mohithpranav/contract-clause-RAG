
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
sys.path.insert(0, parent_dir)
os.chdir(parent_dir)

from app.rag.loader import LegalDocumentLoader
from app.rag.splitter import LegalTextSplitter
from app.rag.embedder import LegalEmbedder
from app.rag.vector_store import LegalVectorStore
import numpy as np

# Step 1: Load documents
print("Loading documents...")
loader = LegalDocumentLoader("data/contracts")
docs = loader.load_pdfs()
print(f"Loaded {len(docs)} pages")

# Step 2: Split into chunks
print("\nSplitting documents...")
splitter = LegalTextSplitter()
chunks = splitter.split_documents(docs)
print(f"Created {len(chunks)} chunks")

# Step 3: Generate embeddings
print("\nGenerating embeddings...")
embedder = LegalEmbedder()
result = embedder.embed_documents(chunks)
embeddings = result["embeddings"]
print(f"Generated embeddings shape: {embeddings.shape}")

# Step 4: Create vector store and index
print("\nCreating FAISS index...")
store = LegalVectorStore("data/faiss_index")
store.create_index(embeddings, chunks)
print("Index created and saved!")

# Step 5: Load the index (to test loading)
print("\nLoading index...")
store.load_index()
print("Index loaded successfully!")

# Step 6: Test search with real query
print("\nTesting search...")
user_query = "Explain the terms, conversion outcome, and accounting treatment of the Convertible Notes issued under the November 9, 2022 Securities Purchase Agreement."
print(f"Query: {user_query}\n")

# Convert query to embedding
query_embedding = embedder.model.encode([user_query], convert_to_numpy=True, normalize_embeddings=True)
results = store.search(query_embedding, top_k=3)

print(f"\nTop 3 search results:")
for idx, result in enumerate(results, 1):
    print(f"\nResult {idx}:")
    print(f"  Score: {result['score']:.4f}")
    print(f"  Source: {result['metadata'].get('source', 'N/A')}")
    print(f"  Page: {result['metadata'].get('page', 'N/A')}")
    print(f"  Chunk ID: {result['metadata'].get('chunk_id', 'N/A')}")
    print(f"  Text preview: {result['text'][:150]}...")
