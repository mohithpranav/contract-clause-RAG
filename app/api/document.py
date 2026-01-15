"""
Document processing endpoint for ClauseInsight
Handles PDF upload, chunking, embedding, and indexing
"""
from pathlib import Path
from typing import Dict

from app.rag.loader import LegalDocumentLoader
from app.rag.splitter import LegalTextSplitter
from app.rag.embedder import LegalEmbedder
from app.rag.vector_store import LegalVectorStore


async def process_and_index_document(pdf_path: str, index_dir: str) -> Dict:
    """
    Process a legal PDF and add it to the FAISS index
    
    Args:
        pdf_path: Path to the PDF file
        index_dir: Directory for FAISS index
        
    Returns:
        Dictionary with processing results
    """
    # Create a temporary directory structure for this PDF
    pdf_file = Path(pdf_path)
    temp_contracts_dir = pdf_file.parent
    
    # Step 1: Load PDF
    loader = LegalDocumentLoader(str(temp_contracts_dir))
    documents = loader.load_pdfs()
    
    # Step 2: Split into chunks (optimized settings from RAG improvements)
    splitter = LegalTextSplitter(chunk_size=400, chunk_overlap=50)
    chunked_docs = splitter.split_documents(documents)
    
    # Step 3: Generate embeddings
    embedder = LegalEmbedder(model_name="BAAI/bge-large-en-v1.5")
    embedded_data = embedder.embed_documents(chunked_docs)
    
    # Step 4: Create/Update FAISS index
    vector_store = LegalVectorStore(index_dir)
    
    # Check if index exists and load it, or create new
    try:
        vector_store.load_index()
        # Index exists, we would need to append (for now, we'll recreate)
        # TODO: Implement incremental indexing
        pass
    except FileNotFoundError:
        pass
    
    # Create new index (overwrites existing)
    vector_store.create_index(
        embeddings=embedded_data["embeddings"],
        documents=embedded_data["documents"]
    )
    
    return {
        "status": "success",
        "chunks_created": len(chunked_docs),
        "pdf_file": pdf_file.name
    }
