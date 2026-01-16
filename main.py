"""
FastAPI Backend for ClauseInsight - Legal RAG System
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from pathlib import Path
import shutil
from typing import Optional
from dotenv import load_dotenv
from motor.motor_asyncio import AsyncIOMotorClient
from datetime import datetime
import os

# Load environment variables
load_dotenv()

from app.api.query import query_clauses
from app.api.document import process_and_index_document
from app.api.analysis import analyze_clause
from app.api.document_analysis import analyze_entire_document
from app.models.query_log import QueryLog, QueryMetadata

# MongoDB Configuration
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
DATABASE_NAME = "clause_insight"
COLLECTION_NAME = "query_history"

# MongoDB client (will be initialized on startup)
mongo_client: Optional[AsyncIOMotorClient] = None
db = None
query_collection = None

# Initialize FastAPI app
app = FastAPI(
    title="ClauseInsight API",
    description="LegalTech RAG system for contract clause retrieval and analysis",
    version="1.0.0"
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://localhost:8081", "http://localhost:8080"],  # Vite dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
CONTRACTS_DIR = Path(__file__).parent / "data" / "contracts"
FAISS_INDEX_DIR = Path(__file__).parent / "data" / "faiss_index"

# Ensure directories exist
CONTRACTS_DIR.mkdir(parents=True, exist_ok=True)
FAISS_INDEX_DIR.mkdir(parents=True, exist_ok=True)


@app.on_event("startup")
async def startup_db_client():
    """Initialize MongoDB connection on startup"""
    global mongo_client, db, query_collection
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL)
        db = mongo_client[DATABASE_NAME]
        query_collection = db[COLLECTION_NAME]
        # Test connection
        await mongo_client.admin.command('ping')
        print(f"✅ Connected to MongoDB at {MONGODB_URL}")
        print(f"📊 Using database: {DATABASE_NAME}, collection: {COLLECTION_NAME}")
    except Exception as e:
        print(f"⚠️ MongoDB connection failed: {e}")
        print("⚠️ Continuing without query logging...")
        mongo_client = None


@app.on_event("shutdown")
async def shutdown_db_client():
    """Close MongoDB connection on shutdown"""
    global mongo_client
    if mongo_client:
        mongo_client.close()
        print("🔌 MongoDB connection closed")


async def log_query_to_db(query: str, response: dict, query_type: str = "query"):
    """Log query and response to MongoDB using Pydantic schema"""
    if query_collection is None:
        return  # Skip if MongoDB not connected
    
    try:
        # Create metadata
        metadata = QueryMetadata(
            clause_title=response.get("clause", {}).get("title", ""),
            confidence=response.get("explanation", {}).get("confidence", 0),
            relevance_score=response.get("relevance", {}).get("score", 0.0) / 100.0 if isinstance(response.get("relevance", {}).get("score"), int) else response.get("relevance", {}).get("score", 0.0)
        )
        
        # Create query log document
        query_log = QueryLog(
            user_id="default_user",
            query=query,
            response=response,
            query_type=query_type,
            metadata=metadata
        )
        
        # Convert to dict and insert into MongoDB
        document = query_log.model_dump(by_alias=True, mode='json')
        await query_collection.insert_one(document)
        print(f"💾 Logged query to MongoDB: '{query[:50]}...'")
    except Exception as e:
        print(f"⚠️ Failed to log query to MongoDB: {e}")
        import traceback
        traceback.print_exc()


class QueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = 3


class AnalysisRequest(BaseModel):
    clause_text: str
    metadata: dict


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "operational",
        "service": "ClauseInsight API",
        "version": "1.0.0"
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and process a legal contract PDF
    """
    # Validate file type
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    try:
        # Save uploaded file
        file_path = CONTRACTS_DIR / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process and index the document
        result = await process_and_index_document(
            str(file_path),
            str(FAISS_INDEX_DIR)
        )
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' processed successfully",
            "chunks_created": result["chunks_created"],
            "filename": file.filename
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/query")
async def query_document(request: QueryRequest):
    """
    Query the indexed documents and get relevant clauses with LLM-generated
    simple explanation
    """
    try:
        top_k = request.top_k if request.top_k is not None else 3
        result = await query_clauses(
            query=request.query,
            index_dir=str(FAISS_INDEX_DIR),
            top_k=top_k
        )
        
        # Log query and response to MongoDB (non-blocking)
        try:
            await log_query_to_db(
                query=request.query,
                response=result,
                query_type="query"
            )
        except Exception as log_error:
            print(f"⚠️ MongoDB logging failed (continuing): {log_error}")
            import traceback
            traceback.print_exc()
        
        return result
    
    except Exception as e:
        print(f"❌ Query endpoint error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/analyze")
async def analyze_clause_detail(request: AnalysisRequest):
    """
    Get detailed analysis of a specific clause (independent of query)
    """
    try:
        result = await analyze_clause(
            clause_text=request.clause_text,
            metadata=request.metadata
        )
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing clause: {str(e)}")


@app.get("/analyze-document")
async def analyze_full_document():
    """
    Get comprehensive analysis of the entire uploaded document
    """
    try:
        result = await analyze_entire_document(
            index_dir=str(FAISS_INDEX_DIR)
        )
        
        return result
    
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing document: {str(e)}")


@app.get("/health")
async def health_check():
    """Extended health check with system status"""
    index_exists = (FAISS_INDEX_DIR / "clauses.index").exists()
    metadata_exists = (FAISS_INDEX_DIR / "metadata.npy").exists()
    
    return {
        "status": "healthy",
        "index_ready": index_exists and metadata_exists,
        "contracts_dir": str(CONTRACTS_DIR),
        "indexed_documents": len(list(CONTRACTS_DIR.glob("*.pdf")))
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
