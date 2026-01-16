"""
MongoDB Schema Models for Query Logging
"""
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional, Dict, Any


class QueryMetadata(BaseModel):
    """Metadata extracted from query response"""
    clause_title: str = ""
    confidence: int = 0
    relevance_score: float = 0.0


class QueryLog(BaseModel):
    """Schema for storing user queries and responses in MongoDB"""
    user_id: str = "default_user"
    query: str
    response: Dict[str, Any]
    query_type: str = "query"  # "query" or "analysis"
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: QueryMetadata
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "default_user",
                "query": "What is the termination clause?",
                "response": {
                    "clause": {"title": "Termination", "content": "..."},
                    "explanation": {"summary": "...", "confidence": 90},
                    "relevance": {"score": 0.85}
                },
                "query_type": "query",
                "timestamp": "2026-01-16T10:30:00Z",
                "metadata": {
                    "clause_title": "Termination",
                    "confidence": 90,
                    "relevance_score": 0.85
                }
            }
        }
