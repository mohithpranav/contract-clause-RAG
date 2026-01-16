"""
Query endpoint for ClauseInsight
Handles semantic search and AI-powered clause explanation
"""
from typing import Dict, List
import re
from app.rag.embedder import LegalEmbedder
from app.rag.vector_store import LegalVectorStore
from app.rag.generator import LegalResponseGenerator


def calculate_answer_likelihood(chunk_text: str, query: str) -> float:
    """
    Question-aware scoring to boost chunks likely to contain answers.
    Returns bonus score (0-1) based on query type and chunk content.
    """
    score = 0.0
    query_lower = query.lower()
    chunk_lower = chunk_text.lower()
    
    # Definition questions (FIX 3)
    if any(word in query_lower for word in ['define', 'defined', 'definition', 'what is', 'what constitutes']):
        if any(word in chunk_lower for word in ['represents', 'means', 'constitutes', 'is defined as', 'refers to']):
            score += 0.3
    
    # Argument/perspective questions
    if any(word in query_lower for word in ['argument', 'argued', 'claimed', 'contended', 'defense']):
        if 'company' in query_lower and 'company' in chunk_lower:
            score += 0.2
        if 'commission' in query_lower and 'commission' in chunk_lower:
            score += 0.2
    
    # Timeline/event questions
    if any(word in query_lower for word in ['visit', 'during', 'when', 'actions', 'first', 'second']):
        # Boost chunks with dates or visit markers
        if any(word in chunk_lower for word in ['visit', 'date:', 'may', 'june', 'first', 'second']):
            score += 0.25
    
    # How/why questions prefer detailed explanations
    if query_lower.startswith('how') or query_lower.startswith('why'):
        if len(chunk_text) > 500:  # Longer chunks often have better explanations
            score += 0.1
    
    # Summary questions prefer overview sections
    if any(word in query_lower for word in ['summary', 'about', 'overview', 'case']):
        if any(word in chunk_lower for word in ['overview', 'introduction', 'summary', 'concern']):
            score += 0.2
    
    # Boost if query keywords appear in chunk
    query_keywords = [word for word in query_lower.split() if len(word) > 4]
    keyword_matches = sum(1 for kw in query_keywords if kw in chunk_lower)
    score += min(0.3, keyword_matches * 0.1)
    
    return score


def rerank_by_answerability(results: List[Dict], query: str) -> List[Dict]:
    """
    Rerank search results by combining embedding similarity + answer likelihood.
    """
    reranked = []
    for result in results:
        # Original embedding score
        embedding_score = result['score']
        
        # Answer likelihood bonus
        answer_bonus = calculate_answer_likelihood(result['text'], query)
        
        # Combined score (weighted)
        final_score = (embedding_score * 0.7) + (answer_bonus * 0.3)
        
        reranked.append({
            **result,
            'original_score': embedding_score,
            'answer_bonus': answer_bonus,
            'score': final_score
        })
    
    # Sort by final score descending
    reranked.sort(key=lambda x: x['score'], reverse=True)
    
    print("\n🔄 Reranking results:")
    for i, r in enumerate(reranked[:3]):
        print(f"  Rank {i+1}: Score {r['score']:.4f} (embedding: {r['original_score']:.4f} + answer: {r['answer_bonus']:.2f})")
    
    return reranked


def is_empty_answer(explanation: Dict) -> bool:
    """
    Detect if LLM returned an empty/non-answer.
    """
    meaning = explanation.get('meaning', '').lower()
    
    # Check for common empty answer phrases
    empty_phrases = [
        'does not contain',
        'not contain information',
        'context does not',
        'does not address',
        'not specified',
        'not mentioned',
        'no information',
        '[topic]',  # Template placeholder leaked
        '[specific aspect'  # Template placeholder
    ]
    
    # Empty if contains any phrase AND is short (< 150 chars)
    has_empty_phrase = any(phrase in meaning for phrase in empty_phrases)
    is_short = len(meaning) < 150
    
    return has_empty_phrase and is_short


async def query_clauses(query: str, index_dir: str, top_k: int = 3) -> Dict:
    """
    Query the FAISS index and generate a structured response matching frontend expectations
    
    Args:
        query: User's natural language question
        index_dir: Directory containing FAISS index
        top_k: Number of top results to retrieve
        
    Returns:
        Structured response with clause, explanation, and relevance data
    """
    # Step 1: Load vector store
    vector_store = LegalVectorStore(index_dir)
    
    try:
        vector_store.load_index()
        print(f"\n{'='*80}")
        print(f"QUERY DEBUG: '{query}'")
        print(f"Total documents in index: {len(vector_store.metadata)}")
    except FileNotFoundError:
        return {
            "error": "No documents have been indexed yet. Please upload a contract first.",
            "clause": None,
            "explanation": None,
            "relevance": None
        }
    
    # Step 2: Embed query
    embedder = LegalEmbedder(model_name="BAAI/bge-large-en-v1.5")
    query_embedding = embedder.model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    print(f"Query embedding shape: {query_embedding.shape}")
    
    # Step 3: Search FAISS index
    search_results = vector_store.search(query_embedding, top_k=top_k)
    print(f"Search returned {len(search_results)} results")
    for i, result in enumerate(search_results[:3]):
        print(f"\nResult {i+1}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  Text preview: {result['text'][:100]}...")
    print(f"{'='*80}\n")
    
    if not search_results:
        return {
            "error": "No relevant clauses found for your query.",
            "clause": None,
            "explanation": None,
            "relevance": None
        }
    
    # Check if top result has sufficient relevance (threshold: 0.5 or 50%)
    top_score = search_results[0]['score']
    if top_score < 0.5:
        print(f"⚠️ Low relevance score ({top_score:.2f} < 0.50) - Query appears unrelated to document")
        return {
            "clause": {
                "title": "Information Not Available",
                "section": "N/A",
                "content": ""
            },
            "explanation": {
                "summary": "I don't have information about that in this document.",
                "meaning": "I don't have information about that in this document.",
                "favoredParty": "N/A",
                "keyTerms": [],
                "practicalImpact": "",
                "confidence": 30,
                "confidenceReason": "Query does not match document content"
            },
            "relevance": {
                "score": int(top_score * 100),
                "matchedTerms": []
            }
        }
    
    # Step 4: Question-aware reranking (FIX 2)
    reranked_results = rerank_by_answerability(search_results[:top_k], query)
    
    # Step 5: Multi-chunk aggregation (FIX 1) - Pass top-3 chunks, not just top-1
    top_result = reranked_results[0]  # Best chunk for display
    
    # Aggregate top-3 chunks - CLEAN format without markers that confuse LLM
    chunk_texts = [r['text'] for r in reranked_results[:3]]
    aggregated_context = "\n\n".join(chunk_texts)
    
    print(f"\n📊 Using {len(chunk_texts)} chunks for context (total: {len(aggregated_context)} chars)")
    print(f"   Chunk 1: {len(chunk_texts[0])} chars - {chunk_texts[0][:80]}...")
    if len(chunk_texts) > 1:
        print(f"   Chunk 2: {len(chunk_texts[1])} chars - {chunk_texts[1][:80]}...")
    if len(chunk_texts) > 2:
        print(f"   Chunk 3: {len(chunk_texts[2])} chars - {chunk_texts[2][:80]}...")
    
    # Step 6: Generate AI explanation with aggregated context
    generator = LegalResponseGenerator()
    ai_explanation = generator.generate_structured_explanation(
        query=query,
        clause_text=aggregated_context,
        metadata=top_result["metadata"]
    )
    
    # Step 7: Retry logic if answer is empty and other chunks exist (FIX 4)
    if is_empty_answer(ai_explanation) and len(reranked_results) > 1:
        print("⚠️ Empty answer detected, retrying with different chunk combination...")
        
        # Try focusing on just the top 2 chunks (sometimes less is more)
        focused_context = "\n\n".join([r['text'] for r in reranked_results[:2]])
        print(f"   Retry using top 2 chunks only ({len(focused_context)} chars)")
        
        ai_explanation = generator.generate_structured_explanation(
            query=query,
            clause_text=focused_context,
            metadata=reranked_results[0]["metadata"]
        )
        
        # If still empty, try chunk 2 alone (highest answer likelihood)
        if is_empty_answer(ai_explanation) and len(reranked_results) > 1:
            print("⚠️ Still empty, trying chunk 2 alone (highest answer likelihood)...")
            ai_explanation = generator.generate_structured_explanation(
                query=query,
                clause_text=reranked_results[0]['text'],  # Just the best reranked chunk
                metadata=reranked_results[0]["metadata"]
            )
    
    # Step 8: Extract clause title and section from top result
    clause_info = extract_clause_info(top_result["text"])
    
    # Step 9: Calculate relevance score and matched terms
    relevance_score = int(top_result["original_score"] * 100)  # Use original embedding score for display
    matched_terms = extract_matched_terms(query, top_result["text"])
    
    # Step 10: Format response to match frontend structure
    response = {
        "clause": {
            "title": clause_info["title"],
            "section": f"{top_result['metadata'].get('source', 'Unknown Document')} — Page {top_result['metadata'].get('page', 'N/A')}",
            "content": top_result["text"]
        },
        "explanation": {
            **ai_explanation,
            "confidenceReason": ai_explanation.get("confidenceReason", "Based on clause analysis")
        },
        "relevance": {
            "score": relevance_score,
            "matchedTerms": matched_terms
        }
    }
    
    # Debug logging
    print("="*80)
    print("QUERY RESPONSE:")
    print(f"Clause Title: {response['clause']['title']}")
    print(f"Clause Section: {response['clause']['section']}")
    print(f"Clause Content Length: {len(response['clause']['content'])} chars")
    print(f"Explanation Keys: {response['explanation'].keys()}")
    print(f"Summary: {response['explanation'].get('summary', 'N/A')}")
    print(f"Meaning: {response['explanation'].get('meaning', 'N/A')}")
    print(f"Favored Party: {response['explanation'].get('favoredParty', 'N/A')}")
    print(f"Key Terms: {response['explanation'].get('keyTerms', [])}")
    print(f"Relevance Score: {response['relevance']['score']}")
    print(f"Matched Terms: {response['relevance']['matchedTerms']}")
    print("="*80)
    
    return response


def extract_clause_info(text: str) -> Dict[str, str]:
    """
    Extract clause title from text using pattern matching
    Looks for all-caps titles or numbered sections
    """
    lines = text.split('\n')
    
    # Look for all-caps title (common in legal documents)
    for line in lines[:3]:  # Check first 3 lines
        line = line.strip()
        if line and line.isupper() and len(line.split()) <= 10:
            return {
                "title": line,
                "section": "Article — Section"
            }
    
    # Look for numbered sections (e.g., "12.2" or "Article 12")
    for line in lines[:3]:
        if re.search(r'\d+\.\d+|\b[Aa]rticle\s+\d+', line):
            return {
                "title": line.strip().upper(),
                "section": "Section Detected"
            }
    
    # Default: use first meaningful line
    first_line = text.split('\n')[0].strip()
    if len(first_line) > 100:
        first_line = first_line[:100] + "..."
    
    return {
        "title": first_line.upper() if first_line else "RETRIEVED CLAUSE",
        "section": "Contract Clause"
    }


def extract_matched_terms(query: str, text: str) -> List[str]:
    """
    Extract terms from query that appear in the retrieved text
    """
    # Normalize for comparison
    query_lower = query.lower()
    text_lower = text.lower()
    
    # Split query into words and filter stopwords
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'what', 'how',
                 'when', 'where', 'which', 'who', 'about', 'this', 'that', 'these', 'those'}
    
    query_words = [
        word.strip('.,!?;:') 
        for word in query_lower.split() 
        if word.strip('.,!?;:') not in stopwords and len(word) > 2
    ]
    
    # Find which query terms appear in the text
    matched = []
    for word in query_words:
        if word in text_lower:
            matched.append(word)
    
    # If no matches, return generic terms
    if not matched:
        matched = ["contract", "clause", "legal"]
    
    return matched[:6]  # Limit to 6 terms
