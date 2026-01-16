"""
Document-level analysis endpoint for ClauseInsight
Simple approach: 4-line summary + party balance + key clauses
"""
from typing import Dict, List
import numpy as np
from pathlib import Path
from app.rag.generator import LegalResponseGenerator


def deduplicate_items(items: List[str]) -> List[str]:
    """Remove duplicate items while preserving order"""
    seen = set()
    deduplicated = []
    for item in items:
        key = item.lower().strip()
        if key not in seen and key:
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


async def analyze_entire_document(index_dir: str) -> Dict:
    """
    Generate simple document analysis: 4-line summary + party balance + key clauses
    
    Args:
        index_dir: Directory containing the FAISS index and metadata
        
    Returns:
        Simple document analysis with grounded outputs
    """
    print("\n" + "="*80, flush=True)
    print("📄 STARTING DOCUMENT ANALYSIS", flush=True)
    print("="*80, flush=True)
    
    index_path = Path(index_dir)
    metadata_path = index_path / "metadata.npy"
    
    if not metadata_path.exists():
        raise ValueError("No document has been indexed yet. Please upload a document first.")
    
    # Load all indexed clauses
    metadata_list = np.load(str(metadata_path), allow_pickle=True)
    
    if len(metadata_list) == 0:
        raise ValueError("No clauses found in the indexed document.")
    
    # Extract all clause texts
    all_clauses = [meta.get('text', '') for meta in metadata_list]
    doc_source = metadata_list[0].get('source', 'Unknown Document')
    
    print(f"📊 Analyzing {len(all_clauses)} total clauses...", flush=True)
    
    # Initialize generator
    generator = LegalResponseGenerator()
    
    # Generate 4-line summary from first 15 clauses (contains intro, parties, key facts)
    summary_context = "\n\n".join(all_clauses[:15])
    print(f"📝 Generating 4-line summary from first 15 clauses ({len(summary_context)} chars)...", flush=True)
    
    # Simple prompt for story summary
    summary_prompt = "Summarize what happened in this document in 4 lines. Focus on the story - who, what, when, outcome."
    
    analysis = generator.generate_structured_explanation(
        query=summary_prompt,
        clause_text=summary_context,
        metadata={"source": doc_source}
    )
    
    doc_summary = analysis.get("meaning", analysis.get("summary", "Document summary unavailable."))
    print(f"✅ Summary generated ({len(doc_summary)} chars)", flush=True)
    
    # Identify key clauses
    print(f"🔍 Identifying key clauses from all {len(all_clauses)} clauses...", flush=True)
    key_clauses = identify_key_clauses(all_clauses, metadata_list)
    
    # Calculate confidence
    confidence = int(70 + min(len(key_clauses) * 3, 20))
    confidence_reason = f"Analyzed {len(all_clauses)} clauses, identified {len(key_clauses)} key provisions"
    
    # Build simple response (NO overall assessment)
    response = {
        "document": {
            "title": doc_source,
            "totalClauses": len(all_clauses),
            "analyzedClauses": len(all_clauses)
        },
        "analysis": {
            "summary": doc_summary,  # 4-line story
            "keyClauses": key_clauses,
            "favoredParty": determine_favored_party(key_clauses),  # Match frontend field name
            "keyTerms": deduplicate_items(extract_key_terms(all_clauses)),
            "practicalImpact": generate_practical_impact_doc(key_clauses),
            "negotiationFlags": deduplicate_items(generate_document_negotiation_flags(key_clauses))
        },
        "metadata": {
            "source": doc_source,
            "totalPages": len(set(meta.get('page', 'N/A') for meta in metadata_list)),
            "confidence": confidence,
            "confidenceReason": confidence_reason
        }
    }
    
    # Log response
    print("="*80, flush=True)
    print("📊 ANALYSIS COMPLETE:", flush=True)
    print(f"Document: {doc_source}", flush=True)
    print(f"Total Clauses: {len(all_clauses)}", flush=True)
    print("-"*80, flush=True)
    print(f"Summary: {doc_summary}", flush=True)
    print(f"Favored Party: {response['analysis']['favoredParty']}", flush=True)
    print(f"Key Clauses: {len(key_clauses)}", flush=True)
    for idx, clause in enumerate(key_clauses[:3], 1):
        print(f"  {idx}. [{clause['category']}] {clause['title'][:60]}...", flush=True)
    print("-"*80, flush=True)
    print(f"Confidence: {confidence}% - {confidence_reason}", flush=True)
    print("="*80, flush=True)
    
    return response


def score_clause_importance(clause: str) -> int:
    """Score clause importance based on legal keywords and length"""
    score = 0
    keywords = ['terminate', 'terminat', 'liable', 'liability', 'indemn', 'govern', 
                'dispute', 'warrant', 'confidential', 'intellectual', 'payment', 'arbitrat',
                'compensation', 'awarded', 'findings', 'held that', 'commission']
    
    clause_lower = clause.lower()
    for keyword in keywords:
        if keyword in clause_lower:
            score += 2
    
    # Length bonus
    score += min(len(clause) // 200, 3)
    
    return score


def identify_key_clauses(clauses: List[str], metadata_list: List) -> List[Dict]:
    """Identify the most important clauses using scoring"""
    # Score and rank all clauses
    scored_clauses = [
        (idx, clause, score_clause_importance(clause)) 
        for idx, clause in enumerate(clauses)
        if len(clause) > 100
    ]
    
    # Sort by score descending and take top 5
    ranked_clauses = sorted(scored_clauses, key=lambda x: x[2], reverse=True)[:5]
    
    key_clauses = []
    for idx, clause, score in ranked_clauses:
        # Determine category from content
        clause_lower = clause.lower()
        category = "General"
        if 'terminat' in clause_lower:
            category = "Termination"
        elif 'liab' in clause_lower or 'indemn' in clause_lower:
            category = "Liability"
        elif 'govern' in clause_lower or 'dispute' in clause_lower:
            category = "Governing Law"
        elif 'payment' in clause_lower or 'compensation' in clause_lower:
            category = "Payment"
        elif 'confidential' in clause_lower:
            category = "Confidentiality"
        elif 'warrant' in clause_lower:
            category = "Warranty"
        
        key_clauses.append({
            "clauseId": idx,
            "title": extract_clause_title(clause),
            "content": clause[:200] + "..." if len(clause) > 200 else clause,
            "fullContent": clause,
            "quote": clause[:250],
            "section": metadata_list[idx].get('source', 'Unknown'),
            "page": metadata_list[idx].get('page', 'N/A'),
            "category": category,
            "importanceScore": score
        })
    
    return key_clauses if key_clauses else [{
        "clauseId": 0,
        "title": "General Provisions", 
        "content": clauses[0][:200], 
        "fullContent": clauses[0],
        "quote": clauses[0][:250],
        "section": "Document", 
        "page": "1", 
        "category": "General",
        "importanceScore": 0
    }]


def determine_favored_party(key_clauses: List[Dict]) -> str:
    """Simple party balance assessment"""
    unilateral_count = 0
    
    for clause in key_clauses:
        clause_text = clause.get('fullContent', '').lower()
        
        # Check for one-sided language
        if 'shall not' in clause_text or 'without cause' in clause_text:
            unilateral_count += 1
        if 'unlimited' in clause_text or 'any and all' in clause_text:
            unilateral_count += 2
    
    if unilateral_count >= 3:
        return "Favors drafting party (unilateral terms detected)"
    elif unilateral_count >= 1:
        return "Mixed (some one-sided provisions)"
    else:
        return "Balanced"


def extract_key_terms(clauses: List[str]) -> List[str]:
    """Extract key legal terms from the document"""
    common_terms = {
        'Confidential Information', 'Intellectual Property', 'Indemnification',
        'Termination', 'Force Majeure', 'Governing Law', 'Dispute Resolution',
        'Payment Terms', 'Warranties', 'Liability', 'Compensation'
    }
    
    found_terms = set()
    combined_text = " ".join(clauses)
    
    for term in common_terms:
        if term.lower() in combined_text.lower():
            found_terms.add(term)
    
    return list(found_terms) if found_terms else ["General Legal Terms"]


def generate_practical_impact_doc(key_clauses: List[Dict]) -> str:
    """Generate practical impact from key clauses"""
    impact_parts = []
    categories_found = {clause.get('category') for clause in key_clauses}
    
    if 'Termination' in categories_found:
        impact_parts.append("Contract termination conditions defined")
    if 'Liability' in categories_found:
        impact_parts.append("Liability limitations specified")
    if 'Payment' in categories_found:
        impact_parts.append("Payment obligations established")
    if 'Governing Law' in categories_found:
        impact_parts.append("Jurisdiction determined")
    
    return ". ".join(impact_parts) + "." if impact_parts else "Key provisions identified in analyzed clauses."


def generate_document_negotiation_flags(key_clauses: List[Dict]) -> List[str]:
    """Generate negotiation flags from clauses"""
    flags = []
    
    for clause in key_clauses:
        category = clause.get('category', '')
        clause_text = clause.get('fullContent', '').lower()
        
        if category == 'Termination' and 'at will' in clause_text:
            flags.append("Negotiate termination notice period")
        if category in ['Liability'] and 'unlimited' in clause_text:
            flags.append("Negotiate liability cap")
        if 'consequential' not in clause_text and category == 'Liability':
            flags.append("Add exclusion for consequential damages")
    
    return list(dict.fromkeys(flags))[:4] if flags else ["Review key clauses with legal counsel"]


def extract_clause_title(text: str) -> str:
    """Extract title from clause text"""
    lines = text.split('\n')
    for line in lines[:3]:
        line = line.strip()
        if line and len(line.split()) <= 10:
            return line.title()
    
    first_words = " ".join(text.split()[:8])
    return first_words + "..." if len(first_words) > 50 else first_words
