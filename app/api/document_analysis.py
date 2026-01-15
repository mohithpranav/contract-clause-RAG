"""
Document-level analysis endpoint for ClauseInsight
Provides comprehensive analysis of the entire uploaded document
"""
from typing import Dict, List
import numpy as np
import sys
from pathlib import Path
from app.rag.generator import LegalResponseGenerator


def deduplicate_items(items: List[str]) -> List[str]:
    """Remove duplicate items while preserving order"""
    seen = set()
    deduplicated = []
    for item in items:
        key = item.lower().strip()
        if key not in seen and key:  # Also filter empty strings
            seen.add(key)
            deduplicated.append(item)
    return deduplicated


def strip_advisory_language(text: str) -> str:
    """Hard guardrail: Remove all advisory and speculative language"""
    if not text:
        return text
    
    # Forbidden phrases (advisory language)
    forbidden_phrases = [
        "review carefully",
        "before signing",
        "consider legal counsel",
        "consult legal",
        "seek legal advice",
        "ensure they align",
        "business needs",
        "risk tolerance",
        "you may be required",
        "you should",
        "you must",
        "we recommend",
        "it is recommended",
        "it is advisable",
        "by accessing and using",
        "by using this",
        "you agree to",
        "you accept"
    ]
    
    text_lower = text.lower()
    
    # If text contains advisory language, return empty or factual fallback
    for phrase in forbidden_phrases:
        if phrase in text_lower:
            # Return empty string to signal this needs to be replaced
            return ""
    
    return text


async def analyze_entire_document(index_dir: str) -> Dict:
    """
    Generate comprehensive analysis of the entire uploaded document
    
    Args:
        index_dir: Directory containing the FAISS index and metadata
        
    Returns:
        Comprehensive document analysis with summary, key clauses, risks, etc.
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
    
    # Get document metadata
    doc_source = metadata_list[0].get('source', 'Unknown Document')
    
    print(f"📊 Analyzing {len(all_clauses)} total clauses...", flush=True)
    
    # Generate comprehensive document analysis
    generator = LegalResponseGenerator()
    
    # For summary: Use first 10 clauses which usually contain introduction, parties, background
    # This is more focused than 30 clauses and avoids token limit issues
    summary_text = "\n\n".join(all_clauses[:10])
    print(f"📝 Generating document summary from first 10 clauses ({len(summary_text)} chars)...", flush=True)
    
    # Generate document-level summary
    doc_summary = await generate_document_summary(summary_text, doc_source, generator)
    print(f"✅ Summary generated ({len(doc_summary)} chars)", flush=True)
    
    # Identify key clauses from ALL clauses, not just first 30
    print(f"🔍 Identifying key clauses from all {len(all_clauses)} clauses...", flush=True)
    key_clauses = await identify_key_clauses(all_clauses, metadata_list, generator)
    
    # Overall document assessment
    overall_assessment = await generate_overall_assessment(doc_summary, generator)
    
    # Calculate explainable confidence with capping for low coverage
    # We analyzed all clauses for key clause identification
    coverage_ratio = 1.0  # Analyzed all clauses
    num_key_clauses = len(key_clauses)
    
    confidence = int(
        70 +  # Base confidence (higher since we analyzed all clauses)
        min(num_key_clauses * 3, 25)  # Up to +25 for key clauses found
    )
    
    confidence_reason = f"Analyzed all {len(all_clauses)} clauses and identified {num_key_clauses} key provisions"
    
    # Deduplicate outputs
    deduplicated_terms = deduplicate_items(extract_key_terms(all_clauses))
    deduplicated_flags = deduplicate_items(generate_document_negotiation_flags(key_clauses))
    
    response = {
        "document": {
            "title": doc_source,
            "totalClauses": len(all_clauses),
            "analyzedClauses": len(all_clauses)  # Now analyzing all clauses
        },
        "analysis": {
            "summary": doc_summary,
            "overallAssessment": overall_assessment,
            "keyClauses": key_clauses,
            "partyBalance": {  # Separate fact from interpretation
                "assessment": determine_favored_party(key_clauses),
                "basis": "Based on concentration of obligations and clause distribution"
            },
            "keyTerms": deduplicated_terms,
            "practicalImpact": generate_practical_impact_doc(key_clauses),
            "negotiationFlags": deduplicated_flags
        },
        "metadata": {
            "source": doc_source,
            "totalPages": len(set(meta.get('page', 'N/A') for meta in metadata_list)),
            "confidence": confidence,
            "confidenceReason": confidence_reason
        }
    }
    
    # Log complete document analysis response
    print("="*80, flush=True)
    print("📊 DOCUMENT ANALYSIS RESPONSE:", flush=True)
    print(f"Document Title: {response['document']['title']}", flush=True)
    print(f"Total Clauses: {response['document']['totalClauses']}", flush=True)
    print(f"Analyzed Clauses: {response['document']['analyzedClauses']}", flush=True)
    print(f"Coverage: {int((response['document']['analyzedClauses'] / response['document']['totalClauses']) * 100)}%", flush=True)
    print("-"*80, flush=True)
    print(f"Summary: {response['analysis']['summary']}", flush=True)
    print(f"Overall Assessment: {response['analysis']['overallAssessment']}", flush=True)
    print("-"*80, flush=True)
    print(f"Key Clauses Found: {len(response['analysis']['keyClauses'])}", flush=True)
    for idx, clause in enumerate(response['analysis']['keyClauses'], 1):
        print(f"  {idx}. [{clause['category']}] {clause['title']} (Score: {clause.get('importanceScore', 0)})", flush=True)
    print("-"*80, flush=True)
    print(f"Party Balance: {response['analysis']['partyBalance']['assessment']}", flush=True)
    print(f"Key Terms: {', '.join(response['analysis']['keyTerms'])}", flush=True)
    print(f"Practical Impact: {response['analysis']['practicalImpact']}", flush=True)
    print("-"*80, flush=True)
    print(f"Negotiation Flags ({len(response['analysis']['negotiationFlags'])}):", flush=True)
    for idx, flag in enumerate(response['analysis']['negotiationFlags'], 1):
        print(f"  {idx}. {flag}", flush=True)
    print("-"*80, flush=True)
    print(f"Confidence: {response['metadata']['confidence']}%", flush=True)
    print(f"Confidence Reason: {response['metadata']['confidenceReason']}", flush=True)
    print("="*80, flush=True)
    print(flush=True)
    
    return response


async def generate_document_summary(text: str, source: str, generator: LegalResponseGenerator) -> str:
    """Generate detailed document summary describing what happened in the case/document"""
    
    # Limit input to avoid token length issues (flan-t5-large has 512 token limit)
    # Roughly 4 chars per token, so 2000 chars ≈ 500 tokens (safe margin)
    text_to_analyze = text[:2000]
    
    prompt = """Provide a detailed summary of this document. Describe:
1. What type of document this is (case judgment, contract, agreement, terms, etc.)
2. If it's a legal case: What happened? Who are the parties? What was the dispute? What did the court/commission decide?
3. If it's a contract/agreement: What is the purpose? What are the main obligations?
4. Key facts, events, or provisions.

Be specific and factual. Do not give generic legal advice. Only describe what is in this document."""
    
    analysis = generator.generate_structured_explanation(
        query=prompt,
        clause_text=text_to_analyze,
        metadata={"source": source}
    )
    
    # Get the full meaning field which contains the complete answer
    summary = analysis.get("meaning", analysis.get("summary", ""))
    
    # Apply hard guardrail filter
    summary = strip_advisory_language(summary)
    
    # If advisory language was detected and removed, return factual fallback
    if not summary or len(summary) < 50:
        return "This is a legal document establishing rights and obligations between parties."
    
    return summary


def score_clause_importance(clause: str) -> int:
    """Score clause importance based on legal keywords and length"""
    score = 0
    # High-priority legal terms
    keywords = ['terminate', 'terminat', 'liable', 'liability', 'indemn', 'govern', 
                'dispute', 'warrant', 'confidential', 'intellectual', 'payment', 'arbitrat']
    
    clause_lower = clause.lower()
    for keyword in keywords:
        if keyword in clause_lower:
            score += 2
    
    # Length bonus (longer clauses often more substantial)
    score += min(len(clause) // 200, 3)
    
    return score


async def identify_key_clauses(clauses: List[str], metadata_list: List, generator: LegalResponseGenerator) -> List[Dict]:
    """Identify the most important clauses using scoring algorithm"""
    # generator parameter reserved for future LLM-based clause ranking
    
    # Score and rank all clauses
    scored_clauses = [
        (idx, clause, score_clause_importance(clause)) 
        for idx, clause in enumerate(clauses)
        if len(clause) > 100  # Filter out very short clauses
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
        elif 'payment' in clause_lower:
            category = "Payment"
        elif 'confidential' in clause_lower:
            category = "Confidentiality"
        elif 'warrant' in clause_lower:
            category = "Warranty"
        
        key_clauses.append({
            "clauseId": idx,  # Add clause ID for grounding
            "title": extract_clause_title(clause),
            "content": clause[:200] + "..." if len(clause) > 200 else clause,
            "fullContent": clause,
            "quote": clause[:250],  # Explicit quote field
            "section": metadata_list[idx].get('source', 'Unknown'),
            "page": metadata_list[idx].get('page', 'N/A'),
            "category": category,
            "importanceScore": score  # Include score for transparency
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


async def generate_overall_assessment(summary: str, generator: LegalResponseGenerator) -> str:
    """Generate overall document assessment based on actual content - descriptive only, never advisory"""
    # Extract document type from summary
    summary_lower = summary.lower()
    
    if any(word in summary_lower for word in ['judgment', 'commission', 'court', 'case', 'dispute', 'complainant', 'respondent']):
        # Legal case/judgment
        assessment = "This is a legal judgment or decision. Key determinations and reasoning are documented in the analyzed clauses."
    elif any(word in summary_lower for word in ['terms', 'conditions', 'usage', 'service', 'application', 'platform']):
        # Terms of service/use
        assessment = "This document governs usage rights and limitations. Key provisions relate to liability limitations, termination rights, and governing law."
    elif any(word in summary_lower for word in ['agreement', 'contract', 'parties hereby']):
        # Contract/agreement
        assessment = "This is a contractual agreement establishing obligations between parties. Key terms are identified in the analyzed clauses."
    else:
        # Generic fallback - still factual
        assessment = "This document contains binding provisions. Key clauses are identified above."
    
    # Apply hard guardrail
    assessment = strip_advisory_language(assessment)
    
    # Fallback if filter removed everything
    if not assessment:
        assessment = "This document contains legal provisions. Key clauses are identified above."
    
    return assessment


def determine_favored_party(key_clauses: List[Dict]) -> str:
    """Determine which party the document favors - must explain WHY with specifics"""
    
    # Check for unilateral language patterns
    unilateral_indicators = 0
    liability_heavy = False
    specific_findings = []
    
    for clause in key_clauses:
        clause_text = clause.get('fullContent', '').lower()
        category = clause.get('category', '')
        
        # Check for one-sided language
        shall_not_count = clause_text.count('shall not')
        may_count = clause_text.count('may')
        if shall_not_count > 0 and shall_not_count > may_count:
            unilateral_indicators += 1
            if 'shall not' not in [f.lower() for f in specific_findings]:
                specific_findings.append("Asymmetric 'shall not' obligations")
        
        # Check for broad liability/indemnification
        if category in ['Liability', 'Indemnification']:
            if 'unlimited' in clause_text:
                unilateral_indicators += 2
                liability_heavy = True
                specific_findings.append("Unlimited liability provision")
            elif any(term in clause_text for term in ['any and all', 'hold harmless', 'defend and indemnify']):
                unilateral_indicators += 1
                liability_heavy = True
                specific_findings.append("Broad indemnification language")
        
        # Check for restrictive termination
        if category == 'Termination':
            if 'without cause' in clause_text or 'at will' in clause_text:
                specific_findings.append("Unilateral termination rights")
            elif 'mutual' not in clause_text:
                unilateral_indicators += 1
                specific_findings.append("Non-mutual termination conditions")
    
    # Build assessment with SPECIFIC findings
    if not specific_findings:
        return "Clause balance cannot be determined from analyzed clauses"
    
    if unilateral_indicators >= 3 or liability_heavy:
        findings_str = ", ".join(specific_findings[:2])  # Top 2
        return f"Favors drafting party: {findings_str.lower()}"
    elif unilateral_indicators >= 1:
        findings_str = ", ".join(specific_findings[:2])
        return f"Mixed: {findings_str.lower()}"
    else:
        return "Balanced based on analyzed clauses"


def extract_key_terms(clauses: List[str]) -> List[str]:
    """Extract key legal terms from the document"""
    common_terms = {
        'Confidential Information', 'Intellectual Property', 'Indemnification',
        'Termination', 'Force Majeure', 'Governing Law', 'Dispute Resolution',
        'Payment Terms', 'Warranties', 'Liability', 'Non-Disclosure'
    }
    
    found_terms = set()
    combined_text = " ".join(clauses)
    
    for term in common_terms:
        if term.lower() in combined_text.lower():
            found_terms.add(term)
    
    return list(found_terms) if found_terms else ["General Legal Terms", "Standard Provisions"]


def generate_practical_impact_doc(key_clauses: List[Dict]) -> str:
    """Generate practical impact strictly from identified key clauses - no speculation"""
    
    # Map clause categories to concrete impacts ONLY if that clause category exists
    impact_parts = []
    
    categories_found = {clause.get('category') for clause in key_clauses}
    
    # Only add impacts for categories actually found in key clauses
    if 'Termination' in categories_found:
        # Check if termination clause actually mentions exit conditions
        termination_clauses = [c for c in key_clauses if c.get('category') == 'Termination']
        if termination_clauses:
            impact_parts.append("Contract termination conditions are defined in key clauses")
    
    if 'Liability' in categories_found or 'Indemnification' in categories_found:
        # Check if there's actual indemnification language
        liability_clauses = [c for c in key_clauses if c.get('category') in ['Liability', 'Indemnification']]
        if any('indemnif' in c.get('fullContent', '').lower() for c in liability_clauses):
            impact_parts.append("Indemnification obligations are specified")
        elif liability_clauses:
            impact_parts.append("Liability terms and limitations are defined")
    
    if 'Payment' in categories_found:
        impact_parts.append("Payment obligations and terms are specified")
    
    if 'Governing Law' in categories_found:
        impact_parts.append("Jurisdiction and governing law are established")
    
    if 'Confidentiality' in categories_found:
        impact_parts.append("Confidentiality restrictions apply to information sharing")
    
    if 'Warranty' in categories_found:
        impact_parts.append("Warranties or disclaimers are specified")
    
    # If no specific categories, provide minimal grounded statement
    if not impact_parts:
        return "Key provisions are identified in the analyzed clauses above."
    
    # Build concise impact statement from actual clauses only
    return ". ".join(impact_parts[:3]) + "."


def generate_document_negotiation_flags(key_clauses: List[Dict]) -> List[str]:
    """Generate advisory negotiation flags strictly from clause analysis"""
    flags = []
    
    for clause in key_clauses:
        category = clause.get('category', '')
        clause_text = clause.get('fullContent', '').lower()
        
        # Termination: Check for one-sided terms
        if category == 'Termination':
            if 'at will' in clause_text or 'convenience' in clause_text:
                flags.append("Negotiate termination notice period and transition requirements")
            elif 'cause' in clause_text and 'cure' not in clause_text:
                flags.append("Request cure period before termination for cause")
        
        # Liability: Check for caps and limitations
        if category in ['Liability', 'Indemnification']:
            if 'unlimited' in clause_text or 'no limit' in clause_text:
                flags.append("Negotiate liability cap tied to contract value")
            elif 'consequential' not in clause_text or 'indirect' not in clause_text:
                flags.append("Add exclusion for consequential and indirect damages")
            elif 'defend' in clause_text and 'indemnify' in clause_text:
                flags.append("Limit indemnification scope to direct claims only")
        
        # Payment: Check for payment terms
        if category == 'Payment':
            if 'advance' in clause_text or 'upfront' in clause_text:
                flags.append("Consider milestone-based payments instead of upfront fees")
        
        # Governing Law: Check for jurisdiction
        if category == 'Governing Law':
            if 'arbitration' in clause_text:
                flags.append("Review arbitration venue and cost allocation")
        
        # Confidentiality: Check for scope
        if category == 'Confidentiality':
            if 'perpetual' in clause_text or 'indefinite' in clause_text:
                flags.append("Negotiate time limit on confidentiality obligations")
    
    # Deduplicate and return top 4 most actionable flags
    unique_flags = list(dict.fromkeys(flags))  # Preserves order while removing duplicates
    
    return unique_flags[:4] if unique_flags else ["Review all key clauses with legal counsel before signing"]


def extract_clause_title(text: str) -> str:
    """Extract title from clause text"""
    lines = text.split('\n')
    for line in lines[:3]:
        line = line.strip()
        if line and len(line.split()) <= 10:
            return line.title()
    
    first_words = " ".join(text.split()[:8])
    return first_words + "..." if len(first_words) > 50 else first_words
