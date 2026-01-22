"""
Detailed analysis endpoint for ClauseInsight
Provides comprehensive analysis of a specific clause
"""
from typing import Dict
from app.rag.generator import LegalResponseGenerator


async def analyze_clause(clause_text: str, metadata: Dict) -> Dict:
    # Generate comprehensive analysis
    generator = LegalResponseGenerator()
    
    # Generic analysis query for detailed breakdown
    analysis_query = "Provide a comprehensive analysis of this clause"
    
    detailed_analysis = generator.generate_structured_explanation(
        query=analysis_query,
        clause_text=clause_text,
        metadata=metadata
    )
    
    # Return comprehensive response
    response = {
        "clause": {
            "title": extract_title(clause_text),
            "section": f"{metadata.get('source', 'Unknown Document')} — Page {metadata.get('page', 'N/A')}",
            "content": clause_text
        },
        "analysis": {
            "summary": detailed_analysis["summary"],
            "meaning": detailed_analysis["meaning"],
            "favoredParty": detailed_analysis.get("favoredParty", "N/A"),
            "keyTerms": detailed_analysis.get("keyTerms", []),
            "practicalImpact": detailed_analysis.get("practicalImpact", "") or generate_practical_impact(clause_text, detailed_analysis),
            "negotiationFlags": generate_negotiation_flags(clause_text)
        },
        "metadata": {
            "source": metadata.get('source', 'Unknown'),
            "page": metadata.get('page', 'N/A'),
            "confidence": detailed_analysis.get("confidence", 85),
            "confidenceReason": detailed_analysis.get("confidenceReason", "Based on comprehensive analysis")
        }
    }
    
    return response


def extract_title(text: str) -> str:
    """Extract title from clause text"""
    lines = text.split('\n')
    for line in lines[:3]:
        line = line.strip()
        if line and line.isupper() and len(line.split()) <= 10:
            return line
    
    first_line = text.split('\n')[0].strip()
    if len(first_line) > 100:
        first_line = first_line[:100] + "..."
    return first_line.upper() if first_line else "CLAUSE ANALYSIS"


def generate_practical_impact(clause_text: str, analysis: Dict) -> str:
    """Generate practical impact description"""
    meaning = analysis.get("meaning", "")
    if len(meaning) > 300:
        return meaning[:300] + "..."
    return meaning


def generate_negotiation_flags(clause_text: str) -> list:
    """Generate negotiation flags from clause text analysis"""
    text_lower = clause_text.lower()
    
    flags = []
    
    # Check for one-sided terms
    if 'shall not' in text_lower and 'may' in text_lower:
        flags.append("Asymmetric obligations - one party 'shall' while other 'may'")
    
    # Check for unlimited liability
    if 'unlimited' in text_lower or ('no limit' in text_lower and 'liability' in text_lower):
        flags.append("Unlimited liability exposure")
    
    # Check for broad indemnification
    if 'indemnify' in text_lower and 'defend' in text_lower and 'hold harmless' in text_lower:
        flags.append("Broad indemnification clause - triple obligation")
    
    # Check for restrictive terms
    if 'non-compete' in text_lower or 'non-solicitation' in text_lower:
        flags.append("Contains restrictive covenants - review scope and duration")
    
    return flags if flags else ["No major negotiation flags identified"]


# Confidence calculation now handled by generator.py
# Keeping this as fallback for backward compatibility
def calculate_confidence(clause_text: str) -> int:
    """Fallback confidence calculation (deprecated - use generator's confidence)"""
    return 85  # Default confidence if generator doesn't provide one
