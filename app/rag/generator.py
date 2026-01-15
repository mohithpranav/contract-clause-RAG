from typing import List, Dict
import re
from transformers import pipeline
import torch


class LegalResponseGenerator:
    """
    Generates grounded legal explanations using Hugging Face LLM.
    """

    def __init__(self, model_name: str = "google/flan-t5-large"):
        """
        Initialize with lazy loading - model loads on first use
        """
        self.model_name = model_name
        self.generator = None
        self._model_loaded = False

    def _ensure_model_loaded(self):
        """Load model lazily on first use"""
        if not self._model_loaded:
            print(f"🔄 Loading Hugging Face model: {self.model_name}...")
            
            # Load model using pipeline
            self.generator = pipeline(
                "text2text-generation",
                model=self.model_name,
                device=0 if torch.cuda.is_available() else -1,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            device = "GPU" if torch.cuda.is_available() else "CPU"
            print(f"✅ Model loaded successfully on {device}")
            self._model_loaded = True

    def generate_structured_explanation(
        self, 
        query: str, 
        clause_text: str, 
        metadata: Dict
    ) -> Dict:
        """
        Generate structured explanation using pure LLM reasoning with retrieved context.
        Simple, robust, grounded approach.
        """
        # Load model on first use
        self._ensure_model_loaded()
        
        print(f"\n🤖 Generating LLM response for: '{query}'")
        
        # Determine context window based on content length
        max_context_chars = 3500 if len(clause_text) > 3000 else len(clause_text)
        context_to_use = clause_text[:max_context_chars]
        
        if len(clause_text) > max_context_chars:
            print(f"⚠️ Context truncated from {len(clause_text)} to {max_context_chars} chars")
        
        # SINGLE UNIFIED PROMPT - Let LLM handle all question types naturally
        main_prompt = f"""Answer the question using the context below. Provide a complete answer with details, not just a title or heading.

Question: {query}

Context:
{context_to_use}

Complete answer:"""
        
        # Generate main answer with parameters tuned for T5 models
        answer = self.generator(
            main_prompt, 
            max_length=600,  # Increased to allow fuller answers
            min_length=50,   # Force minimum length to prevent title-only outputs
            do_sample=True, 
            temperature=0.7,  # Higher temp for more detailed outputs
            top_p=0.95,
            repetition_penalty=1.2  # Prevent repetition
        )[0]['generated_text'].strip()
        
        print(f"✓ Answer generated ({len(answer)} chars)")
        
        # Practical impact ONLY if asked
        practical_impact = ""
        if any(word in query.lower() for word in ['impact', 'affect', 'consequence', 'mean', 'result', 'happen']):
            impact_prompt = f"""What is the practical impact based on this context?

Context: {clause_text[:1000]}

Impact:"""
            practical_impact = self.generator(
                impact_prompt, 
                max_length=200, 
                min_length=30,
                do_sample=True, 
                temperature=0.7
            )[0]['generated_text'].strip()
            print(f"✓ Impact generated")
        
        # Calculate honest confidence
        confidence, confidence_reason = self._calculate_honest_confidence(
            query, clause_text, answer
        )
        
        # Build response
        result = {
            "summary": answer[:200] if len(answer) > 200 else answer,  # Short summary
            "meaning": answer,  # Full answer
            "favoredParty": "N/A",  # Not relevant for most queries
            "keyTerms": self._extract_key_terms(clause_text)[:4],
            "practicalImpact": practical_impact,
            "confidence": confidence,
            "confidenceReason": confidence_reason
        }
        
        print(f"✅ Response complete (confidence: {confidence}% - {confidence_reason})\n")
        return result
    
    def _calculate_honest_confidence(
        self, query: str, context: str, answer: str
    ) -> tuple[int, str]:
        """
        Calculate confidence based on answer quality, not assumptions.
        Returns (score, reason)
        """
        score = 70  # Base
        
        # Check if answer contains context-specific information
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # Answer uses words from context (good sign of grounding)
        answer_context_overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        
        if answer_context_overlap > 0.3:
            score += 15  # Well grounded
            reason = "Answer references specific content from retrieved context"
        else:
            score -= 10
            reason = "Limited grounding in retrieved context"
        
        # Answer is specific (not vague)
        if len(answer) > 100 and any(word in answer.lower() for word in ['specific', 'explicit', 'state', 'mention', 'indicate']):
            score += 5
        
        # Answer admits uncertainty appropriately
        uncertainty_phrases = ['not contain', 'does not specify', 'unclear', 'not mentioned', 'not provided']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            score = min(score, 65)  # Cap at 65% for uncertain answers
            # Check if this is due to context selection issue
            if 'not contain' in answer.lower() and len(context) > 1000:
                reason = "Answer not found in selected context - may exist in alternate retrieved chunks"
            else:
                reason = "Information not fully specified in retrieved context"
        
        # Context is substantial
        if len(context) > 800:
            score += 5
        
        # Decrease if answer is very short for complex question
        if len(query.split()) > 6 and len(answer) < 50:
            score -= 15
            reason = "Brief answer for complex question - may lack detail"
        
        return (max(40, min(95, score)), reason)
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract important legal terms from the clause"""
        legal_terms = []
        
        # Extract all-caps terms (important in contracts)
        caps_pattern = r'\b[A-Z]{2,}[A-Z\s]{0,20}\b'
        caps_terms = re.findall(caps_pattern, text)
        for term in caps_terms[:4]:
            if len(term.strip()) > 2 and term not in legal_terms:
                legal_terms.append(term.title())
        
        # Common legal keywords
        keywords = ['agreement', 'party', 'parties', 'termination', 'notice', 
                   'confidential', 'liability', 'indemnify', 'breach', 'obligation',
                   'payment', 'warranty', 'representation', 'dispute']
        
        for keyword in keywords:
            if keyword in text.lower() and keyword.title() not in legal_terms and len(legal_terms) < 6:
                legal_terms.append(keyword.title())
        
        return legal_terms[:6] if legal_terms else ["Contract", "Agreement", "Clause"]
