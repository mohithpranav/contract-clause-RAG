from typing import List, Dict
import re
from transformers import pipeline
import torch

# Keep this minimal - LLM echoes verbose instructions instead of following them
MASTER_SYSTEM_PROMPT = "Answer using ONLY the context provided. Be precise and factual."


class LegalResponseGenerator:
  

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
        
        # Calculate available context space (T5 limit: 512 tokens ≈ 2048 chars)
        # Reserve space for prompt and query
        prompt_overhead = 350 + len(query)  # More realistic overhead with instructions
        max_total_chars = 2000  # Safe limit to stay under 512 tokens
        max_context_chars = max(400, max_total_chars - prompt_overhead)  # At least 400 chars
        
        # Determine context window based on content length and prompt size
        if len(clause_text) > max_context_chars:
            context_to_use = clause_text[:max_context_chars]
            print(f"⚠️ Context truncated from {len(clause_text)} to {max_context_chars} chars (prompt length: {len(query)})")
        else:
            context_to_use = clause_text
        # GENERATE DETAILED ANSWER - Keep prompt concise to avoid LLM echoing instructions
        main_prompt = f"""Answer this question using ONLY the context below.

Rules:
- For Yes/No questions: Start with Yes or No, then explain with facts from context
- For causation ("Did X cause Y?"): If context shows different cause Z, say "No, it was caused by Z" 
- For definitions: Give the definition from context only
- For numbers/compensation: List exact figures from context
- If context mentions alternative cause, that's a valid No answer
- Only say "no information" if context has nothing relevant

Question: {query}

Context:
{context_to_use}

Detailed Answer:"""
        
        # Generate main answer with parameters tuned for T5 models
        answer = self.generator(
            main_prompt, 
            max_length=1024,  # Increased significantly to prevent truncation
            min_length=50,    # Force minimum length to prevent title-only outputs
            do_sample=False,  # Use deterministic beam search (more stable)
            num_beams=4,      # Beam search for better quality
            early_stopping=True  # Stop when all beams finish
        )[0]['generated_text'].strip()
        
        # Post-process: Ensure answer ends with proper punctuation
        answer = self._ensure_complete_sentence(answer)
        
        # Check if answer indicates information not found in context
        if self._is_out_of_context_answer(answer):
            answer = "I don't have information about that in this document."
            summary = answer
        else:
            # GENERATE 2-3 LINE SUMMARY
            summary_prompt = f"""Answer this question in 2-3 lines using ONLY the context.

- For Yes/No: Start with Yes or No
- For causation: If context shows different cause, say No and state actual cause
- Be concise and factual

Question: {query}

Context:
{context_to_use}

Brief Answer:"""
            
            summary = self.generator(
                summary_prompt,
                max_length=200,  # Shorter for concise summary
                min_length=30,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )[0]['generated_text'].strip()
            summary = self._ensure_complete_sentence(summary)
        
        print(f"✓ Answer generated ({len(answer)} chars), Summary generated ({len(summary)} chars)")
        
        # Practical impact ONLY if asked
        practical_impact = ""
        if any(word in query.lower() for word in ['impact', 'affect', 'consequence', 'mean', 'result', 'happen']):
            impact_prompt = f"""What is the practical impact based on this context? Provide a complete answer.

Context: {clause_text[:1000]}

Impact:"""
            practical_impact = self.generator(
                impact_prompt, 
                max_length=300,  # Increased
                min_length=30,
                do_sample=False,  # Deterministic
                num_beams=4,
                early_stopping=True
            )[0]['generated_text'].strip()
            practical_impact = self._ensure_complete_sentence(practical_impact)
            print(f"✓ Impact generated")
        
        # Calculate honest confidence
        confidence, confidence_reason = self._calculate_honest_confidence(
            query, clause_text, answer
        )
        
        # Build response
        result = {
            "summary": summary,  # 2-3 line summary from LLM
            "meaning": answer,  # Detailed explanation from LLM
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
        score = 60  # Lower base score
        reasons = []
        
        # Check if answer contains context-specific information
        query_words = set(query.lower().split())
        answer_words = set(answer.lower().split())
        context_words = set(context.lower().split())
        
        # 1. Grounding check - more granular
        answer_context_overlap = len(answer_words & context_words) / max(len(answer_words), 1)
        
        if answer_context_overlap > 0.5:
            score += 20
            reasons.append("strongly grounded in context")
        elif answer_context_overlap > 0.3:
            score += 10
            reasons.append("moderately grounded in context")
        elif answer_context_overlap > 0.15:
            score += 5
            reasons.append("weakly grounded in context")
        else:
            score -= 10
            reasons.append("limited context grounding")
        
        # 2. Query-answer relevance - check if answer addresses query terms
        query_words_clean = query_words - {'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where', 'did'}
        query_answer_overlap = len(query_words_clean & answer_words) / max(len(query_words_clean), 1)
        
        if query_answer_overlap > 0.6:
            score += 10
            reasons.append("directly addresses query")
        elif query_answer_overlap > 0.3:
            score += 5
        else:
            score -= 5
            reasons.append("may not fully address query")
        
        # 3. Answer specificity - penalize generic answers
        generic_phrases = ['the context', 'the document', 'it states', 'according to', 'as mentioned']
        generic_count = sum(1 for phrase in generic_phrases if phrase in answer.lower())
        
        if generic_count > 2:
            score -= 10
            reasons.append("contains generic phrasing")
        
        # 4. Answer completeness - length relative to query complexity
        query_complexity = len(query.split())
        answer_length = len(answer)
        
        if query_complexity > 10 and answer_length > 200:
            score += 10
            reasons.append("comprehensive answer for complex query")
        elif query_complexity > 10 and answer_length < 100:
            score -= 15
            reasons.append("brief answer for complex question")
        elif query_complexity <= 5 and answer_length > 150:
            score += 5
            reasons.append("detailed answer")
        
        # 5. Uncertainty phrases - cap confidence
        uncertainty_phrases = ['not contain', 'does not specify', 'unclear', 'not mentioned', 'not provided', 'no information']
        if any(phrase in answer.lower() for phrase in uncertainty_phrases):
            score = min(score, 55)
            reasons.append("information not found in context")
        
        # 6. Context quality
        if len(context) > 1500:
            score += 8
            reasons.append("substantial context")
        elif len(context) > 800:
            score += 4
        elif len(context) < 300:
            score -= 5
            reasons.append("limited context")
        
        # 7. Answer truncation indicator (often means incomplete thought)
        if answer.endswith(('...', 'etc', 'and more')):
            score -= 5
            reasons.append("answer may be incomplete")
        
        # Build confidence reason
        main_reason = reasons[0] if reasons else "standard response quality"
        if len(reasons) > 1:
            confidence_reason = f"{main_reason.capitalize()} ({', '.join(reasons[1:3])})"
        else:
            confidence_reason = main_reason.capitalize()
        
        return (max(35, min(95, score)), confidence_reason)
    
    def _get_first_sentence(self, text: str) -> str:
        """
        Extract the first sentence for summary.
        Simple and clean.
        """
        if not text:
            return text
        
        # Find first sentence ending
        for i, char in enumerate(text):
            if char in '.!?' and (i + 1 >= len(text) or text[i + 1] == ' '):
                return text[:i + 1]
        
        # No sentence ending found, return first 150 chars
        if len(text) > 150:
            return text[:150] + '...'
        
        return text
    
    def _is_out_of_context_answer(self, answer: str) -> bool:
        """
        Detect if the answer indicates the information is not in the context.
        """
        answer_lower = answer.lower()
        
        # Phrases that indicate the answer is not in the context
        out_of_context_phrases = [
            "i don't have",
            "i do not have",
            "not mentioned",
            "not specified",
            "not provided",
            "not contain",
            "does not contain",
            "no information",
            "cannot find",
            "not found",
            "not available",
            "context does not",
            "the context does not",
            "not in the context"
        ]
        
        # If answer is very short and contains uncertainty phrases
        if len(answer) < 100:
            for phrase in out_of_context_phrases:
                if phrase in answer_lower:
                    return True
        
        return False
    
    def _ensure_complete_sentence(self, text: str) -> str:
        """
        Ensure the answer ends with proper punctuation and doesn't cut off mid-sentence.
        """
        if not text:
            return text
        
        # If ends with proper punctuation, return as-is
        if text[-1] in '.!?':
            return text
        
        # Find the last sentence-ending punctuation
        last_period = text.rfind('.')
        last_exclamation = text.rfind('!')
        last_question = text.rfind('?')
        
        last_punct = max(last_period, last_exclamation, last_question)
        
        # If we found punctuation, truncate to the last complete sentence
        if last_punct > len(text) * 0.5:  # Only if it's past halfway (avoid losing too much)
            return text[:last_punct + 1]
        
        # If no good punctuation found, add a period
        return text + '.'
    
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
