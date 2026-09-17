import re
from typing import List

class ClaimExtractor:
    """Extracts atomic factual claims from generated LLM answers."""
    
    def extract_claims(self, answer_text: str) -> List[str]:
        """Splits answer text into clean atomic sentence claims."""
        if not answer_text or not answer_text.strip():
            return []

        # Remove citation markers like [1], [Annual Report - Page 12]
        clean_text = re.sub(r'\[.*?\]', '', answer_text)

        # Split into sentences
        raw_sentences = re.split(r'(?<=[.!?])\s+', clean_text)
        
        claims = []
        for sent in raw_sentences:
            s = sent.strip()
            # Ignore meta sentences like "Based on the documents..." or very short phrases
            if len(s) > 15 and not s.lower().startswith(("insufficient evidence", "here is the answer")):
                claims.append(s)

        return claims if claims else [clean_text.strip()]
