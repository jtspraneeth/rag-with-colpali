import re
from typing import List
from src.ingestion.metadata import RetrievalResult

class TemporalFilter:
    """Filters or boosts document chunks matching temporal queries (years, FY periods, dates)."""
    
    def filter_by_temporal_query(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """Boosts or filters candidates matching date/year keywords in query."""
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', query)
        if not years:
            return results

        target_year = years[0]
        boosted_results: List[RetrievalResult] = []

        for r in results:
            period = str(r.metadata.get("reporting_period", "")).lower()
            doc_date = str(r.metadata.get("document_date", "")).lower()
            text_lower = r.text.lower()

            r_dict = r.model_dump()
            score = r.score

            if target_year in period or target_year in doc_date or target_year in text_lower:
                score = min(1.0, score + 0.15)
                r_dict["metadata"]["temporal_match"] = True
            else:
                score = max(0.0, score - 0.05)
                r_dict["metadata"]["temporal_match"] = False

            r_dict["score"] = round(float(score), 4)
            boosted_results.append(RetrievalResult(**r_dict))

        boosted_results.sort(key=lambda x: x.score, reverse=True)
        return boosted_results
