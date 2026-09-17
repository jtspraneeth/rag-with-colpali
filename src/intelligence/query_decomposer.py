import re
from typing import List
from pydantic import BaseModel

class SubQuestion(BaseModel):
    id: int
    sub_query: str
    target_entity: str

class DecompositionResult(BaseModel):
    original_query: str
    is_decomposed: bool
    sub_questions: List[SubQuestion]
    explanation: str

class QueryDecomposer:
    """Decomposes comparative and multi-hop questions into independently retrievable sub-questions."""

    def decompose(self, query: str) -> DecompositionResult:
        q_lower = query.lower()

        # Check for comparison across years (e.g. "Compare revenue between 2023 and 2025")
        years = re.findall(r'\b(20\d{2}|19\d{2})\b', query)
        if len(years) >= 2 and any(w in q_lower for w in ["compare", "versus", "vs", "difference", "growth", "between"]):
            sub_q = []
            y1, y2 = years[0], years[1]
            
            # Identify metrics (revenue, employees, income, etc.)
            metrics = []
            if "revenue" in q_lower or "sales" in q_lower:
                metrics.append("revenue")
            if "employee" in q_lower or "headcount" in q_lower or "staff" in q_lower:
                metrics.append("employee count")
            if "income" in q_lower or "profit" in q_lower:
                metrics.append("net income")

            if not metrics:
                metrics = ["financial data and metrics"]

            sub_id = 1
            for m in metrics:
                sub_q.append(SubQuestion(id=sub_id, sub_query=f"What was the {m} in {y1}?", target_entity=f"{m} {y1}"))
                sub_id += 1
                sub_q.append(SubQuestion(id=sub_id, sub_query=f"What was the {m} in {y2}?", target_entity=f"{m} {y2}"))
                sub_id += 1

            return DecompositionResult(
                original_query=query,
                is_decomposed=True,
                sub_questions=sub_q,
                explanation=f"Decomposed query into {len(sub_q)} sub-questions targeting {y1} and {y2} independently."
            )

        # Non-decomposed query
        return DecompositionResult(
            original_query=query,
            is_decomposed=False,
            sub_questions=[SubQuestion(id=1, sub_query=query, target_entity="main_query")],
            explanation="Query is direct and does not require sub-question decomposition."
        )
