"""Input-layer orchestrator (M2 + M3), driven by the Constraint Curator agent.

Runs deterministic type pre-classification followed by hybrid retrieval, and
returns the shared Top-K constraint rules (formatted text plus raw records)."""

from typing import List, Dict, Any, Optional

try:
    from .preclassifier import ConstraintPreClassifier
    from .retriever import (
        HybridRetriever,
        create_hybrid_retriever,
    )
    from ..config import DEFAULT_TOP_K
except ImportError:
    from input.preclassifier import ConstraintPreClassifier
    from input.retriever import (
        HybridRetriever,
        create_hybrid_retriever,
    )
    from config import DEFAULT_TOP_K


class CRAFTInputLayer:
    def __init__(self,
                 preclassifier: Optional[ConstraintPreClassifier] = None,
                 retriever: Optional[HybridRetriever] = None,
                 top_k: int = DEFAULT_TOP_K):
        self.preclassifier = preclassifier or ConstraintPreClassifier()
        self.retriever = retriever or create_hybrid_retriever()
        self.top_k = top_k

        self.stats = {
            "total_calls": 0,
            "preclassifier_reduction": [],
            "retrieval_scores": [],
        }

        self._cache_query = None
        self._cache_raw_results = []

    def retrieve(self, task_description: str,
                 top_k: Optional[int] = None,
                 verbose: bool = True) -> str:
        top_k = top_k or self.top_k
        self.stats["total_calls"] += 1

        if verbose:
            print(f"\n{'='*60}")
            print("[CRAFT input layer] retrieval start")
            print(f"[CRAFT input layer] task description: {task_description[:100]}...")
            print(f"{'='*60}")

        target_types = self.preclassifier.select_types(task_description)

        top_constraints = self.retriever.retrieve(
            query=task_description,
            constraints=None,
            top_k=top_k,
            candidate_types=target_types,
            verbose=verbose,
        )

        self._cache_query = task_description
        self._cache_raw_results = top_constraints

        text = self._format_for_prompt(top_constraints)

        if verbose:
            avg_score = (sum(c["_retrieval_score"] for c in top_constraints) /
                         len(top_constraints)) if top_constraints else 0
            self.stats["retrieval_scores"].append(avg_score)
            print(f"[CRAFT input layer] retrieval done: {len(top_constraints)} constraints formatted")

        return text

    def retrieve_raw(self, task_description: str,
                     top_k: Optional[int] = None,
                     verbose: bool = True) -> List[Dict[str, Any]]:
        top_k = top_k or self.top_k

        if self._cache_query == task_description and self._cache_raw_results:
            return self._cache_raw_results

        target_types = self.preclassifier.select_types(task_description)
        return self.retriever.retrieve(
            query=task_description, constraints=None,
            top_k=top_k, candidate_types=target_types, verbose=verbose,
        )

    def _format_for_prompt(self, constraints: List[Dict[str, Any]]) -> str:
        if not constraints:
            return "(no relevant constraint retrieved)"

        import json

        lines = ["[Relevant constraints retrieved from the knowledge base - follow them strictly while planning]", ""]

        for i, c in enumerate(constraints, 1):
            cid = c.get("id", "")
            ctype = c.get("type", "")
            desc = c.get("desc", "")
            source = c.get("source", "")
            params = c.get("params", {})

            lines.append(f"constraint {i}: [{cid}] [{ctype}] {desc}")
            lines.append(f"  params: {json.dumps(params, ensure_ascii=False)}")
            lines.append(f"  source: {source}")
            lines.append("")

        return "\n".join(lines)

    def get_stats(self) -> Dict[str, Any]:
        avg_reduction = (sum(self.stats["preclassifier_reduction"]) /
                         max(len(self.stats["preclassifier_reduction"]), 1))
        avg_score = (sum(self.stats["retrieval_scores"]) /
                     max(len(self.stats["retrieval_scores"]), 1))

        return {
            "total_calls": self.stats["total_calls"],
            "avg_preclassifier_reduction_pct": round(avg_reduction * 100, 1),
            "avg_retrieval_score": round(avg_score, 4),
            "retriever_stats": self.retriever.stats,
        }
