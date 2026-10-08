"""Optimization-layer orchestrator (M7).

Runs the Plan Optimizer agent on a verified plan; the adjusted schedule is
adopted only after the re-audit performed in the meta layer."""

from typing import Dict, Any

try:
    from .optimizer import create_optimizer
except ImportError:
    from optimization.optimizer import create_optimizer


class CRAFTOptimizationLayer:
    def __init__(self,
                 llm_engine=None,
                 input_layer=None):
        self.optimizer = create_optimizer(llm_engine=llm_engine)

        self.stats = {
            "total_optimizations": 0,
            "total_improvement_pct": [],
        }

    def optimize(self, plan: Dict[str, Any],
                 resources: Dict[str, float] = None,
                 vehicle_capacity: Dict[str, Any] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        self.stats["total_optimizations"] += 1

        if verbose:
            print(f"\n{'='*50}")
            print("[Optimisation layer] starting the optimisation (M7 Plan Optimizer)")
            print("=" * 50)

        result = self.optimizer.optimize(
            plan=plan,
            resources=resources,
            vehicle_capacity=vehicle_capacity,
            verbose=verbose,
        )

        improvement = result.get("improvement_pct", 0)
        self.stats["total_improvement_pct"].append(improvement)

        if verbose:
            print(f"[Optimisation layer] optimisation done: makespan improved by {improvement:.1f}%")

        return result

    def get_stats(self) -> Dict[str, Any]:
        avg_improvement = (
            sum(self.stats["total_improvement_pct"]) /
            max(len(self.stats["total_improvement_pct"]), 1)
        )
        return {
            **self.stats,
            "avg_makespan_improvement_pct": round(avg_improvement, 1),
        }


def create_optimization_layer(
        llm_engine=None,
        input_layer=None,
) -> CRAFTOptimizationLayer:
    return CRAFTOptimizationLayer(
        llm_engine=llm_engine,
        input_layer=input_layer,
    )
