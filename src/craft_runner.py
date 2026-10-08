"""CRAFT runner: the REALM-Bench adapter.

Converts a REALM-Bench TaskDefinition into the CRAFT input format, drives the
meta layer (whose constraints are retrieved from the RAGFlow knowledge base),
converts the result back into the benchmark output schema, and registers the
ablation configurations (CRAFT-C / -P / -O / -A / -full)."""

import sys
import os
import time
import traceback
from typing import Dict, List, Any

_craft_src = os.path.join(os.path.dirname(os.path.abspath(__file__)))
if _craft_src not in sys.path:
    sys.path.insert(0, _craft_src)


class CRAFTRunner:
    def __init__(self,
                 top_k: int = 15,
                 planning_mode: str = "collaborative",
                 enable_verification: bool = True,
                 enable_optimization: bool = True,
                 max_feedback_iterations: int = 2,
                 constraint_filter: set = None,
                 verbose: bool = True):
        self.top_k = top_k
        self.planning_mode = planning_mode
        self.enable_verification = enable_verification
        self.enable_optimization = enable_optimization
        self.max_feedback_iterations = max_feedback_iterations
        self.constraint_filter = constraint_filter
        self.verbose = verbose

        self._meta_layer = None
        self._input_layer = None
        self._auditor = None

        self.execution_times: List[float] = []
        self.memory_usage: List[Dict] = []
        self.token_usage: Dict[str, int] = {}

    def __call__(self, task_definition) -> Dict[str, Any]:
        start_time = time.time()

        try:
            task_input = self._task_to_craft_input(task_definition)

            result = self._run_craft_pipeline(task_input, task_definition)

            satisfied_constraints, compliance_rate, auditor_addressed = \
                self._extract_satisfied_constraints(result, task_definition)

            result["_compliance_rate"] = compliance_rate
            result["_auditor_addressed"] = auditor_addressed

            execution_time = time.time() - start_time
            self.execution_times.append(execution_time)

            benchmark_result = self._craft_output_to_benchmark(
                result, task_definition, satisfied_constraints, execution_time)

            return benchmark_result

        except Exception as e:
            print(f"[CRAFTRunner] run failed: {e}")
            traceback.print_exc()
            return self._empty_result(task_definition)

    def _task_to_craft_input(self, td) -> Dict[str, Any]:
        goals = [(g.goal_id, g.description) for g in td.goals]

        constraints = []

        resources = {}
        for c in td.constraints:
            params = c.parameters
            if "resources" in params:
                resources.update(params["resources"])

        regions = td.resources.get("regions", [])

        vehicle_capacity = {}
        for v in td.resources.get("vehicles", []):
            vtype = "helicopter" if "helicopter" in v.get("id", "") else "truck"
            vehicle_capacity[f"{vtype}_max_kg_per_trip"] = v.get("capacity", 0)

        return {
            "task_id": td.task_id,
            "task_name": td.name,
            "task_description": td.description,
            "goals": goals,
            "constraints": constraints,
            "resources": resources,
            "regions": regions,
            "vehicle_capacity": vehicle_capacity,
            "total_population": sum(r.get("population", 0) for r in regions),
        }

    def _run_craft_pipeline(self, task_input: Dict[str, Any],
                             td) -> Dict[str, Any]:
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[CRAFTRunner] task: {task_input['task_id']} — {task_input['task_name']}")
            print(f"[CRAFTRunner] regions: {len(task_input['regions'])}, "
                  f"constraints: {len(task_input['constraints'])}")
            print(f"{'='*60}")

        meta = self._get_meta_layer()

        task_context = {
            "total_population": task_input["total_population"],
            "regions": task_input["regions"],
            "total_resources": task_input["resources"],
        }

        result = meta.orchestrate(
            task_description=task_input["task_description"],
            goals=task_input["goals"],
            resources=task_input["resources"],
            constraints=task_input["constraints"],
            regions=task_input["regions"],
            vehicle_capacity=task_input["vehicle_capacity"],
            planning_mode=self.planning_mode,
            enable_verification=self.enable_verification,
            enable_optimization=self.enable_optimization,
            task_context=task_context,
            verbose=self.verbose,
        )

        result["_task_input"] = task_input

        return result

    def _extract_satisfied_constraints(self, result: Dict[str, Any],
                                        td) -> List[str]:
        try:
            from verification.auditor import ConstraintAuditor

            constraints = result.get("_retrieved_constraints") or []
            if not constraints:
                if self.verbose:
                    print("[CRAFTRunner] no constraint was retrieved from RAGFlow, "
                          "skipping the final audit")
                return (result.get("satisfied_constraints", []), 0, [])
            task_context = {
                "total_population": sum(
                    r.get("population", 0)
                    for r in td.resources.get("regions", [])
                ),
                "regions": td.resources.get("regions", []),
            }

            auditor = ConstraintAuditor()
            if self.constraint_filter:
                constraints = [c for c in constraints
                              if c.get("id", c.get("constraint_id", "")) in self.constraint_filter]
            audit = auditor.audit(result, constraints, task_context)
            auditor_addressed = ([p["constraint_id"] for p in audit.get("passed", [])] +
                                [v["constraint_id"] for v in audit.get("violated", [])])
            result["_audit_undefined"] = [u["constraint_id"]
                                          for u in audit.get("undefined", [])]
            result["_audit_undefined_count"] = audit.get("undefined_count", 0)
            return ([p["constraint_id"] for p in audit.get("passed", [])],
                    audit.get("compliance_rate", 0),
                    auditor_addressed)

        except Exception as e:
            if self.verbose:
                print(f"[CRAFTRunner] constraint verification failed: {e}")
            return (result.get("satisfied_constraints", []), 0, [])

    def _craft_output_to_benchmark(self, result: Dict[str, Any],
                                     td, satisfied_constraints: List[str],
                                     execution_time: float) -> Dict[str, Any]:
        task_completion_times = {}
        for t in result.get("schedule", []):
            tid = t.get("task_id", "")
            if tid:
                task_completion_times[tid] = t.get("end_time", 0)

        goals_achieved = result.get("goals", [])

        return {
            "goals": goals_achieved,
            "goals_achieved": goals_achieved,
            "addressed_constraints": result.get("addressed_constraints", []),
            "schedule": result.get("schedule", []),
            "allocated_resources": result.get("allocated_resources", {}),
            "assigned_personnel": result.get("assigned_personnel", {}),
            "region_response_times": result.get("region_response_times", {}),
            "task_completion_times": task_completion_times,
            "satisfied_constraints": satisfied_constraints,
            "constraints_satisfied": satisfied_constraints,

            "_craft_meta": result.get("_craft_meta", {}),
            "_compliance_rate": result.get("_compliance_rate"),
            "_improvement_pct": result.get("_improvement_pct"),
            "_plan_summary": result.get("plan_summary", result.get("plan", "")),
            "_mode": result.get("_mode", "collaborative"),

            "resource_usage": {
                "execution_time": execution_time,
                "memory_usage": self.memory_usage,
                "token_usage": self.token_usage,
            },
        }

    def _empty_result(self, td) -> Dict[str, Any]:
        return {
            "goals": [],
            "goals_achieved": [],
            "schedule": [],
            "allocated_resources": {},
            "assigned_personnel": {},
            "region_response_times": {},
            "task_completion_times": {},
            "satisfied_constraints": [],
            "constraints_satisfied": [],
            "_craft_meta": {"error": "pipeline_failed"},
            "resource_usage": {
                "execution_time": 0,
                "memory_usage": [],
                "token_usage": {},
            },
        }

    def _get_meta_layer(self):
        if self._meta_layer is None:
            from meta.layer import CRAFTMetaLayer
            self._meta_layer = CRAFTMetaLayer(
                top_k=self.top_k,
                feedback_max_iterations=self.max_feedback_iterations,
            )
        return self._meta_layer

    def _get_eval_constraints(self, td) -> List[Dict]:
        constraints = []
        for c in td.constraints:
            constraints.append({
                "id": c.constraint_id,
                "type": c.constraint_type,
                "desc": c.description,
                "params": c.parameters,
            })
        return constraints

    def run_ablation(self, td,
                     config: str = None,
                     skip_verification: bool = False,
                     skip_optimization: bool = False,
                     planning_mode: str = None) -> Dict[str, Any]:
        if config:
            preset = ABLATION_CONFIGS.get(config)
            if preset is None:
                raise ValueError(
                    f"unknown ablation configuration: {config}; available: {list(ABLATION_CONFIGS)}")
            planning_mode = preset["planning_mode"]
            skip_verification = not preset["enable_verification"]
            skip_optimization = not preset["enable_optimization"]

        orig_verify = self.enable_verification
        orig_opt = self.enable_optimization
        orig_mode = self.planning_mode

        self.enable_verification = not skip_verification
        self.enable_optimization = not skip_optimization
        if planning_mode:
            self.planning_mode = planning_mode

        try:
            result = self.__call__(td)
        finally:
            self.enable_verification = orig_verify
            self.enable_optimization = orig_opt
            self.planning_mode = orig_mode

        return result


ABLATION_CONFIGS: Dict[str, Dict[str, Any]] = {
    "CRAFT-C": {
        "planning_mode": "global",
        "enable_verification": False,
        "enable_optimization": False,
        "activated_agents": ["constraint_curator"],
    },
    "CRAFT-P": {
        "planning_mode": "decomposed",
        "enable_verification": False,
        "enable_optimization": False,
        "activated_agents": ["constraint_curator", "regional_planner",
                             "resource_arbitrator"],
    },
    "CRAFT-O": {
        "planning_mode": "collaborative",
        "enable_verification": False,
        "enable_optimization": False,
        "activated_agents": ["constraint_curator", "regional_planner",
                             "resource_arbitrator", "chief_orchestrator"],
    },
    "CRAFT-A": {
        "planning_mode": "collaborative",
        "enable_verification": True,
        "enable_optimization": False,
        "activated_agents": ["constraint_curator", "regional_planner",
                             "resource_arbitrator", "chief_orchestrator",
                             "constraint_auditor"],
    },
    "CRAFT-full": {
        "planning_mode": "collaborative",
        "enable_verification": True,
        "enable_optimization": True,
        "activated_agents": ["constraint_curator", "regional_planner",
                             "resource_arbitrator", "chief_orchestrator",
                             "constraint_auditor", "plan_optimizer",
                             "consensus_builder"],
    },
}


def list_ablation_configs() -> Dict[str, Dict[str, Any]]:
    return {name: dict(cfg) for name, cfg in ABLATION_CONFIGS.items()}


def create_craft_runner(**kwargs) -> CRAFTRunner:
    return CRAFTRunner(**kwargs)


if __name__ == "__main__":
    import json

    class MockGoal:
        def __init__(self, gid, desc):
            self.goal_id = gid
            self.description = desc

    class MockConstraint:
        def __init__(self, cid, ctype, desc, params):
            self.constraint_id = cid
            self.constraint_type = ctype
            self.description = desc
            self.parameters = params

    class MockTaskDef:
        def __init__(self):
            self.task_id = "P7"
            self.name = "Disaster Relief Deployment (Mock)"
            self.description = (
                "Disaster relief scenario: two affected regions need resource allocation. "
                "Region1 critical severity, many collapsed buildings, 3000 people; "
                "Region2 urgent severity, water and power cut, 1500 people. "
                "Allocate medical supplies, food, drinking water and shelter materials, "
                "deploy medical staff and transport vehicles, and respond within the "
                "golden rescue window."
            )
            self.goals = [
                MockGoal("G1", "Allocate medical supplies and food to both regions"),
                MockGoal("G2", "Guarantee the minimum daily drinking water supply per person"),
                MockGoal("G3", "Reach every region within the response deadline"),
            ]
            self.constraints = [
                MockConstraint("C01", "resource", "Minimum drinking water guarantee",
                              {"ratios": {"water_liters_per_person_per_day_standard": 15}}),
                MockConstraint("C03", "resource", "Total medical supply cap",
                              {"resources": {"medical_supplies_kg": 2500}}),
                MockConstraint("C69", "deadline", "Disaster response deadline tiers",
                              {"deadlines": {"critical": 2, "urgent": 6}}),
            ]
            self.resources = {
                "regions": [
                    {"id": "region1", "severity": "critical", "population": 3000},
                    {"id": "region2", "severity": "urgent", "population": 1500},
                ],
                "vehicles": [
                    {"id": "truck1", "capacity": 5000},
                    {"id": "helicopter1", "capacity": 1000},
                ],
            }
            self.disruption_scenarios = []

    print("CRAFTRunner standalone test")
    print("=" * 60)

    runner = CRAFTRunner(top_k=10, verbose=True)
    mock_td = MockTaskDef()
    result = runner(mock_td)

    print("\nResult summary:")
    print(f"  goals: {result.get('goals', [])}")
    print(f"  resource allocation: {json.dumps(result.get('allocated_resources', {}), ensure_ascii=False)[:200]}")
    print(f"  personnel allocation: {json.dumps(result.get('assigned_personnel', {}), ensure_ascii=False)[:200]}")
    print(f"  scheduled tasks: {len(result.get('schedule', []))}")
    print(f"  constraints satisfied: {len(result.get('satisfied_constraints', []))}")
    print(f"  execution time: {result.get('resource_usage', {}).get('execution_time', 0):.1f}s")
    print(f"  CRAFT meta: {json.dumps(result.get('_craft_meta', {}), ensure_ascii=False)[:200]}")
