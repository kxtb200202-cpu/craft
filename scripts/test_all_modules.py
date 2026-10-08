"""Component self-test for the verification, optimization, meta and agent modules."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
os.environ["PYTHONIOENCODING"] = "utf-8"

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  [PASS] {name}")
    else:
        failed += 1
        print(f"  [FAIL] {name} -- {detail}")

print("=" * 60)
print("1. M5: Constraint Auditor")
print("=" * 60)

from verification.auditor import ConstraintAuditor

TEST_CONSTRAINTS = [
    {"id": "C01", "type": "resource", "desc": "Minimum drinking water guarantee",
     "source": "Sphere Handbook 2018", "slug": "water_minimum_standard",
     "params": {"ratios": {"water_liters_per_person_per_day_standard": 15}}},
    {"id": "C03", "type": "resource", "desc": "Total medical supply cap",
     "source": "Sphere Handbook 2018", "slug": "medical_supply_cap",
     "params": {"resources": {"medical_supplies_kg": 2500}}},
    {"id": "C05", "type": "resource", "desc": "Blanket provision",
     "source": "Sphere Handbook 2018", "slug": "blanket_per_person",
     "params": {"ratios": {"blankets_per_person": 1}}},
    {"id": "C32", "type": "capacity", "desc": "Medical staff establishment",
     "source": "WHO", "slug": "medical_staff_limit",
     "params": {"personnel": {"doctors": 40}}},
    {"id": "C69", "type": "deadline", "desc": "Disaster response deadline tiers",
     "source": "IASC", "slug": "response_deadline_tiers",
     "params": {"deadlines": {"critical": 2, "urgent": 6}}},
    {"id": "C45", "type": "logic", "desc": "Assessment before procurement",
     "source": "IASC", "slug": "assessment_before_procurement",
     "params": {"logic": "Procurement must not start before the rapid assessment."}},
]
print(f"[test] inline constraint fixture: {len(TEST_CONSTRAINTS)} rules")

test_plan = {
    "allocated_resources": {"medical_supplies": 3000, "food": 8000, "water": 5000, "blankets": 5000, "tents": 800, "shelter_materials": 0},
    "assigned_personnel": {"doctors": 25, "nurses": 50},
    "region_response_times": {"region1": 4, "region2": 6},
}
task_ctx = {"total_population": 4500, "regions": [
    {"id": "region1", "severity": "critical", "population": 3000},
    {"id": "region2", "severity": "urgent", "population": 1500},
]}

auditor = ConstraintAuditor()
report = auditor.audit(test_plan, TEST_CONSTRAINTS, task_ctx)

check("Has total/passed/violated", all(k in report for k in ["total", "passed", "violated"]))
check("compliance_rate is float", isinstance(report["compliance_rate"], float))
check("Detects violations", len(report["violated"]) > 0, f"found {len(report['violated'])}")
check("Has critical_violations", len(report.get("critical_violations", [])) > 0)
check("format_report works", len(auditor.format_report(report)) > 100)
check("Deterministic without LLM (no llm_engine)", auditor.llm_engine is None)

if report["violated"]:
    print(f"  Violations: {[v['constraint_id'] for v in report['violated']]}")
print(f"  Compliance: {report['compliance_rate']:.1%}")

print("\n" + "=" * 60)
print("2. M6: Feedback Correction Loop")
print("=" * 60)

from verification.feedback import FeedbackLoop

fl = FeedbackLoop(auditor=auditor, max_iterations=2)
check("max_iterations defaults to 2", fl.max_iterations == 2)
cp = fl._build_correction_prompt(report, test_plan, "test task")
check("Builds correction prompt", len(cp) > 50)

print("\n" + "=" * 60)
print("3. M7: Plan Optimizer")
print("=" * 60)

from optimization.optimizer import PlanOptimizer, create_optimizer

plan_sched = {
    "plan": "test",
    "schedule": [
        {"task_id": "T1", "name": "medical transport", "start_time": 0, "end_time": 4, "assigned_to": "truck1", "region": "region1"},
        {"task_id": "T2", "name": "water transport", "start_time": 4, "end_time": 8, "assigned_to": "truck1", "region": "region1"},
        {"task_id": "T3", "name": "food transport", "start_time": 0, "end_time": 3, "assigned_to": "truck2", "region": "region2"},
        {"task_id": "T4", "name": "tent setup", "start_time": 8, "end_time": 14, "assigned_to": "shelter_team", "region": "region1"},
        {"task_id": "T5", "name": "search rescue", "start_time": 0, "end_time": 6, "assigned_to": "rescue_team", "region": "region1"},
    ],
    "allocated_resources": {"medical_supplies": 2000, "water": 8000, "food": 6000},
}

opt = create_optimizer()
check("create_optimizer returns PlanOptimizer without LLM", isinstance(opt, PlanOptimizer))
opt_r = opt.optimize(plan_sched, verbose=False)

check("Has optimized_plan", "optimized_plan" in opt_r)
check("Has improvement_pct", "improvement_pct" in opt_r)
check("Has optimizations_applied", len(opt_r.get("optimizations_applied", [])) > 0)
check("Schedule size preserved", len(opt_r["optimized_plan"]["schedule"]) == 5)

print(f"  Makespan: {opt_r['original_makespan']}h -> {opt_r['optimized_makespan']}h")
print(f"  Strategies: {opt_r['optimizations_applied']}")

print("\n" + "=" * 60)
print("4. Meta Layer: Chief Orchestrator + Consensus Builder")
print("=" * 60)

from meta.chief_orchestrator import ChiefOrchestrator
from meta.consensus_builder import ConsensusBuilder

co = ChiefOrchestrator()

d1 = co.design("single region flood 5000 people", constraints_count=5)
dec1 = d1["assembly_decision"]
check("Low complexity -> pi_light", dec1.mode.value == "light", f"got {dec1.mode.value}")
check("Low -> no verify", not dec1.activate_verification)
check("Low -> global plan", dec1.planning_mode == "global")

d2 = co.design("earthquake Region1 critical 3000 Region2 urgent 1500 Region3 normal 1000",
               constraints_count=50, optimize_makespan=True)
dec2 = d2["assembly_decision"]
check("High complexity -> pi_full", dec2.mode.value == "full")
check("High -> collaborative", dec2.planning_mode == "collaborative")
check("High -> verify ON", dec2.activate_verification)
check("High -> optimize ON", dec2.activate_optimization)
check("4 execution steps (input/planning/verification/optimization)",
      len(d2["execution_plan"]) == 4, f"got {len(d2['execution_plan'])}")
check("No disruption step", all("disruption" not in s["action"] for s in d2["execution_plan"]))

builder = ConsensusBuilder()
versions = [
    {"source": "planning", "plan": {"plan": "v1", "allocated_resources": {"water": 8000}, "assigned_personnel": {"doctors": 18}, "schedule": [{"task_id": "T1", "start_time": 0, "end_time": 4}], "region_response_times": {"region1": 2}, "goals_achieved": ["G1"]}},
    {"source": "verification", "plan": {"plan": "v2", "allocated_resources": {"water": 10000}, "assigned_personnel": {"doctors": 20}, "schedule": [{"task_id": "T1", "start_time": 0, "end_time": 3}], "region_response_times": {"region1": 2}}},
]
merged = builder.build(versions, chief_orchestrator=co, verbose=False)
check("Merged has resources", len(merged.get("allocated_resources", {})) > 0)
check("Picks latest values", merged["allocated_resources"].get("water") == 10000)
check("Has _meta", "_meta" in merged)

bench = builder.format_for_benchmark(merged)
check("Benchmark has goals", len(bench.get("goals", [])) > 0)
check("Benchmark has schedule", len(bench.get("schedule", [])) > 0)

check("Resolves resource conflict", co.resolve_dispute({"type": "resource_conflict", "value_a": 5000, "value_b": 8000, "conservative_value": 5000})["resolution"] == "conservative")
check("Auditor wins constraint dispute", co.resolve_dispute({"type": "constraint_disagreement"})["resolution"] == "auditor_wins")
check("Design report generated", len(co.generate_design_report(d2)) > 200)

print(f"  Mode: pi_{dec2.mode.value}, Steps: {[s['layer'] for s in d2['execution_plan']]}")

print("\n" + "=" * 60)
print("5. Roles: 7 agents + assembly templates")
print("=" * 60)

from agents.roles import ALL_ROLES, get_assembly_templates
check("Exactly 7 roles", len(ALL_ROLES) == 7, f"got {len(ALL_ROLES)}")
ids = {r.agent_id for r in ALL_ROLES}
check("Chief orchestrator present", "chief_orchestrator" in ids)
check("No out-of-paper roles", not ids & {"global_strategist", "cross_region_negotiator", "disruption_adapter"})
tpl = get_assembly_templates()
check("Templates light/standard/full", set(tpl.keys()) == {"light", "standard", "full"})

print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed+failed}")
print("=" * 60)
if failed > 0:
    sys.exit(1)
