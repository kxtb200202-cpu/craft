"""Component self-test for the planning layer (M1 decomposition, M4 planning)."""

import sys
import os
import json

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from planning.decomposer import PlanDecomposer
from planning.llm_engine import create_llm_engine
from planning.layer import create_planning_layer
from input.layer import CRAFTInputLayer


def test_module3_decompose():
    print("\n" + "=" * 60)
    print("Test M1: task decomposition (Chief Orchestrator duty)")
    print("=" * 60)

    decomposer = PlanDecomposer()

    test_cases = [
        "Earthquake disaster relief: Region1 critical severity 3000 people, "
        "Region2 urgent severity 1500 people",
        "Flood relief: area A (extreme, 5000 people), area B (moderate, 2000 people)",
        "Two affected areas: Region alpha (severe, 8000 people), "
        "Region beta (normal, 3000 people)",
    ]

    for desc in test_cases:
        print(f"\nDescription: {desc[:60]}...")
        regions = decomposer.extract_regions(desc)
        for r in regions:
            print(f"  → {r['id']}: severity={r['severity']}, population={r['population']}")
        sub_tasks = decomposer.decompose(desc)
        print(f"  decomposed into {len(sub_tasks)} subtasks")
        for st in sub_tasks:
            print(f"    subtask: {st['region']['id']}")


def test_module4_plan():
    print("\n" + "=" * 60)
    print("Test M4: regional planning engine (LLM planning)")
    print("=" * 60)

    engine = create_llm_engine()

    task = """Earthquake relief: Region1 critical 3000 people, Region2 urgent 1500 people.
Allocate medical supplies, food and drinking water.
Resources: medical_supplies=2500, food=8000, water=10000"""

    constraints = """[Constraints]
[C01] [resource] Minimum drinking water guarantee: at least 15 litres per person per day
[C03] [resource] Total medical supply cap: at most 2500 kg
[C69] [deadline] Disaster response tiers: critical <= 2 hours, urgent <= 6 hours"""

    print("\nCalling the LLM to generate a plan...")
    result = engine.plan(
        task_description=task,
        constraints_text=constraints,
        regions=[
            {"id": "region1", "severity": "critical", "population": 3000},
            {"id": "region2", "severity": "urgent", "population": 1500},
        ],
        verbose=True,
    )

    print(f"\nplan summary: {result.get('plan', 'N/A')}")
    print(f"resource allocation: {json.dumps(result.get('allocated_resources', {}), ensure_ascii=False)}")
    print(f"personnel allocation: {json.dumps(result.get('assigned_personnel', {}), ensure_ascii=False)}")
    print(f"scheduled tasks: {len(result.get('schedule', []))}")
    print(f"region response times: {result.get('region_response_times', {})}")

    return result


def test_full_planning_layer(with_input_layer=True, mode="decomposed"):
    print("\n" + "=" * 60)
    mode_label = {"global": "global planning", "decomposed": "decomposed planning",
                  "collaborative": "collaborative planning"}.get(mode, mode)
    print(f"Test CRAFT planning layer: {mode_label} mode")
    print("=" * 60)

    input_layer = None
    if with_input_layer:
        input_layer = CRAFTInputLayer(top_k=15)

    planning_layer = create_planning_layer(
        input_layer=input_layer,
        top_k=15,
    )

    task = (
        "Disaster relief scenario: two affected regions need resource allocation. "
        "Region1 critical severity, many collapsed buildings, 3000 people; "
        "Region2 urgent severity, water and power cut, 1500 people. "
        "Allocate medical supplies, food, drinking water and shelter materials, "
        "deploy medical staff and transport vehicles (trucks and helicopters), "
        "and respond within the golden rescue window."
    )

    goals = [
        ("G1", "Allocate enough medical supplies and food to both regions"),
        ("G2", "Guarantee the minimum daily drinking water supply per person"),
        ("G3", "Reach every region within the response deadline"),
        ("G4", "Keep the allocation fair and non-discriminatory"),
    ]

    resources = {
        "medical_supplies": 2500,
        "food": 8000,
        "water": 10000,
        "shelter_materials": 2000,
        "tents": 800,
        "blankets": 5000,
        "first_aid_kits": 1000,
        "diesel": 800,
        "gasoline": 400,
    }

    print(f"\nstarting planning (with_input_layer={with_input_layer}, mode={mode})...")
    result = planning_layer.plan(
        task_description=task,
        goals=goals,
        resources=resources,
        mode=mode,
        verbose=True,
    )

    print("\n=== Planning result ===")
    print(f"summary: {result.get('plan', 'N/A')[:100]}")
    print(f"allocated resources ({len(result.get('allocated_resources', {}))} items): "
          f"{json.dumps(result.get('allocated_resources', {}), ensure_ascii=False)[:200]}")
    print(f"personnel allocation: {json.dumps(result.get('assigned_personnel', {}), ensure_ascii=False)[:200]}")
    print(f"scheduled tasks: {len(result.get('schedule', []))} in total")
    print(f"region responses: {result.get('region_response_times', {})}")

    stats = planning_layer.get_stats()
    print(f"\nengine stats: calls {stats['engine_stats']['total_calls']}, "
          f"parse success rate {stats['engine_stats']['parse_success_rate']:.0%}")

    return result


if __name__ == "__main__":
    test_module3_decompose()

    try:
        test_module4_plan()
    except Exception as e:
        print(f"M4 test skipped (the LLM endpoint may be down): {e}")

    try:
        test_full_planning_layer(with_input_layer=True, mode="global")
    except Exception as e:
        print(f"global planning test failed: {e}")

    try:
        test_full_planning_layer(with_input_layer=True, mode="decomposed")
    except Exception as e:
        print(f"decomposed planning test failed: {e}")

    print("\n" + "=" * 60)
    print("All tests finished!")
    print("=" * 60)
