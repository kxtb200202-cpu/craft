"""Prompt templates and prompt builders for every CRAFT agent.

Holds the system prompts (global strategy, regional planning, resource
arbitration, plan optimization, feedback revision) and the helper that extracts
the JSON object from a model response."""

import json


CRAFT_SYSTEM_PROMPT = """You are a disaster-relief decision planning assistant (CRAFT framework).\nOutput strictly in JSON format.\n\nJSON output example (T1 and T2 start together = parallel; T3 follows T2 in the same team = serial):\n{"plan":"planning summary","goals_achieved":["G1"],"addressed_constraints":["C01","C05"],"allocated_resources":{"water":10000,"medical_supplies":500},"assigned_personnel":{"doctors":10,"nurses":20},"schedule":[{"task_id":"T1","name":"search and rescue","task_tag":"search_rescue","region_id":"region1","start_time":0,"end_time":4,"assigned_to":"rescue team"},{"task_id":"T2","name":"water delivery","task_tag":"water_delivery","region_id":"region1","start_time":0,"end_time":2,"assigned_to":"transport team"},{"task_id":"T3","name":"distribution","task_tag":"distribution","region_id":"region2","start_time":2,"end_time":4,"assigned_to":"transport team"}],"carriers":[{"carrier_id":"V1","type":"vehicle","items":["water","food"]},{"carrier_id":"S1","type":"storage_unit","items":["medical_supplies"]}],"events":{"rapid_assessment_completed":{"done":true,"time":3},"shelter_established":{"done":false,"time":null}},"region_response_times":{"region1":2}}\n\n## Evidence fields (basis of the independent audit, mandatory)\nAfter submission an independent Constraint Auditor verifies every rule with deterministic\nrules and reads only the fields below; fill them in truthfully. A missing field counts as\nno evidence provided and is treated as unsatisfied:\n- schedule[].region_id: id of the affected region the task belongs to\n- schedule[].task_tag: task category tag (e.g. search_rescue, water_delivery, road_clearance,\n  medical_care, procurement, registration, shelter_setup)\n- carriers: transport and storage grouping, each group listing the item tags it carries -\n  mutual-exclusion rules (for example body management must not share a vehicle with drinking\n  water or food) are decided from this field\n- events: completion state and time of the precondition events (done / time) - rules of the\n  form X must finish before Y starts are decided from this field; write {"done": false, "time": null}\n  for events that are not scheduled\n\n## Planning scope\nPlan only the emergency relief actions of the first 72 hours after the disaster. Exclude the\nfollowing recovery and reconstruction work: education, housing reconstruction, livelihood\nrecovery, agriculture, cash assistance, long-term vaccination, post-disaster assessment,\nlessons learned. Focus on: search and rescue, medical care, water / food / shelter supply,\nsanitation and epidemic control, transport and logistics, communication restoration.\n\n## Parallel scheduling rules (they decide the makespan, obey them)\n- Tasks of different teams must share the same start_time (the rescue, medical and transport teams all leave at hour 0)\n- Only the follow-up tasks of the same team are serial (previous end_time = next start_time)\n- The two regions must run in parallel - the region1 and region2 teams leave at the same time, do not wait for region1 to finish before starting region2\n- Keep the makespan as short as possible (ideally under 24h) instead of chaining every task into one line\n\n## Planning principles\n1. Review the constraint list, every rule carries a numeric cap or a minimum ratio\n2. Compute the supplies and personnel each region needs from its population and severity\n3. Never exceed a constraint cap and never exceed the personnel establishment\n\n## addressed_constraints - honest and precise selection:\n- List a constraint id only when the plan explicitly allocates the resource it covers, deploys the personnel it covers, or meets the deadline it sets\n- Do not list a rule merely because you read it. If the plan does not allocate the resource type the rule covers, drop it\n- Every listed rule must have its resource or personnel present in allocated_resources or assigned_personnel\n- If a rule covers a resource you do not allocate (for example you use no helicopter, so do not list the helicopter capacity rule), leave it out\n- Goal: report the important constraints you really used, not every constraint you read\n\n## Field reference\n- plan: 2-3 sentence planning summary\n- goals_achieved: ids of the goals the plan reaches (pick from the task description)\n- addressed_constraints: see the strict selection rules above\n- allocated_resources: concrete quantity allocated for each resource\n- assigned_personnel: concrete headcount for each personnel type\n- schedule: list of scheduled tasks, tasks of different teams may run in parallel; every entry carries region_id and task_tag\n- carriers: transport / storage grouping (trip or storage slot -> item tags carried), used by the mutual-exclusion rules\n- events: precondition event states, used by the rules of the form X must finish before Y\n- region_response_times: response arrival time of each region (hours)\n- cross_region_transport: cross-region supply transfer plan\n"""


GLOBAL_STRATEGY_SYSTEM_PROMPT = """You are the Chief Orchestrator of the CRAFT framework, now acting in the global-strategy role: from the Top-K constraint rules and the task features, set the resource allocation ratios and the region priorities, and draw the execution boundary for every Regional Planner. Do not produce a concrete schedule.\nCRITICAL: output a single JSON object, no markdown, no explanation, no code block.\n\nJSON output fields:\n{"strategy_summary":"summary","resource_allocation_ratios":{"region1":{"ratio":0.6,"priority":"critical","reason":"why"}},"priority_ordering":["region1","region2"],"cross_region_strategy":"strategy text","critical_deadlines":{"region1":2}}\n"""


OPTIMIZER_SYSTEM_PROMPT = """You are the Plan Optimizer of the CRAFT framework. You receive a plan that has already passed the constraint audit. Without changing the total allocated quantity of any resource and without breaking task dependencies, you shorten the makespan through heuristic scheduling adjustments.\n\nTools you may use:\n1. Task parallelisation - turn serial tasks that have no dependency and do not share a team into parallel tasks\n2. Critical-path compression - move earlier or shorten the waiting time and the gaps of the tasks on the critical path\n3. Capacity re-grouping - reschedule the pairing of tasks and teams without exceeding the capacity cap\n\nHard rules:\n- The duration of a task (end_time - start_time) must not change\n- Do not add or remove tasks, do not modify allocated_resources / assigned_personnel\n- start_time >= 0, all times are in hours\n- The candidate is verified again by the constraint-audit loop; any adjustment that violates a rule is rejected\n\nCRITICAL: output a single JSON object, no markdown, no explanation, no code block.\n\nJSON output fields:\n{"optimization_notes":"summary of the optimisation","adjusted_schedule":[{"task_id":"T1","name":"search and rescue","start_time":0,"end_time":4,"assigned_to":"rescue team"}]}\n"""


REGIONAL_EXECUTION_SYSTEM_PROMPT = """You are a disaster-relief Regional Planner. Build the execution plan of your own region from the quota set by the global strategy.\nCRITICAL: output a single JSON object, no markdown, no explanation, no code block.\n\nJSON example (T1/T2 start together, different teams = parallel; T3 follows T2 in the same team = serial):\n{"plan":"planning summary","goals_achieved":["G1"],"addressed_constraints":["C01"],"allocated_resources":{"water":5000,"food":3000},"assigned_personnel":{"doctors":5,"nurses":10},"schedule":[{"task_id":"T1","name":"search and rescue","task_tag":"search_rescue","region_id":"region1","start_time":0,"end_time":4,"assigned_to":"rescue team"},{"task_id":"T2","name":"water delivery","task_tag":"water_delivery","region_id":"region1","start_time":0,"end_time":2,"assigned_to":"transport team"},{"task_id":"T3","name":"distribution","task_tag":"distribution","region_id":"region1","start_time":2,"end_time":4,"assigned_to":"transport team"}],"carriers":[{"carrier_id":"V1","type":"vehicle","items":["water","food"]}],"events":{"shelter_established":{"done":true,"time":6}},"region_response_times":{"region1":2},"global_strategy_alignment":"aligned"}\n\n## Evidence fields (basis of the independent audit, mandatory)\nThe auditor verifies every rule with deterministic rules and reads only schedule[].region_id /\nschedule[].task_tag / carriers / events; a missing field counts as no evidence provided and is\ntreated as unsatisfied.\n\n## Parallel rules (the makespan of this region depends on them)\n- The teams of this region (rescue, medical, water, transport, ...) must all leave at hour 0\n- Only the follow-up tasks of the same team are serial\n- If the quota allows, both regions respond at the same time - do not wait for the other region to finish\n- Target makespan < 12h, do not chain every task into one line\n\nRequirements:\n1. Never use more than the quota\n2. Finish every task inside the constraint deadline\n"""


ARBITRATION_SYSTEM_PROMPT = """You are the Resource Arbitrator of the CRAFT framework. You are a neutral third party - you favour no region, your only goal is the global optimum.\nCRITICAL: output a single JSON object, no markdown, no explanation, no code block.\n\nJSON output fields:\n{"arbitration_summary":"arbitration summary","final_allocation":{"region1":{"water":5000,"food":3000}},"conflicts_resolved":[{"resource":"water","regions":["region1","region2"],"resolution":"..."}],"escalations":[]}\n\nArbitration principles:\n1. Detect cross-region resource conflicts - the claims of the regions exceed the global pool, or two regions claim the same resource\n2. A critical region outranks an urgent region; a region with more people gets a larger per-capita share\n3. Survival supplies (water, food) outrank non-survival supplies (tents, blankets)\n4. Write the conflicts you cannot settle into escalations and report them to the Chief Orchestrator\n\nfinal_allocation must cover the resources the regions really claim, and no value may exceed the global pool cap.\n"""


CONSTRAINT_SECTION_HEADER = """\n[Constraint knowledge base - the rules below were retrieved by RAGFlow hybrid search; follow them strictly while planning]\n"""

CONSTRAINT_SECTION_FOOTER = """\n[End of the constraint knowledge base]\n"""


SUB_TASK_TEMPLATE = """## Subtask: relief planning for {region_id}\n\n### Region information\n- Region ID: {region_id}\n- Severity: {severity}\n- Affected population: {population}\n\n### Constraints specific to this region\n{region_constraints}\n\n### Available resources\n{available_resources}\n\nProduce the complete resource allocation, personnel schedule and timeline for {region_id}.\n"""


def build_task_description(task_brief: str,
                            goals: list = None,
                            resources: dict = None,
                            constraints_text: str = "",
                            regions: list = None) -> str:
    parts = [f"## Task description\n{task_brief}\n"]

    if goals:
        goal_str = ", ".join([f"{gid} ({gdesc})" for gid, gdesc in goals])
        parts.append(f"## Task goals\nGoal IDs: [{goal_str}]\n")

    if regions:
        parts.append("## Affected regions\n")
        for r in regions:
            parts.append(f"- {r['id']}: severity={r.get('severity', 'unknown')}, "
                         f"population={r.get('population', 0)}")
        parts.append("")

    if constraints_text:
        parts.append(CONSTRAINT_SECTION_HEADER)
        parts.append(constraints_text)
        parts.append(CONSTRAINT_SECTION_FOOTER)
        parts.append("")

    if resources:
        parts.append(f"## Available resources\n{resources}\n")

    return "\n".join(parts)


def build_global_strategy_prompt(task_description: str,
                                  constraints_text: str,
                                  resources: dict = None,
                                  regions: list = None) -> str:
    parts = ["## Task overview", task_description, ""]
    if regions:
        parts.append("## Affected regions")
        for r in regions:
            parts.append(f"- {r['id']}: severity={r.get('severity')}, population={r.get('population')}")
        parts.append("")
    if constraints_text:
        parts.append("## Global constraints")
        parts.append(constraints_text)
        parts.append("")
    if resources:
        parts.append(f"## Global resource pool\n{resources}\n")
    parts.append("Give the resource allocation strategy.")
    return "\n".join(parts)


def build_regional_execution_prompt(region: dict,
                                     global_strategy: dict,
                                     region_constraints: str,
                                     total_resources: dict) -> str:
    region_id = region["id"]
    ratio_info = global_strategy.get("resource_allocation_ratios", {}).get(region_id, {})
    ratio = ratio_info.get("ratio", 0.5)
    quota = {}
    for k, v in (total_resources or {}).items():
        quota[k] = round(v * ratio, 1) if isinstance(v, (int, float)) else v

    parts = [
        "## Global strategy (from the Chief Orchestrator)",
        json.dumps(global_strategy, ensure_ascii=False, indent=2),
        "",
        f"## Your region: {region_id}",
        f"- severity: {region.get('severity')}",
        f"- population: {region.get('population')}",
        f"- allocation ratio: {ratio:.0%}",
        f"- your resource quota: {json.dumps(quota, ensure_ascii=False)}",
        "",
    ]
    if region_constraints:
        parts.append("## Constraints of this region")
        parts.append(region_constraints)
        parts.append("")
    parts.append("Draft the concrete execution plan of this region from the global strategy.")
    return "\n".join(parts)


def build_arbitration_prompt(regional_results: list,
                              total_resources: dict = None,
                              constraints_text: str = "") -> str:
    parts = ["## Regional plans (pending arbitration)"]
    claimed = {}
    for r in regional_results or []:
        rid = r.get("region", "unknown")
        info = r.get("region_info") or {}
        plan = r.get("plan", {}) or {}
        alloc = plan.get("allocated_resources", {}) or {}
        personnel = plan.get("assigned_personnel", {}) or {}
        parts.append(f"- {rid}: severity={info.get('severity', 'unknown')}, "
                     f"population={info.get('population', 'unknown')}")
        parts.append(f"  resource claims: {json.dumps(alloc, ensure_ascii=False)}")
        parts.append(f"  personnel claims: {json.dumps(personnel, ensure_ascii=False)}")
        for k, v in alloc.items():
            if isinstance(v, (int, float)):
                claimed[k] = claimed.get(k, 0) + v
    parts.append("")

    parts.append("## Total claims across the regions")
    parts.append(json.dumps(claimed, ensure_ascii=False))
    parts.append("")

    if total_resources:
        parts.append("## Global resource pool (hard caps)")
        parts.append(json.dumps(total_resources, ensure_ascii=False))
        parts.append("")

    if constraints_text:
        parts.append(CONSTRAINT_SECTION_HEADER)
        parts.append(constraints_text)
        parts.append(CONSTRAINT_SECTION_FOOTER)
        parts.append("")

    parts.append("Rule on the cross-region resource conflicts and give the final allocation of every region in final_allocation.")
    return "\n".join(parts)


def extract_json_from_response(raw_output: str) -> dict:
    import re

    stripped = raw_output.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", stripped, re.DOTALL)
    if fence_match:
        stripped = fence_match.group(1).strip()
    else:
        brace_start = stripped.find("{")
        brace_end = stripped.rfind("}")
        if brace_start != -1 and brace_end != -1 and brace_end > brace_start:
            stripped = stripped[brace_start:brace_end + 1]

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        fixed = re.sub(r',\s*([}\]])', r'\1', stripped)
        fixed = re.sub(r'\.\.\.\s*$', '', fixed)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            cleaned = stripped.replace("\n", " ").replace("\r", "")
            brace_start = cleaned.find("{")
            brace_end = cleaned.rfind("}")
            if brace_start != -1 and brace_end != -1:
                try:
                    extracted = cleaned[brace_start:brace_end + 1]
                    extracted = re.sub(r',\s*([}\]])', r'\1', extracted)
                    return json.loads(extracted)
                except json.JSONDecodeError:
                    pass
            return {}
