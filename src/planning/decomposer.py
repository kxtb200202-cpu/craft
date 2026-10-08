"""M1: task decomposition.

Splits a planning task into one sub-task per task unit (region id, severity,
population) and merges the resulting sub-plans. Decomposition is deterministic
parsing driven by the Chief Orchestrator."""

import re
from typing import List, Dict, Any

try:
    from ..agents.prompts import SUB_TASK_TEMPLATE
except ImportError:
    from agents.prompts import SUB_TASK_TEMPLATE


class PlanDecomposer:
    def __init__(self, input_layer=None):
        self.input_layer = input_layer

    def extract_regions(self, task_description: str) -> List[Dict[str, Any]]:
        regions = []

        text = task_description.replace(";", " ").replace("\n", " ")

        pattern0 = re.findall(
            r'[Rr]egion\s+([a-zA-Z0-9_-]+)\s*\(([^,)]+),\s*([\d,]+)\s*'
            r'(?:people|persons|population|residents)',
            text
        )
        for region_id, severity, pop in pattern0:
            sev = self._normalize_severity(severity)
            rid_nums = re.findall(r'(\d+)', region_id)
            rid = f"region{rid_nums[0]}" if rid_nums else region_id
            regions.append({
                "id": rid,
                "severity": sev,
                "population": int(pop.replace(",", "")),
            })

        if not regions:
            pattern1 = re.findall(
                r'[Rr]egion\s*(\d+)\s*'
                r'(critical|urgent|normal|severe|extreme|high|moderate|medium|low)?'
                r'.*?([\d,]+)\s*(?:people|persons|population|residents)',
                text
            )
            for num, severity, pop in pattern1:
                sev = self._normalize_severity(severity)
                regions.append({
                    "id": f"region{num}",
                    "severity": sev,
                    "population": int(pop.replace(",", "")),
                })

        if not regions:
            pattern2 = re.findall(
                r'(?:region|area|zone)\s*([A-Za-z0-9]).*?'
                r'(critical|urgent|normal|severe|extreme|moderate)?.*?'
                r'([\d,]+)\s*(?:people|persons|population|residents)',
                text, re.IGNORECASE
            )
            for num, severity, pop in pattern2:
                sev = self._normalize_severity(severity)
                regions.append({
                    "id": f"region{num}",
                    "severity": sev,
                    "population": int(pop.replace(",", "")),
                })

        if not regions:
            pops = re.findall(r'([\d,]+)\s*(?:people|persons|population|residents)',
                              text, re.IGNORECASE)
            for i, pop in enumerate(pops[:4], 1):
                regions.append({
                    "id": f"region{i}",
                    "severity": "urgent",
                    "population": int(pop.replace(",", "")),
                })

        return regions

    def _normalize_severity(self, severity_raw: str) -> str:
        if not severity_raw:
            return "urgent"
        s = severity_raw.lower().strip()
        if s in ("critical", "severe", "extreme", "high"):
            return "critical"
        elif s in ("urgent", "moderate", "medium"):
            return "urgent"
        else:
            return "normal"

    def build_sub_task_description(self, region: Dict[str, Any],
                                    base_task: str,
                                    region_constraints: str = "",
                                    available_resources: str = "") -> str:
        return SUB_TASK_TEMPLATE.format(
            region_id=region["id"],
            severity=region["severity"],
            population=region["population"],
            region_constraints=region_constraints or "(using the global constraints)",
            available_resources=available_resources or "(using the global resource pool)",
        )

    def decompose(self, task_description: str,
                  goals: list = None,
                  resources: dict = None,
                  top_k: int = 15) -> List[Dict[str, Any]]:
        regions = self.extract_regions(task_description)

        if not regions:
            print("[M1] no multi-region structure detected, using the global planning mode")
            return [{
                "region": {"id": "global", "severity": "critical", "population": 0},
                "task_desc": task_description,
                "constraints_text": "",
                "raw_constraints": [],
            }]

        print(f"[M1] detected {len(regions)} regions: "
              f"{[r['id'] + '(' + r['severity'] + ',' + str(r['population']) + ' people)' for r in regions]}")

        sub_tasks = []
        for region in regions:
            region_desc = f"{task_description}\nCurrent planning region: {region['id']}"

            constraints_text = ""
            raw_constraints = []
            if self.input_layer:
                try:
                    region_query = (
                        f"{task_description} "
                        f"region: {region['id']} "
                        f"severity: {region['severity']} "
                        f"population: {region['population']}"
                    )
                    constraints_text = self.input_layer.retrieve(
                        region_query, top_k=top_k, verbose=False)
                    raw_constraints = self.input_layer.retrieve_raw(
                        region_query, top_k=top_k, verbose=False)
                except Exception as e:
                    print(f"[M1] region {region['id']} constraint retrieval failed: {e}")

            sub_task_desc = self.build_sub_task_description(
                region=region,
                base_task=region_desc,
                region_constraints=constraints_text,
                available_resources=str(resources) if resources else "",
            )

            sub_tasks.append({
                "region": region,
                "task_desc": sub_task_desc,
                "constraints_text": constraints_text,
                "raw_constraints": raw_constraints,
            })

        return sub_tasks


def merge_sub_plans(sub_plans: List[Dict[str, Any]],
                     global_resources: dict = None) -> Dict[str, Any]:
    merged = {
        "plan": "",
        "region_plans": {},
        "allocated_resources": {},
        "assigned_personnel": {},
        "schedule": [],
        "carriers": [],
        "events": {},
        "region_response_times": {},
        "addressed_constraints": [],
    }

    plan_summaries = []
    all_addressed = []

    for sp in sub_plans:
        region_id = sp.get("region", "unknown")
        plan = sp.get("plan", {})

        if "region_plans" in plan:
            merged["region_plans"].update(plan["region_plans"])
        else:
            merged["region_plans"][region_id] = plan

        for k, v in plan.get("allocated_resources", {}).items():
            merged["allocated_resources"][k] = merged["allocated_resources"].get(k, 0) + v

        for k, v in plan.get("assigned_personnel", {}).items():
            merged["assigned_personnel"][k] = merged["assigned_personnel"].get(k, 0) + v

        merged["schedule"].extend(plan.get("schedule", []))

        merged["carriers"].extend(plan.get("carriers", []) or [])
        for _ek, _ev in (plan.get("events", {}) or {}).items():
            merged["events"].setdefault(_ek, _ev)

        if "response_time_hours" in plan:
            merged["region_response_times"][region_id] = plan["response_time_hours"]

        for ac in plan.get("addressed_constraints", []):
            if ac not in all_addressed:
                all_addressed.append(ac)

        plan_summaries.append(
            f"{region_id}: resources{len(plan.get('allocated_resources', {}))} items, "
            f"tasks{len(plan.get('schedule', []))}"
        )

    merged["addressed_constraints"] = all_addressed

    merged["plan"] = " | ".join(plan_summaries)

    if global_resources:
        exceeded = []
        for k, total in merged["allocated_resources"].items():
            if k in global_resources and total > global_resources[k]:
                exceeded.append(f"{k}: used {total} > cap {global_resources[k]}")
        if exceeded:
            merged["_resource_exceeded"] = exceeded
            print(f"[M1] [WARN] resource overrun: {exceeded}")

    print(f"[M1] merge done: {len(sub_plans)} regions -> {len(merged['schedule'])} scheduled tasks")
    return merged
