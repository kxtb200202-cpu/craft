"""Meta layer: the Consensus Builder (deterministic, no LLM).

Merges the plan versions produced by the planning, verification and
optimization stages, reconciles conflicts without altering agent-produced
values, and formats the final plan for the benchmark."""

from typing import List, Dict, Any


class ConsensusBuilder:
    def __init__(self):
        self.stats = {
            "total_builds": 0,
            "conflicts_detected": [],
            "conflicts_resolved": [],
            "conflicts_escalated": [],
        }

    def build(self, plan_versions: List[Dict[str, Any]],
              chief_orchestrator=None,
              task_context: Dict[str, Any] = None,
              verbose: bool = True) -> Dict[str, Any]:
        self.stats["total_builds"] += 1

        if verbose:
            print(f"\n[ConsensusBuilder] merging {len(plan_versions)} plan versions")

        versions = self._extract_versions(plan_versions)

        conflicts = self._detect_conflicts(versions, verbose)

        resolutions = []
        for conflict in conflicts:
            if conflict["severity"] == "critical" and chief_orchestrator:
                resolution = chief_orchestrator.resolve_dispute(conflict)
                self.stats["conflicts_escalated"].append(conflict)
            else:
                resolution = self._auto_resolve(conflict, versions)
                self.stats["conflicts_resolved"].append(conflict)
            resolutions.append(resolution)

        self.stats["conflicts_detected"].append(len(conflicts))

        if verbose and conflicts:
            print(f"[ConsensusBuilder] detected {len(conflicts)} conflicts, "
                  f"resolved {len(resolutions)} in total")

        final = self._merge(versions, resolutions, task_context)

        final["_meta"] = {
            "source_trace": self._build_source_trace(versions),
            "conflicts": conflicts,
            "resolutions": resolutions,
            "version_count": len(versions),
        }

        if verbose:
            print(f"[ConsensusBuilder] final plan: "
                  f"{len(final.get('allocated_resources', {}))} resources, "
                  f"{len(final.get('schedule', []))} tasks")

        return final

    def _extract_versions(self, plan_versions: List[Dict[str, Any]]
                          ) -> List[Dict[str, Any]]:
        versions = []
        for pv in plan_versions:
            source = pv.get("source", "unknown")
            plan = pv.get("plan", pv)

            versions.append({
                "source": source,
                "plan": plan.get("plan", ""),
                "allocated_resources": plan.get("allocated_resources", {}),
                "assigned_personnel": plan.get("assigned_personnel", {}),
                "schedule": plan.get("schedule", []),
                "region_response_times": plan.get("region_response_times", {}),
                "goals_achieved": plan.get("goals_achieved", []),
                "satisfied_constraints": plan.get("satisfied_constraints", []),
                "addressed_constraints": plan.get("addressed_constraints", []),
                "compliance_rate": plan.get("compliance_rate",
                                             pv.get("compliance_rate")),
                "improvement_pct": pv.get("improvement_pct"),
                "raw": plan,
            })

        return versions

    def _detect_conflicts(self, versions: List[Dict[str, Any]],
                           verbose: bool) -> List[Dict[str, Any]]:
        conflicts = []

        if len(versions) < 2:
            return conflicts

        v_a = versions[0]
        v_b = versions[-1]

        all_resources = set(v_a["allocated_resources"]) | set(v_b["allocated_resources"])
        for res in all_resources:
            val_a = v_a["allocated_resources"].get(res, 0)
            val_b = v_b["allocated_resources"].get(res, 0)
            if val_a and val_b and abs(val_a - val_b) / max(val_a, val_b, 1) > 0.3:
                conflicts.append({
                    "type": "resource_conflict",
                    "resource": res,
                    "value_a": val_a,
                    "value_b": val_b,
                    "source_a": v_a["source"],
                    "source_b": v_b["source"],
                    "severity": "critical" if res in ("water", "medical_supplies", "food") else "minor",
                })

        all_pers = set(v_a["assigned_personnel"]) | set(v_b["assigned_personnel"])
        for pers in all_pers:
            val_a = v_a["assigned_personnel"].get(pers, 0)
            val_b = v_b["assigned_personnel"].get(pers, 0)
            if val_a and val_b and abs(val_a - val_b) / max(val_a, val_b, 1) > 0.3:
                conflicts.append({
                    "type": "personnel_conflict",
                    "personnel": pers,
                    "value_a": val_a,
                    "value_b": val_b,
                    "source_a": v_a["source"],
                    "source_b": v_b["source"],
                    "severity": "minor",
                })

        sched_a = {(t.get("task_id"), t.get("start_time")) for t in v_a["schedule"]}
        sched_b = {(t.get("task_id"), t.get("start_time")) for t in v_b["schedule"]}
        changed_tasks = sched_a.symmetric_difference(sched_b)
        if changed_tasks and len(changed_tasks) > len(sched_a) * 0.5:
            conflicts.append({
                "type": "schedule_conflict",
                "changed_tasks": len(changed_tasks),
                "source_a": v_a["source"],
                "source_b": v_b["source"],
                "severity": "minor",
            })

        return conflicts

    def _auto_resolve(self, conflict: Dict[str, Any],
                       versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        ctype = conflict.get("type", "")

        if ctype == "resource_conflict":
            return {
                "conflict_type": ctype,
                "resolution": "use_latest",
                "chosen_value": conflict["value_b"],
                "reason": "take the value of the latest (optimised) version",
            }
        elif ctype == "personnel_conflict":
            return {
                "conflict_type": ctype,
                "resolution": "use_latest",
                "chosen_value": conflict["value_b"],
                "reason": "take the personnel configuration of the latest version",
            }
        elif ctype == "schedule_conflict":
            return {
                "conflict_type": ctype,
                "resolution": "use_latest",
                "reason": "take the schedule of the latest version",
            }
        else:
            return {
                "conflict_type": ctype,
                "resolution": "keep_original",
                "reason": "cannot be settled automatically, keeping the original plan",
            }

    def _merge(self, versions: List[Dict[str, Any]],
                resolutions: List[Dict[str, Any]],
                task_context: Dict[str, Any] = None) -> Dict[str, Any]:
        if not versions:
            return {}

        first = versions[0]
        last = versions[-1]

        latest_tasks = {t.get("task_id"): t for t in last["schedule"]}
        for t in first["schedule"]:
            tid = t.get("task_id")
            if tid not in latest_tasks:
                latest_tasks[tid] = t
        merged_schedule = list(latest_tasks.values())
        merged_schedule.sort(key=lambda t: t.get("start_time", 0))

        merged_response_times = dict(first["region_response_times"])
        merged_response_times.update(last["region_response_times"])

        return {
            "plan": first["plan"] or last["plan"] or "",
            "goals_achieved": first["goals_achieved"],
            "addressed_constraints": first.get("addressed_constraints", []),
            "satisfied_constraints": last.get("satisfied_constraints",
                                               first.get("satisfied_constraints", [])),
            "allocated_resources": last["allocated_resources"],
            "assigned_personnel": last["assigned_personnel"],
            "schedule": merged_schedule,
            "region_response_times": merged_response_times,
            "compliance_rate": last.get("compliance_rate"),
            "improvement_pct": last.get("improvement_pct"),
        }

    def _build_source_trace(self, versions: List[Dict[str, Any]]) -> Dict[str, Any]:
        trace = {}
        for v in versions:
            source = v["source"]
            trace[source] = {
                "plan_summary": str(v.get("plan", ""))[:100],
                "resource_count": len(v.get("allocated_resources", {})),
                "task_count": len(v.get("schedule", [])),
                "compliance_rate": v.get("compliance_rate"),
                "improvement_pct": v.get("improvement_pct"),
            }
        return trace

    def format_for_benchmark(self, final_plan: Dict[str, Any]) -> Dict[str, Any]:
        schedule = final_plan.get("schedule", [])
        task_completion_times = {}
        for t in schedule:
            tid = t.get("task_id", "")
            if tid:
                task_completion_times[tid] = t.get("end_time", 0)

        return {
            "goals": final_plan.get("goals_achieved", []),
            "schedule": schedule,
            "allocated_resources": final_plan.get("allocated_resources", {}),
            "assigned_personnel": final_plan.get("assigned_personnel", {}),
            "region_response_times": final_plan.get("region_response_times", {}),
            "task_completion_times": task_completion_times,
            "plan_summary": final_plan.get("plan", ""),
            "addressed_constraints": final_plan.get("addressed_constraints", []),
            "_compliance_rate": final_plan.get("compliance_rate"),
            "_improvement_pct": final_plan.get("improvement_pct"),
        }

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "avg_conflicts_per_build": round(
                sum(self.stats["conflicts_detected"]) /
                max(self.stats["total_builds"], 1), 1),
        }
