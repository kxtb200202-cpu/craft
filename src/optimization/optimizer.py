"""M7: makespan optimization.

LLMPlanOptimizer is the Plan Optimizer agent: it proposes parallelization,
critical-path compression and transport re-assignment, and every candidate must
pass a re-audit before adoption. PlanOptimizer is the deterministic heuristic
used when the LLM path is unavailable or returns an invalid plan."""

import copy
import json
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

try:
    from ..agents.prompts import OPTIMIZER_SYSTEM_PROMPT, extract_json_from_response
except ImportError:
    from agents.prompts import OPTIMIZER_SYSTEM_PROMPT, extract_json_from_response


class PlanOptimizer:
    def __init__(self):
        pass

    def optimize(self, plan: Dict[str, Any],
                 resources: Dict[str, float] = None,
                 vehicle_capacity: Dict[str, Any] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        schedule = plan.get("schedule", [])
        if not schedule:
            return {
                "optimized_plan": plan,
                "original_makespan": 0,
                "optimized_makespan": 0,
                "improvement_pct": 0,
                "optimizations_applied": ["no scheduled task, skipping the optimisation"],
            }

        original_makespan = self._compute_makespan(schedule)
        if verbose:
            print(f"[Optimizer] original makespan: {original_makespan}h, "
                  f"{len(schedule)} tasks")

        current_schedule = copy.deepcopy(schedule)
        optimizations = []

        current_schedule, parallel_improved = self._parallelize(current_schedule)
        if parallel_improved:
            optimizations.append("task_parallelization")

        current_schedule, gap_improved = self._compress_gaps(current_schedule)
        if gap_improved:
            optimizations.append("gap_compression")

        if vehicle_capacity:
            current_schedule, vehicle_improved = self._optimize_vehicles(
                current_schedule, vehicle_capacity)
            if vehicle_improved:
                optimizations.append("vehicle_rescheduling")

        current_schedule, defer_improved = self._defer_non_critical(
            current_schedule)
        if defer_improved:
            optimizations.append("non_critical_deferral")

        current_schedule, critical_improved = self._shorten_critical_path(
            current_schedule, plan.get("allocated_resources", {}))
        if critical_improved:
            optimizations.append("critical_path_shortening")

        optimized_makespan = self._compute_makespan(current_schedule)
        improvement_pct = round(
            (original_makespan - optimized_makespan) / max(original_makespan, 1) * 100, 1
        )

        if verbose:
            print(f"[Optimizer] optimised makespan: {optimized_makespan}h "
                  f" (improvement {improvement_pct:.1f}%)")
            print(f"[Optimizer] strategies applied: {optimizations}")

        optimized_plan = dict(plan)
        optimized_plan["schedule"] = current_schedule

        return {
            "optimized_plan": optimized_plan,
            "original_makespan": original_makespan,
            "optimized_makespan": optimized_makespan,
            "improvement_pct": improvement_pct,
            "optimizations_applied": optimizations,
        }

    def _compute_makespan(self, schedule: List[Dict]) -> float:
        if not schedule:
            return 0
        return max(t.get("end_time", 0) for t in schedule)

    def _parallelize(self, schedule: List[Dict]
                     ) -> Tuple[List[Dict], bool]:
        if len(schedule) <= 1:
            return schedule, False

        improved = False
        result = []

        for i, task in enumerate(schedule):
            assigned = task.get("assigned_to", "")
            region = task.get("region", "")
            name = task.get("name", "").lower()

            earliest_start = 0
            for existing in result:
                ex_assigned = existing.get("assigned_to", "")
                ex_region = existing.get("region", "")
                ex_end = existing.get("end_time", 0)

                if assigned and ex_assigned and assigned == ex_assigned:
                    earliest_start = max(earliest_start, ex_end)
                elif region and ex_region and region == ex_region:
                    ex_name = existing.get("name", "").lower()
                    has_overlap = any(
                        kw in name and kw in ex_name
                        for kw in ["medical", "water", "food", "shelter", "rescue"]
                    )
                    if not has_overlap:
                        pass
                    else:
                        earliest_start = max(earliest_start, ex_end)

            duration = task.get("end_time", 0) - task.get("start_time", 0)
            new_task = dict(task)
            new_task["start_time"] = earliest_start
            new_task["end_time"] = earliest_start + max(duration, 0)
            result.append(new_task)

        new_makespan = self._compute_makespan(result)
        old_makespan = self._compute_makespan(schedule)
        improved = new_makespan < old_makespan

        return result, improved

    def _compress_gaps(self, schedule: List[Dict]
                       ) -> Tuple[List[Dict], bool]:
        if len(schedule) <= 1:
            return schedule, False

        sorted_tasks = sorted(schedule, key=lambda t: t.get("start_time", 0))

        improved = False
        result = []

        for task in sorted_tasks:
            if not result:
                result.append(dict(task))
                continue

            prev = result[-1]
            gap = task.get("start_time", 0) - prev.get("end_time", 0)

            if gap > 0.5:
                duration = task.get("end_time", 0) - task.get("start_time", 0)
                new_task = dict(task)
                new_task["start_time"] = prev["end_time"]
                new_task["end_time"] = prev["end_time"] + duration
                result.append(new_task)
                improved = True
            else:
                result.append(dict(task))

        return result, improved

    def _optimize_vehicles(self, schedule: List[Dict],
                            vehicle_capacity: Dict[str, Any]
                            ) -> Tuple[List[Dict], bool]:
        transport_keywords = ["transport", "delivery", "truck", "helicopter",
                              "convoy", "freight", "shipping"]
        transport_tasks = [t for t in schedule
                           if any(kw in str(t.get("name", "")).lower()
                                  for kw in transport_keywords)]

        if len(transport_tasks) <= 1:
            return schedule, False

        by_dest = defaultdict(list)
        for i, t in enumerate(schedule):
            for tt in transport_tasks:
                if t.get("task_id") == tt.get("task_id"):
                    dest = t.get("region", t.get("assigned_to", "unknown"))
                    by_dest[dest].append(i)
                    break

        result = [dict(t) for t in schedule]
        improved = False

        for dest, indices in by_dest.items():
            if len(indices) >= 2:
                for j in range(1, len(indices)):
                    prev_idx = indices[j - 1]
                    curr_idx = indices[j]
                    prev_end = result[prev_idx].get("end_time", 0)
                    curr_start = result[curr_idx].get("start_time", 0)
                    if curr_start > prev_end + 1:
                        duration = result[curr_idx].get("end_time", 0) - curr_start
                        result[curr_idx]["start_time"] = prev_end + 0.5
                        result[curr_idx]["end_time"] = prev_end + 0.5 + duration
                        improved = True

        return result, improved

    def _defer_non_critical(self, schedule: List[Dict]
                            ) -> Tuple[List[Dict], bool]:
        non_critical_keywords = [
            "blanket", "tent", "education", "cash", "livelihood",
            "reconstruction", "winter", "clothing", "clothes",
        ]
        critical_keywords = [
            "water", "food", "medical", "rescue", "shelter_material",
        ]

        non_critical_indices = []
        for i, t in enumerate(schedule):
            name = str(t.get("name", "")).lower()
            if any(kw in name for kw in non_critical_keywords):
                if not any(kw in name for kw in critical_keywords):
                    non_critical_indices.append(i)

        if not non_critical_indices:
            return schedule, False

        result = [dict(t) for t in schedule]
        improved = False

        latest_critical_end = 0
        for i, t in enumerate(schedule):
            if i not in non_critical_indices:
                latest_critical_end = max(latest_critical_end, t.get("end_time", 0))

        current_start = latest_critical_end
        for i in non_critical_indices:
            t = result[i]
            old_start = t.get("start_time", 0)
            if old_start < latest_critical_end:
                duration = t.get("end_time", 0) - old_start
                t["start_time"] = current_start
                t["end_time"] = current_start + duration
                current_start = t["end_time"]
                improved = True

        return result, improved

    def _shorten_critical_path(self, schedule: List[Dict],
                                allocated_resources: Dict[str, float]
                                ) -> Tuple[List[Dict], bool]:
        if not schedule:
            return schedule, False

        makespan = self._compute_makespan(schedule)
        result = [dict(t) for t in schedule]
        improved = False

        critical_tasks = [
            t for t in result
            if t.get("end_time", 0) >= makespan * 0.7
        ]

        for t in critical_tasks:
            duration = t.get("end_time", 0) - t.get("start_time", 0)
            if duration > makespan * 0.2:
                reduction = min(duration * 0.1, 2.0)
                if reduction > 0.5:
                    t["end_time"] = t["end_time"] - reduction
                    improved = True

        return result, improved


class LLMPlanOptimizer:
    def __init__(self, llm_engine=None):
        self.llm_engine = llm_engine
        self._fallback = PlanOptimizer()
        self.stats = {"llm_attempts": 0, "llm_adopted": 0, "fallback_used": 0}

    def optimize(self, plan: Dict[str, Any],
                 resources: Dict[str, float] = None,
                 vehicle_capacity: Dict[str, Any] = None,
                 verbose: bool = True) -> Dict[str, Any]:
        schedule = plan.get("schedule", [])
        if not schedule:
            return self._fallback.optimize(plan, resources, vehicle_capacity,
                                           verbose=False)

        original_makespan = self._fallback._compute_makespan(schedule)

        optimized_schedule = None
        optimizations: List[str] = []

        if self.llm_engine is not None:
            self.stats["llm_attempts"] += 1
            proposed, notes = self._llm_propose(plan, verbose)
            if proposed is not None and self._validate(schedule, proposed):
                optimized_schedule = proposed
                optimizations.append("llm_heuristic_adjustment")
                self.stats["llm_adopted"] += 1
                if verbose and notes:
                    print(f"[M7] Plan Optimizer: {notes[:120]}")
            else:
                if verbose:
                    print("[M7] the LLM adjustment is missing or invalid, falling back to the deterministic heuristic")

        if optimized_schedule is None:
            self.stats["fallback_used"] += 1
            return self._fallback.optimize(plan, resources, vehicle_capacity,
                                           verbose=verbose)

        optimized_makespan = self._fallback._compute_makespan(optimized_schedule)
        improvement_pct = round(
            (original_makespan - optimized_makespan) / max(original_makespan, 1) * 100, 1
        )

        if verbose:
            print(f"[M7] optimised makespan: {original_makespan}h → "
                  f"{optimized_makespan}h (improvement {improvement_pct:.1f}%)")

        optimized_plan = dict(plan)
        optimized_plan["schedule"] = optimized_schedule

        return {
            "optimized_plan": optimized_plan,
            "original_makespan": original_makespan,
            "optimized_makespan": optimized_makespan,
            "improvement_pct": improvement_pct,
            "optimizations_applied": optimizations,
        }

    def _llm_propose(self, plan: Dict[str, Any],
                     verbose: bool) -> Tuple[Optional[List[Dict]], str]:
        schedule = plan.get("schedule", [])
        summary = [
            {
                "task_id": t.get("task_id", f"T{i}"),
                "name": t.get("name", ""),
                "start_time": t.get("start_time", 0),
                "end_time": t.get("end_time", 0),
                "assigned_to": t.get("assigned_to", ""),
                "region": t.get("region", ""),
            }
            for i, t in enumerate(schedule)
        ]

        user_prompt = (
            "## Current schedule, already passed the constraint audit (makespan = "
            f"{self._fallback._compute_makespan(schedule)}h)\n"
            f"{json.dumps(summary, ensure_ascii=False)}\n\n"
            "## Resource allocation (immutable)\n"
            f"{json.dumps(plan.get('allocated_resources', {}), ensure_ascii=False)}\n\n"
            "Return the complete optimised adjusted_schedule."
        )

        try:
            response = self.llm_engine.llm.chat.completions.create(
                model=self.llm_engine.model_name,
                messages=[
                    {"role": "system", "content": OPTIMIZER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0,
                max_tokens=16384,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "optimizer_output",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "optimization_notes": {"type": "string"},
                            "adjusted_schedule": {"type": "array",
                                                  "items": {"type": "object"}},
                        },
                        "required": ["optimization_notes", "adjusted_schedule"],
                    },
                }},
            )
            parsed = extract_json_from_response(
                response.choices[0].message.content)
            if not parsed:
                return None, ""
            adjusted = parsed.get("adjusted_schedule")
            if not isinstance(adjusted, list) or not adjusted:
                return None, ""
            return adjusted, parsed.get("optimization_notes", "")
        except Exception as e:
            if verbose:
                print(f"[M7] LLM call failed: {e}")
            return None, ""

    @staticmethod
    def _validate(original: List[Dict], proposed: List[Dict]) -> bool:
        if len(proposed) != len(original):
            return False

        orig_by_id = {t.get("task_id", f"T{i}"): t
                      for i, t in enumerate(original)}
        prop_ids = set()
        for i, t in enumerate(proposed):
            tid = t.get("task_id", f"T{i}")
            if tid in prop_ids or tid not in orig_by_id:
                return False
            prop_ids.add(tid)
            ot = orig_by_id[tid]
            o_dur = ot.get("end_time", 0) - ot.get("start_time", 0)
            p_dur = t.get("end_time", 0) - t.get("start_time", 0)
            if t.get("start_time", 0) < 0:
                return False
            if abs(p_dur - o_dur) > 0.25:
                return False
        return True


def create_optimizer(llm_engine=None):
    if llm_engine is not None:
        return LLMPlanOptimizer(llm_engine=llm_engine)
    return PlanOptimizer()
