"""Planning-layer orchestrator (M1 + M4).

Implements the three planning modes - global (CRAFT-C), decomposed (CRAFT-P)
and collaborative (CRAFT-O / -A / full) - covering the global-strategy phase,
the parallel regional planning phase and the resource-arbitration phase."""

from typing import Dict, Any, List, Optional

try:
    from .decomposer import PlanDecomposer, merge_sub_plans
    from .llm_engine import LLMPlanningEngine, create_llm_engine
    from ..agents.roles import get_role
    from ..agents.prompts import (
        GLOBAL_STRATEGY_SYSTEM_PROMPT,
        REGIONAL_EXECUTION_SYSTEM_PROMPT,
        ARBITRATION_SYSTEM_PROMPT,
        build_global_strategy_prompt,
        build_regional_execution_prompt,
        build_arbitration_prompt,
        extract_json_from_response,
    )
except ImportError:
    from planning.decomposer import PlanDecomposer, merge_sub_plans
    from planning.llm_engine import LLMPlanningEngine, create_llm_engine
    from agents.roles import get_role
    from agents.prompts import (
        GLOBAL_STRATEGY_SYSTEM_PROMPT,
        REGIONAL_EXECUTION_SYSTEM_PROMPT,
        ARBITRATION_SYSTEM_PROMPT,
        build_global_strategy_prompt,
        build_regional_execution_prompt,
        build_arbitration_prompt,
        extract_json_from_response,
    )


class CRAFTPlanningLayer:
    def __init__(self,
                 input_layer=None,
                 llm_engine: Optional[LLMPlanningEngine] = None,
                 top_k: int = 15):
        self.input_layer = input_layer
        self.llm_engine = llm_engine or create_llm_engine()
        self.top_k = top_k
        self.decomposer = PlanDecomposer(input_layer=input_layer)

        self.stats = {
            "total_plans": 0,
            "global_plans": 0,
            "decomposed_plans": 0,
            "collaborative_plans": 0,
            "arbitration_calls": 0,
        }

    def plan(self,
             task_description: str,
             goals: list = None,
             resources: dict = None,
             regions: list = None,
             mode: str = "decomposed",
             constraints_text: str = "",
             verbose: bool = True) -> Dict[str, Any]:
        self.stats["total_plans"] += 1

        mode_labels = {
            "global": "global planning",
            "decomposed": "decomposed planning",
            "collaborative": "collaborative planning (strategy -> execution -> reconciliation)",
        }

        if verbose:
            print(f"\n{'='*60}")
            print(f"[CRAFT planning layer] {mode_labels.get(mode, mode)}")
            print(f"[CRAFT planning layer] task: {task_description[:80]}...")
            print(f"{'='*60}")

        if mode == "global":
            return self._plan_global(task_description, goals, resources,
                                      regions, verbose, constraints_text)
        elif mode == "collaborative":
            return self._plan_collaborative(task_description, goals,
                                             resources, verbose, constraints_text)
        else:
            return self._plan_decomposed(task_description, goals,
                                          resources, verbose)

    def _plan_global(self, task_description: str,
                      goals: list, resources: dict,
                      regions: list, verbose: bool,
                      constraints_text: str = "") -> Dict[str, Any]:
        self.stats["global_plans"] += 1

        if not constraints_text:
            constraints_text = self._retrieve_constraints(task_description,
                                                           verbose=verbose)

        if verbose:
            print("[CRAFT planning layer] LLM global planning...")

        result = self.llm_engine.plan(
            task_description=task_description,
            constraints_text=constraints_text,
            goals=goals,
            resources=resources,
            regions=regions,
            verbose=verbose,
        )
        result["_mode"] = "global"
        return result

    def _plan_decomposed(self, task_description: str,
                          goals: list, resources: dict,
                          verbose: bool) -> Dict[str, Any]:
        self.stats["decomposed_plans"] += 1

        sub_tasks = self.decomposer.decompose(
            task_description=task_description,
            goals=goals,
            resources=resources,
            top_k=self.top_k,
        )

        if len(sub_tasks) == 1:
            if verbose:
                print("[CRAFT planning layer] a single region, falling back to the global mode")
            return self._plan_global(task_description, goals, resources,
                                      [sub_tasks[0]["region"]], verbose)

        if verbose:
            print(f"[CRAFT planning layer] decomposed into {len(sub_tasks)} subtasks, planning them independently...")

        sub_plans = self.llm_engine.plan_batch(sub_tasks, verbose=verbose)
        merged = merge_sub_plans(sub_plans, global_resources=resources)
        merged["_mode"] = "decomposed"
        merged["_sub_task_count"] = len(sub_tasks)

        arbitration = self._phase_arbitrate(sub_plans, resources, verbose=verbose)
        if arbitration:
            merged = self._apply_arbitration(merged, arbitration)

        return merged

    def _plan_collaborative(self, task_description: str,
                             goals: list, resources: dict,
                             verbose: bool,
                             constraints_text: str = "") -> Dict[str, Any]:
        self.stats["collaborative_plans"] += 1

        regions = self.decomposer.extract_regions(task_description)
        if not regions:
            print("[CRAFT planning layer] no multi-region structure detected, falling back to the global mode")
            return self._plan_global(task_description, goals, resources,
                                      None, verbose, constraints_text)

        if verbose:
            print("[CRAFT planning layer] Phase 1/4: formulating the global strategy...")

        global_strategy = self._phase1_global_strategy(
            task_description, resources, regions, verbose, constraints_text)

        if verbose:
            ratios = global_strategy.get("resource_allocation_ratios", {})
            for r_id, r_info in ratios.items():
                print(f"  {r_id}: {r_info.get('ratio', 0):.0%} "
                      f"({r_info.get('reason', '')[:40]})")

        if verbose:
            print(f"[CRAFT planning layer] Phase 2/4: {len(regions)} regions planning...")

        regional_results = self._phase2_regional_execution(
            task_description, regions, global_strategy, resources, verbose,
            constraints_text)

        arbitration = self._phase_arbitrate(
            regional_results, resources, constraints_text, verbose)

        if verbose:
            print("[CRAFT planning layer] Phase 4/4: reconciliation and merge...")

        final_plan = self._phase3_reconcile(
            global_strategy, regional_results, resources, regions, verbose)

        if arbitration:
            final_plan = self._apply_arbitration(final_plan, arbitration)

        final_plan["_mode"] = "collaborative"
        final_plan["_global_strategy"] = global_strategy
        final_plan["_regional_results"] = [
            {"region": r["region"], "plan_summary": r["plan"].get("plan", "")[:80]}
            for r in regional_results
        ]

        return final_plan

    def _phase1_global_strategy(self, task_description: str,
                                 resources: dict, regions: list,
                                 verbose: bool,
                                 constraints_text: str = "") -> Dict[str, Any]:
        if not constraints_text:
            constraints_text = self._retrieve_constraints(task_description,
                                                           verbose=False)

        prompt = build_global_strategy_prompt(
            task_description=task_description,
            constraints_text=constraints_text,
            resources=resources,
            regions=regions,
        )

        if verbose:
            print(f"  [Phase 1] Prompt: {len(prompt)} chars, calling the LLM...")

        response = self.llm_engine.llm.chat.completions.create(
            model=self.llm_engine.model_name,
            messages=[
                {"role": "system", "content": GLOBAL_STRATEGY_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=16384,
            response_format={"type": "json_schema", "json_schema": {
                "name": "global_strategy",
                "schema": {
                    "type": "object",
                    "properties": {
                        "strategy_summary": {"type": "string"},
                        "resource_allocation_ratios": {"type": "object"},
                        "priority_ordering": {"type": "array", "items": {"type": "string"}},
                        "cross_region_strategy": {"type": "string"},
                        "critical_deadlines": {"type": "object"},
                    },
                    "required": ["strategy_summary", "resource_allocation_ratios",
                                 "priority_ordering", "cross_region_strategy",
                                 "critical_deadlines"],
                },
            }},
        )
        self.llm_engine.stats["total_calls"] += 1
        strategy = extract_json_from_response(response.choices[0].message.content)

        if strategy:
            self.llm_engine.stats["successful_parses"] += 1
        else:
            strategy = self._fallback_strategy(regions)

        return strategy

    def _fallback_strategy(self, regions: list) -> Dict[str, Any]:
        weights = {"critical": 3.0, "urgent": 2.0, "normal": 1.0, "routine": 0.5}
        weighted = {}
        total_weight = 0

        for r in regions:
            sev = r.get("severity", "urgent")
            pop = r.get("population", 1)
            w = weights.get(sev, 1.0) * pop
            weighted[r["id"]] = w
            total_weight += w

        ratios = {}
        for r_id, w in weighted.items():
            ratios[r_id] = {
                "ratio": round(w / total_weight, 2) if total_weight > 0 else 0.5,
                "priority": next((r["severity"] for r in regions if r["id"] == r_id), "urgent"),
                "reason": f"weighted by severity and population: {w:.0f}/{total_weight:.0f}",
            }

        priority_order = sorted(ratios.keys(),
                                key=lambda rid: ratios[rid]["ratio"],
                                reverse=True)

        return {
            "strategy_summary": f"rule-based allocation: {len(regions)} regions weighted by severity + population",
            "resource_allocation_ratios": ratios,
            "priority_ordering": priority_order,
            "cross_region_strategy": "respond in priority order",
            "critical_deadlines": {
                rid: 2 if any(r["severity"] == "critical" for r in regions if r["id"] == rid) else 6
                for rid in [r["id"] for r in regions]
            },
            "key_principles": ["allocate on demand", "severity first", "fair and impartial"],
        }

    def _phase2_regional_execution(self, task_description: str,
                                    regions: list,
                                    global_strategy: dict,
                                    resources: dict,
                                    verbose: bool,
                                    constraints_text: str = "") -> List[Dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        shared_constraints = constraints_text or self._retrieve_constraints(
            task_description, verbose=False)

        def _plan_one(idx: int, region: dict) -> dict:
            region_id = region["id"]
            ratio = global_strategy.get("resource_allocation_ratios", {}) \
                .get(region_id, {}).get("ratio", 0)
            if verbose:
                print(f"  [{idx + 1}/{len(regions)}] {region_id}: quota {ratio:.0%}")

            prompt = build_regional_execution_prompt(
                region=region,
                global_strategy=global_strategy,
                region_constraints=shared_constraints,
                total_resources=resources,
            )

            response_text = self._call_llm_json(
                system_prompt=REGIONAL_EXECUTION_SYSTEM_PROMPT,
                user_prompt=prompt,
            )
            plan = extract_json_from_response(response_text)
            if not plan:
                plan = {"plan": f"{region_id} planning failed", "allocated_resources": {},
                        "assigned_personnel": {}, "schedule": []}

            return {
                "idx": idx,
                "region": region_id,
                "region_info": region,
                "plan": plan,
                "raw_output": response_text,
            }

        results = [None] * len(regions)
        with ThreadPoolExecutor(max_workers=min(len(regions), 4)) as executor:
            futures = {executor.submit(_plan_one, i, r): i for i, r in enumerate(regions)}
            for future in as_completed(futures):
                item = future.result()
                results[item["idx"]] = item
                self.llm_engine.stats["total_calls"] += 1
                if item["plan"].get("plan") and "planning failed" not in item["plan"].get("plan", ""):
                    self.llm_engine.stats["successful_parses"] += 1

        return results

    def _phase_arbitrate(self,
                         regional_results: List[Dict[str, Any]],
                         resources: dict,
                         constraints_text: str = "",
                         verbose: bool = True) -> Optional[Dict[str, Any]]:
        if len(regional_results or []) < 2:
            return None

        role = get_role("resource_arbitrator")
        system_prompt = role.system_prompt if role else ARBITRATION_SYSTEM_PROMPT

        prompt = build_arbitration_prompt(
            regional_results=regional_results,
            total_resources=resources,
            constraints_text=constraints_text,
        )

        if verbose:
            print(f"[CRAFT planning layer] Phase 3/4: resource arbitration"
                  f" ({len(regional_results)} regions)...")

        try:
            response = self.llm_engine.llm.chat.completions.create(
                model=self.llm_engine.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=16384,
                response_format={"type": "json_schema", "json_schema": {
                    "name": "arbitration",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "arbitration_summary": {"type": "string"},
                            "final_allocation": {"type": "object"},
                            "conflicts_resolved": {
                                "type": "array", "items": {"type": "object"}},
                            "escalations": {
                                "type": "array", "items": {"type": "object"}},
                        },
                        "required": ["arbitration_summary", "final_allocation",
                                     "conflicts_resolved", "escalations"],
                    },
                }},
            )
        except Exception as e:
            if verbose:
                print(f"  [Arbitrator] call failed, falling back to the deterministic merge: {e}")
            return None

        self.llm_engine.stats["total_calls"] += 1
        arbitration = extract_json_from_response(response.choices[0].message.content)
        if not arbitration:
            if verbose:
                print("  [Arbitrator] the output could not be parsed, falling back to the deterministic merge")
            return None

        self.llm_engine.stats["successful_parses"] += 1
        self.stats["arbitration_calls"] += 1

        if verbose:
            print(f"  [Arbitrator] {arbitration.get('arbitration_summary', '')[:60]}")
            print(f"  [Arbitrator] conflicts resolved: {len(arbitration.get('conflicts_resolved') or [])}; "
                  f"escalated {len(arbitration.get('escalations') or [])} items")

        return arbitration

    @staticmethod
    def _apply_arbitration(plan: Dict[str, Any],
                           arbitration: Dict[str, Any]) -> Dict[str, Any]:
        allocation = arbitration.get("final_allocation") or {}
        if not isinstance(allocation, dict):
            allocation = {}

        region_plans = plan.get("region_plans", {})
        merged_resources: Dict[str, Any] = {}
        for rid, region_alloc in allocation.items():
            if not isinstance(region_alloc, dict):
                continue
            if rid in region_plans:
                region_plans[rid]["allocated_resources"] = dict(region_alloc)
            for k, v in region_alloc.items():
                if isinstance(v, (int, float)):
                    merged_resources[k] = merged_resources.get(k, 0) + v

        if merged_resources:
            plan["allocated_resources"] = merged_resources

        plan["_arbitration"] = {
            "summary": arbitration.get("arbitration_summary", ""),
            "final_allocation": allocation,
            "conflicts_resolved": arbitration.get("conflicts_resolved", []),
            "escalations": arbitration.get("escalations", []),
        }
        return plan

    def _phase3_reconcile(self, global_strategy: dict,
                           regional_results: list,
                           resources: dict,
                           regions: list,
                           verbose: bool) -> Dict[str, Any]:
        ratios = global_strategy.get("resource_allocation_ratios", {})

        actual_by_region = {}
        all_resources = set()
        all_personnel = set()
        all_schedules = []
        all_carriers = []
        all_events = {}
        all_response_times = {}
        all_addressed = []

        for rr in regional_results:
            rid = rr["region"]
            plan = rr["plan"]
            actual = {
                "resources": dict(plan.get("allocated_resources", {})),
                "personnel": dict(plan.get("assigned_personnel", {})),
                "schedule": plan.get("schedule", []),
                "alignment": plan.get("global_strategy_alignment", "unknown"),
            }
            actual_by_region[rid] = actual
            for ac in plan.get("addressed_constraints", []):
                if ac not in all_addressed:
                    all_addressed.append(ac)

            for k in actual["resources"]:
                all_resources.add(k)
            for k in actual["personnel"]:
                all_personnel.add(k)
            all_schedules.extend(actual["schedule"])
            all_carriers.extend(plan.get("carriers", []) or [])
            for _ek, _ev in (plan.get("events", {}) or {}).items():
                all_events.setdefault(_ek, _ev)

            rt = plan.get("region_response_times", {})
            if rt:
                all_response_times.update(rt)

        deviations = []
        for res_name in all_resources:
            total_used = sum(
                actual_by_region[rid]["resources"].get(res_name, 0)
                for rid in actual_by_region
            )
            total_available = resources.get(res_name, total_used) if resources else total_used

            for rid in actual_by_region:
                actual_amt = actual_by_region[rid]["resources"].get(res_name, 0)
                target_ratio = ratios.get(rid, {}).get("ratio", 0.5)
                target_amt = total_available * target_ratio

                if target_amt > 0 and abs(actual_amt - target_amt) / target_amt > 0.2:
                    deviations.append({
                        "region": rid,
                        "resource": res_name,
                        "strategic_target": round(target_amt, 1),
                        "actual": actual_amt,
                        "deviation_pct": round((actual_amt - target_amt) / target_amt * 100, 1),
                    })

        if deviations and verbose:
            print(f"  [Phase 4] found {len(deviations)} deviations:")
            for d in deviations[:5]:
                print(f"    {d['region']}/{d['resource']}: "
                      f"target {d['strategic_target']} vs actual {d['actual']} "
                      f"({d['deviation_pct']:+.1f}%)")

        merged_resources = {}
        merged_personnel = {}
        for res_name in all_resources:
            merged_resources[res_name] = sum(
                actual_by_region[rid]["resources"].get(res_name, 0)
                for rid in actual_by_region
            )
        for pers_name in all_personnel:
            merged_personnel[pers_name] = sum(
                actual_by_region[rid]["personnel"].get(pers_name, 0)
                for rid in actual_by_region
            )

        region_summaries = []
        for rid in actual_by_region:
            r_ratio = ratios.get(rid, {}).get("ratio", 0)
            r_actual = actual_by_region[rid]
            region_summaries.append(
                f"{rid} (quota {r_ratio:.0%}): "
                f"{len(r_actual['resources'])} resources, "
                f"{len(r_actual['schedule'])} tasks"
            )

        if verbose:
            print(f"  [Phase 4] reconciliation done: {len(region_summaries)} regions, "
                  f"{len(all_schedules)} scheduled tasks, {len(deviations)} deviations")

        return {
            "plan": "collaborative plan: " + " | ".join(region_summaries),
            "allocated_resources": merged_resources,
            "assigned_personnel": merged_personnel,
            "schedule": all_schedules,
            "carriers": all_carriers,
            "events": all_events,
            "region_response_times": all_response_times,
            "addressed_constraints": all_addressed,
            "region_plans": {
                rid: {
                    "allocated_resources": actual_by_region[rid]["resources"],
                    "assigned_personnel": actual_by_region[rid]["personnel"],
                    "schedule": actual_by_region[rid]["schedule"],
                }
                for rid in actual_by_region
            },
            "_reconciliation": {
                "deviations": deviations,
                "strategy": global_strategy.get("strategy_summary", ""),
            },
        }

    def _call_llm_json(self, system_prompt: str,
                                 user_prompt: str) -> str:
        response = self.llm_engine.llm.chat.completions.create(
            model=self.llm_engine.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
            max_tokens=16384,
            response_format={"type": "json_schema", "json_schema": {
                "name": "craft_plan",
                "schema": {
                    "type": "object",
                    "properties": {
                        "plan": {"type": "string"},
                        "goals_achieved": {"type": "array", "items": {"type": "string"}},
                        "addressed_constraints": {"type": "array", "items": {"type": "string"}},
                        "allocated_resources": {"type": "object"},
                        "assigned_personnel": {"type": "object"},
                        "schedule": {"type": "array", "items": {"type": "object"}},
                        "region_response_times": {"type": "object"},
                    },
                    "required": ["plan", "goals_achieved", "addressed_constraints",
                                 "allocated_resources", "assigned_personnel",
                                 "schedule", "region_response_times"],
                },
            }},
        )
        return response.choices[0].message.content

    def _retrieve_constraints(self, query: str, verbose: bool = False) -> str:
        if not self.input_layer:
            return ""
        try:
            return self.input_layer.retrieve(query, top_k=self.top_k,
                                              verbose=verbose)
        except Exception as e:
            if verbose:
                print(f"[CRAFT planning layer] constraint retrieval failed: {e}")
            return ""

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "engine_stats": self.llm_engine.get_stats(),
        }


def create_planning_layer(
    input_layer=None,
    model: str = None,
    base_url: str = None,
    api_key: str = None,
    top_k: int = 15,
) -> CRAFTPlanningLayer:
    engine = create_llm_engine(
        model=model,
        base_url=base_url,
        api_key=api_key,
    )
    return CRAFTPlanningLayer(
        input_layer=input_layer,
        llm_engine=engine,
        top_k=top_k,
    )
