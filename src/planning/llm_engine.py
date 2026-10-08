"""M4: LLM planning engine.

Builds the planning prompt from the task description, the global strategy and
the retrieved constraint rules, calls the OpenAI-compatible endpoint under a
JSON schema, and parses the response into a structured plan (allocated
resources, personnel, schedule and the evidence fields used by the auditor)."""

import time
from typing import Dict, Any, List

from openai import OpenAI

try:
    from ..agents.prompts import (
        CRAFT_SYSTEM_PROMPT,
        build_task_description,
        extract_json_from_response,
    )
    from ..config import (
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL,
    )
except ImportError:
    from agents.prompts import (
        CRAFT_SYSTEM_PROMPT,
        build_task_description,
        extract_json_from_response,
    )
    from config import (
        LLM_API_KEY,
        LLM_BASE_URL,
        LLM_MODEL,
    )


class LLMPlanningEngine:
    def __init__(self,
                 model: str = None,
                 base_url: str = None,
                 api_key: str = None,
                 temperature: float = 0.0,
                 max_retries: int = 2):
        self.model_name = model or LLM_MODEL
        self.base_url = base_url or LLM_BASE_URL
        self.api_key = api_key or LLM_API_KEY
        self.temperature = temperature
        self.max_retries = max_retries

        self.llm = OpenAI(
            api_key=self.api_key or LLM_API_KEY,
            base_url=self.base_url or LLM_BASE_URL,
        )
        self.model_name = self.model_name or LLM_MODEL

        self.stats = {
            "total_calls": 0,
            "successful_parses": 0,
            "failed_parses": 0,
            "total_tokens": 0,
            "total_time": 0.0,
        }

    def plan(self,
             task_description: str,
             constraints_text: str = "",
             goals: list = None,
             resources: dict = None,
             regions: list = None,
             verbose: bool = True) -> Dict[str, Any]:
        self.stats["total_calls"] += 1

        full_task = build_task_description(
            task_brief=task_description,
            goals=goals,
            resources=resources,
            constraints_text=constraints_text,
            regions=regions,
        )

        if verbose:
            prompt_len = len(full_task)
            constraint_len = len(constraints_text)
            print(f"[M4] prompt built: {prompt_len} chars "
                  f"(constraints {constraint_len} chars)")

        start_time = time.time()
        result = None

        for attempt in range(self.max_retries + 1):
            try:
                raw_output = self._call_llm(full_task)
                parsed = extract_json_from_response(raw_output)

                if parsed:
                    self.stats["successful_parses"] += 1
                    elapsed = time.time() - start_time
                    self.stats["total_time"] += elapsed

                    region_plans = parsed.get("region_plans", {})

                    alloc = dict(parsed.get("allocated_resources", {}))
                    personnel = dict(parsed.get("assigned_personnel", {}))
                    all_schedules = list(parsed.get("schedule", []))
                    resp_times = dict(parsed.get("region_response_times", {}))
                    all_carriers = list(parsed.get("carriers", []) or [])
                    all_events = dict(parsed.get("events", {}) or {})

                    for r_id, r_plan in region_plans.items():
                        for k, v in r_plan.get("allocated_resources", {}).items():
                            alloc[k] = alloc.get(k, 0) + v
                        for k, v in r_plan.get("assigned_personnel", {}).items():
                            personnel[k] = personnel.get(k, 0) + v
                        all_schedules.extend(r_plan.get("schedule", []))
                        all_carriers.extend(r_plan.get("carriers", []) or [])
                        for ek, ev in (r_plan.get("events", {}) or {}).items():
                            all_events.setdefault(ek, ev)
                        if "response_time_hours" in r_plan:
                            resp_times[r_id] = r_plan["response_time_hours"]

                    result = {
                        "plan": parsed.get("plan", ""),
                        "region_plans": region_plans,
                        "allocated_resources": alloc,
                        "assigned_personnel": personnel,
                        "schedule": all_schedules,
                        "carriers": all_carriers,
                        "events": all_events,
                        "region_response_times": resp_times,
                        "task_completion_times": parsed.get("task_completion_times", {}),
                        "goals_achieved": parsed.get("goals_achieved", []),
                        "addressed_constraints": parsed.get("addressed_constraints", []),
                        "raw_output": raw_output,
                    }

                    if verbose:
                        alloc_count = len(result["allocated_resources"])
                        sched_count = len(result["schedule"])
                        print(f"[M4] planning done: {alloc_count} resource allocations, "
                              f"{sched_count} scheduled tasks, "
                              f"elapsed {elapsed:.1f}s")
                        if alloc_count == 0:
                            print(f"[M4] WARNING: the resource allocation is empty! raw[:500]={raw_output[:500]}")

                    break
                else:
                    if verbose:
                        print(f"[M4] JSON parsing failed (attempt {attempt + 1}), "
                              f"raw[:300]={raw_output[:300]}")

            except Exception as e:
                if verbose:
                    print(f"[M4] LLM call failed (attempt {attempt + 1}): {e}")

            if attempt < self.max_retries:
                time.sleep(0.5)

        if result is None:
            self.stats["failed_parses"] += 1
            elapsed = time.time() - start_time
            self.stats["total_time"] += elapsed
            result = {
                "plan": "",
                "region_plans": {},
                "allocated_resources": {},
                "assigned_personnel": {},
                "schedule": [],
                "carriers": [],
                "events": {},
                "region_response_times": {},
                "task_completion_times": {},
                "goals_achieved": [],
                "addressed_constraints": [],
                "raw_output": "",
                "_error": "All LLM calls failed to produce valid JSON",
            }

        return result

    def _call_llm(self, task_description: str) -> str:
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": CRAFT_SYSTEM_PROMPT},
                {"role": "user", "content": task_description},
            ],
            temperature=self.temperature,
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
                        "carriers": {"type": "array", "items": {"type": "object"}},
                        "events": {"type": "object"},
                        "region_response_times": {"type": "object"},
                    },
                    "required": ["plan", "goals_achieved", "addressed_constraints",
                                 "allocated_resources", "assigned_personnel",
                                 "schedule", "carriers", "events",
                                 "region_response_times"],
                },
            }},
        )
        return response.choices[0].message.content

    def plan_batch(self, sub_tasks: List[Dict[str, Any]],
                   verbose: bool = True) -> List[Dict[str, Any]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def _plan_one(i, sub):
            region = sub.get("region", {})
            region_id = region.get("id", f"subtask_{i}")
            if verbose:
                print(f"\n[M4] planning subtask {i + 1}/{len(sub_tasks)}: {region_id}")
            plan = self.plan(
                task_description=sub.get("task_desc", ""),
                constraints_text=sub.get("constraints_text", ""),
                regions=[region],
                verbose=verbose,
            )
            return i, {"region": region_id, "region_info": region, "plan": plan}

        results = [None] * len(sub_tasks)
        with ThreadPoolExecutor(max_workers=min(len(sub_tasks), 4)) as executor:
            futures = {executor.submit(_plan_one, i, sub): i for i, sub in enumerate(sub_tasks)}
            for future in as_completed(futures):
                i, result = future.result()
                results[i] = result

        if verbose:
            print(f"[M4] parallel planning done: {len(sub_tasks)} subtasks")
        return results

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self.stats,
            "parse_success_rate": (
                self.stats["successful_parses"] / max(self.stats["total_calls"], 1)
            ),
            "avg_time_per_call": (
                self.stats["total_time"] / max(self.stats["total_calls"], 1)
            ),
        }


def create_llm_engine(
    model: str = None,
    base_url: str = None,
    api_key: str = None,
) -> LLMPlanningEngine:
    return LLMPlanningEngine(
        model=model or LLM_MODEL,
        base_url=base_url or LLM_BASE_URL,
        api_key=api_key or LLM_API_KEY,
    )
