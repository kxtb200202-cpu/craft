"""M5: the Constraint Auditor - deterministic rule-by-rule verification.

Numeric rules are checked by interval comparison against bounds normalized from
capacity, per-capita ratio and deadline parameters; non-numeric rules are
checked by evaluating the structured predicate stored in the knowledge base
over the plan's evidence fields. Verdicts are three-valued (PASS / FAIL /
UNDEFINED) and no LLM is involved, so repeated audits of a plan agree."""

import math
from typing import List, Dict, Any, Optional


class ConstraintAuditor:
    NUMERIC_TYPES = ("resource", "capacity", "deadline")

    PREDICATE_OPS = ("no_cooccur", "before", "precedes_event")

    def __init__(self, constraints: List[Dict[str, Any]] = None,
                 llm_engine=None):
        self.constraints = constraints or []
        self.llm_engine = llm_engine
        self.stats = {"predicate_checks": 0, "predicate_undefined": 0}

    def audit(self, plan: Dict[str, Any],
              constraints: List[Dict[str, Any]] = None,
              task_context: Dict[str, Any] = None) -> Dict[str, Any]:
        constraints = constraints or self.constraints
        if not constraints:
            return {"total": 0, "passed": [], "violated": [],
                    "not_applicable": [], "compliance_rate": 1.0,
                    "critical_violations": []}

        allocated = plan.get("allocated_resources", {})
        personnel = plan.get("assigned_personnel", {})
        response_times = plan.get("region_response_times", {})
        schedule = plan.get("schedule", [])
        region_plans = plan.get("region_plans", {})

        if not allocated and region_plans:
            for rid, rplan in region_plans.items():
                for k, v in rplan.get("allocated_resources", {}).items():
                    allocated[k] = allocated.get(k, 0) + v
                for k, v in rplan.get("assigned_personnel", {}).items():
                    personnel[k] = personnel.get(k, 0) + v

        passed = []
        violated = []
        not_addressed = []
        not_applicable = []
        undefined = []

        for c in constraints:
            cid = c.get("id", "unknown")
            ctype = c.get("type", "")
            cdesc = c.get("desc", "")
            params = c.get("params", {})

            is_addressed = self._is_constraint_addressed(c, allocated, personnel)

            if ctype == "resource":
                result = self._check_resource(cid, cdesc, allocated, params,
                                               task_context)
            elif ctype == "capacity":
                result = self._check_capacity(cid, cdesc, personnel, allocated,
                                               params, task_context)
            elif ctype == "deadline":
                result = self._check_deadline(cid, cdesc, response_times,
                                               params, schedule)
                is_addressed = True
            else:
                verdict = self._check_predicate(c, plan, task_context)
                status = verdict["status"]
                if status == "PASS":
                    passed.append(verdict["record"])
                elif status == "FAIL":
                    violated.append(verdict["record"])
                elif status == "NOT_ADDRESSED":
                    not_addressed.append(verdict["record"])
                else:
                    self.stats["predicate_undefined"] += 1
                    undefined.append(verdict["record"])
                continue

            if result is None:
                not_applicable.append({
                    "constraint_id": cid, "desc": cdesc,
                    "reason": "cannot be verified (required data missing)"
                })
            elif not is_addressed:
                not_addressed.append({
                    "constraint_id": cid, "desc": cdesc,
                    "reason": "the plan does not involve the resource / personnel covered by this rule"
                })
            elif result.get("passed"):
                passed.append(result)
            else:
                violated.append(result)

        addressed = len(passed) + len(violated)
        total = addressed + len(not_addressed)
        denominator = len(passed) + len(violated) + len(undefined)
        compliance_rate = len(passed) / denominator if denominator > 0 else 0.0

        critical = [v for v in violated
                    if abs(v.get("deviation_pct") or 0) > 50
                    or any(kw in v.get("constraint_id", "")
                           for kw in ["water", "medical", "food", "critical", "rescue"])]

        return {
            "total": total,
            "passed": passed,
            "violated": violated,
            "not_addressed": not_addressed,
            "not_applicable": not_applicable,
            "undefined": undefined,
            "undefined_count": len(undefined),
            "compliance_rate": round(compliance_rate, 4),
            "critical_violations": critical,
        }

    def _check_resource(self, cid: str, cdesc: str,
                         allocated: Dict[str, float],
                         params: Dict[str, Any],
                         task_context: Dict = None) -> Optional[Dict]:
        limits = params.get("resources", {})
        for res_name, max_val in limits.items():
            actual = allocated.get(res_name, 0)
            if isinstance(actual, (int, float)) and isinstance(max_val, (int, float)):
                if actual > max_val:
                    return self._violation(cid, cdesc, res_name, actual, max_val,
                                           "exceeds_resource_limit")

        ratios = params.get("ratios", {})
        total_pop = self._get_population(task_context)
        if total_pop > 0:
            ratio_checks = {
                "water_liters_per_person_per_day": ("water", total_pop, "multiply"),
                "water_liters_per_person_per_day_survival": ("water", total_pop, "multiply"),
                "water_liters_per_person_per_day_standard": ("water", total_pop, "multiply"),
                "food_kg_per_person_per_day": ("food", total_pop, "multiply"),
                "kcal_per_person_per_day": ("food", None, None),
                "cereal_grams_per_person_per_day": ("food", total_pop / 1000, "multiply"),
                "medical_supplies_kg_per_100_patients": ("medical_supplies", total_pop / 100, "multiply"),
                "first_aid_kits_per_100_patients": ("first_aid_kits", total_pop / 100, "multiply"),
                "essential_medicines_courses_per_100_patients": ("medical_supplies", total_pop / 100, "multiply"),
                "blankets_per_person": ("blankets", total_pop, "multiply"),
                "sleeping_mats_per_person": ("sleeping_mats", total_pop, "multiply"),
                "mosquito_nets_per_person": ("mosquito_nets", total_pop, "multiply"),
                "covered_area_m2_per_person": ("shelter_materials", total_pop, "multiply"),
                "winter_blankets_per_person": ("blankets", total_pop, "multiply"),
                "winter_jackets_per_person": ("winter_jackets", total_pop, "multiply"),
                "infant_formula_grams_per_infant_per_day": ("infant_formula_kg", total_pop * 0.1, "multiply_grams"),
                "rutf_sachets_per_severely_malnourished_child_per_day": ("ready_to_use_therapeutic_food_cartons", total_pop * 0.05, "multiply"),
                "assistive_devices_per_1000_people": ("wheelchairs", total_pop / 1000, "multiply"),
            }
            for ratio_key, (res_key, multiplier, op) in ratio_checks.items():
                if ratio_key in ratios and multiplier is not None:
                    min_required = ratios[ratio_key] * multiplier
                    if op == "multiply_grams":
                        min_required = min_required / 1000
                    actual_key = self._find_key_in_dict(res_key, allocated)
                    actual = allocated.get(actual_key, 0)
                    if isinstance(actual, (int, float)) and actual < min_required:
                        return self._violation(cid, cdesc, actual_key, actual,
                                               min_required, "below_ratio_minimum")

        ratio_pcts = params.get("ratios", {})
        if "max_single_region_share_pct" in ratio_pcts:
            max_pct = ratio_pcts["max_single_region_share_pct"]
            region_plans = task_context.get("region_plans", {}) if task_context else {}
            if region_plans:
                for rid, rplan in region_plans.items():
                    r_alloc = rplan.get("allocated_resources", {})
                    for res_name, r_amount in r_alloc.items():
                        total = allocated.get(res_name, 1)
                        if total > 0 and (r_amount / total) > (max_pct / 100):
                            return self._violation(
                                cid, cdesc, f"{rid}.{res_name}",
                                f"{r_amount}/{total}={(r_amount/total)*100:.0f}%",
                                f"≤{max_pct}%", "exceeds_region_share_limit")

        return self._passed(cid, cdesc, allocated)

    def _check_capacity(self, cid: str, cdesc: str,
                         personnel: Dict[str, float],
                         allocated: Dict[str, float],
                         params: Dict[str, Any],
                         task_context: Dict = None) -> Optional[Dict]:
        limits = params.get("personnel", {})
        for pers_type, max_count in limits.items():
            actual = personnel.get(pers_type, 0)
            if isinstance(actual, (int, float)) and isinstance(max_count, (int, float)):
                if actual > max_count:
                    return self._violation(cid, cdesc, pers_type, actual,
                                           max_count, "exceeds_personnel_limit")

        cap_limits = params.get("limits", {})
        direct_limits = {
            "icu_beds": "icu_beds",
            "general_beds": "general_beds",
            "isolation_beds": "isolation_beds",
            "recovery_beds": "recovery_beds",
            "maternity_beds": "maternity_beds",
            "ambulances": "ambulances",
            "triage_stations": "triage_stations",
            "distribution_points": "distribution_points",
            "operating_theatres": "operating_theatres",
            "vaccination_teams": "vaccination_teams",
        }
        for limit_key, alloc_key in direct_limits.items():
            if limit_key in cap_limits:
                max_val = cap_limits[limit_key]
                actual = allocated.get(alloc_key, 0)
                if isinstance(actual, (int, float)) and actual > max_val:
                    return self._violation(cid, cdesc, alloc_key, actual,
                                           max_val, "exceeds_capacity_limit")

        ratios = params.get("ratios", {})
        total_pop = self._get_population(task_context)

        if total_pop > 0:
            ratio_checks = {
                "max_patients_per_doctor": ("doctors", total_pop, "max_patients"),
                "max_patients_per_nurse": ("nurses", total_pop, "max_patients"),
                "max_patients_per_doctor_per_day": ("doctors", total_pop, "max_patients"),
                "max_icu_patients_per_nurse": ("nurses", None, None),
                "max_ward_patients_per_nurse": ("nurses", None, None),
                "max_sessions_per_counselor_daily": ("counselors", total_pop, "max_sessions"),
                "max_volunteers_per_coordinator": ("volunteer_coordinators", None, None),
                "max_volunteer_shift_hours": ("volunteers", None, None),
                "max_people_per_latrine": ("portable_toilets", total_pop, "max_people_per"),
                "max_people_per_bathing_cubicle": ("portable_toilets", total_pop, "max_people_per"),
                "min_distribution_points_per_1000_people": ("distribution_points", total_pop / 1000, "min_count"),
                "guards_per_distribution_point": ("security_guards", None, None),
                "min_security_staff_per_1000_beneficiaries": ("security_guards", total_pop / 1000, "min_count"),
                "min_child_protection_officer_per_1000_children": ("child_protection_officers", total_pop * 0.3 / 1000, "min_count"),
                "max_households_per_community_health_worker": ("community_health_workers", total_pop / 4, "max_per"),
            }
            for ratio_key, (pers_key, multiplier, check_type) in ratio_checks.items():
                if ratio_key in ratios and multiplier is not None:
                    if check_type == "max_patients":
                        max_per = ratios[ratio_key]
                        min_needed = math.ceil(total_pop / max_per)
                        actual_key = self._find_key_in_dict(pers_key, personnel)
                        actual = self._as_number(personnel.get(actual_key, 0))
                        if actual < min_needed:
                            return self._violation(cid, cdesc, actual_key, actual,
                                                   min_needed, "below_min_personnel")
                    elif check_type == "max_people_per":
                        max_per = ratios[ratio_key]
                        min_needed = math.ceil(total_pop / max_per)
                        actual_key = self._find_key_in_dict(pers_key, allocated)
                        actual = self._as_number(allocated.get(actual_key, 0))
                        if actual < min_needed and actual > 0:
                            return self._violation(cid, cdesc, actual_key, actual,
                                                   min_needed, "below_min_facilities")
                    elif check_type == "min_count":
                        min_needed = math.ceil(ratios[ratio_key] * multiplier)
                        ak = self._find_key_in_dict(pers_key, allocated)
                        pk = self._find_key_in_dict(pers_key, personnel)
                        actual = self._as_number(allocated.get(ak, 0)) or self._as_number(personnel.get(pk, 0))
                        if actual < min_needed:
                            return self._violation(cid, cdesc, pers_key, actual,
                                                   min_needed, "below_min_count")

        return self._passed(cid, cdesc, personnel)

    def _check_deadline(self, cid: str, cdesc: str,
                         response_times: Dict[str, Any],
                         params: Dict[str, Any],
                         schedule: List[Dict] = None) -> Optional[Dict]:
        deadlines = params.get("deadlines", {})

        if not deadlines:
            return None

        hours_values = []
        for k, v in deadlines.items():
            if isinstance(v, (int, float)):
                if v >= 100:
                    hours_values.append(v / 24)
                else:
                    hours_values.append(v)

        if not hours_values:
            return None

        tightest = min(hours_values)

        has_response_data = any(
            isinstance(v, (int, float)) for v in response_times.values()
        )
        has_schedule_data = bool(schedule)

        if not has_response_data and not has_schedule_data:
            return None

        for region_id, rt_val in response_times.items():
            if isinstance(rt_val, (int, float)):
                if rt_val > tightest:
                    return self._violation(
                        cid, cdesc, f"{region_id}.response_time",
                        f"{rt_val}h", f"≤{tightest}h", "exceeds_deadline")

        if schedule:
            for task in schedule:
                end_time = task.get("end_time", 0)
                if isinstance(end_time, (int, float)) and end_time > tightest:
                    return self._violation(
                        cid, cdesc, f"task.{task.get('task_id', '?')}",
                        f"completed at {end_time}h", f"≤{tightest}h",
                        "schedule_exceeds_deadline")

        return self._passed(cid, cdesc, response_times)

    def _check_predicate(self, constraint: Dict[str, Any],
                          plan: Dict[str, Any],
                          task_context: Dict = None) -> Dict[str, Any]:
        cid = constraint.get("id", "unknown")
        cdesc = constraint.get("desc", "")
        spec = constraint.get("check") or (constraint.get("params") or {}).get("check")

        self.stats["predicate_checks"] += 1

        if not isinstance(spec, dict) or not spec:
            return {"status": "UNDEFINED", "record": {
                "constraint_id": cid, "desc": cdesc, "passed": False,
                "actual_summary": "this rule has no predicate yet (the knowledge base carries no check spec), verdict UNDEFINED",
                "reason": "missing_predicate",
                "required_evidence": list(constraint.get("requires") or []),
            }}

        op = spec.get("op")
        if op == "no_cooccur":
            return self._eval_no_cooccur(cid, cdesc, spec, plan)
        if op == "before":
            return self._eval_before(cid, cdesc, spec, plan, task_context)
        if op == "precedes_event":
            return self._eval_precedes_event(cid, cdesc, spec, plan, task_context)

        return {"status": "UNDEFINED", "record": {
            "constraint_id": cid, "desc": cdesc, "passed": False,
            "actual_summary": f"unknown predicate operator: {op}",
            "reason": "unknown_predicate_op",
            "required_evidence": list(constraint.get("requires") or []),
        }}

    @staticmethod
    def _tasks(plan: Dict[str, Any]) -> List[Dict]:
        return [t for t in (plan.get("schedule") or []) if isinstance(t, dict)]

    @staticmethod
    def _measure(task: Dict, name: str):
        v = task.get(name)
        return v if isinstance(v, (int, float)) else None

    @staticmethod
    def _tag_list(spec_part: Any) -> List[str]:
        if not isinstance(spec_part, dict):
            return []
        tags = spec_part.get("tags") or spec_part.get("tag") or []
        if isinstance(tags, str):
            tags = [tags]
        return [str(t).lower() for t in tags]

    @staticmethod
    def _severity_of(region_id, task_context) -> str:
        if not task_context or region_id is None:
            return ""
        for r in (task_context.get("regions") or []):
            if str(r.get("id")) == str(region_id):
                return str(r.get("severity", ""))
        return ""

    def _match_region(self, task: Dict, region_spec: Any,
                       task_context: Dict) -> bool:
        if not isinstance(region_spec, dict) or not region_spec:
            return True
        if region_spec.get("id") is not None:
            if str(task.get("region_id") or task.get("region")) != str(region_spec["id"]):
                return False
        sev = region_spec.get("severity")
        if sev:
            actual = str(task.get("region_severity")
                         or self._severity_of(task.get("region_id") or task.get("region"),
                                               task_context)).lower()
            sev = str(sev).lower()
            if sev.startswith("!="):
                return actual != sev[2:]
            return actual == sev
        return True

    def _matching_tasks(self, plan: Dict[str, Any], spec: Any,
                         task_context: Dict) -> List[Dict]:
        if not isinstance(spec, dict):
            return []
        out = []
        for t in self._tasks(plan):
            tag = spec.get("task_tag")
            if tag:
                hay = " ".join(str(t.get(k, ""))
                               for k in ("task_tag", "name", "task_id")).lower()
                if str(tag).lower() not in hay:
                    continue
            if not self._match_region(t, spec.get("region"), task_context):
                continue
            out.append(t)
        return out

    def _eval_no_cooccur(self, cid: str, cdesc: str, spec: Dict,
                          plan: Dict[str, Any]) -> Dict[str, Any]:
        carriers = plan.get("carriers")
        if not carriers:
            return {"status": "UNDEFINED", "record": {
                "constraint_id": cid, "desc": cdesc, "passed": False,
                "actual_summary": "the plan carries no carriers evidence field (transport / storage grouping), so mutual exclusion cannot be decided",
                "reason": "missing_evidence",
                "required_evidence": ["carriers[].carrier_id", "carriers[].items"],
            }}

        a_tags = self._tag_list(spec.get("a"))
        b_tags = self._tag_list(spec.get("b"))
        seen_a = seen_b = False

        for carrier in carriers:
            if not isinstance(carrier, dict):
                continue
            items = " ".join(str(x).lower() for x in (carrier.get("items") or []))
            has_a = any(t in items for t in a_tags)
            has_b = any(t in items for t in b_tags)
            seen_a = seen_a or has_a
            seen_b = seen_b or has_b
            if has_a and has_b:
                return {"status": "FAIL", "record": self._violation(
                    cid, cdesc, f"carrier.{carrier.get('carrier_id', '?')}",
                    f"co-loaded: {a_tags} and {b_tags}", "the two entity classes must not be co-loaded",
                    "exclusivity_violation")}

        if not seen_a and not seen_b:
            return {"status": "NOT_ADDRESSED", "record": {
                "constraint_id": cid, "desc": cdesc,
                "reason": "the plan involves neither entity of this mutual-exclusion rule"}}

        return {"status": "PASS", "record": self._passed(
            cid, cdesc, {"no_cooccur": f"{a_tags} and {b_tags} are not co-loaded"})}

    def _eval_before(self, cid: str, cdesc: str, spec: Dict,
                      plan: Dict[str, Any], task_context: Dict) -> Dict[str, Any]:
        measure = spec.get("measure", "end_time")
        subj = self._matching_tasks(plan, spec.get("subject"), task_context)
        obj = self._matching_tasks(plan, spec.get("object"), task_context)

        if not subj and not obj:
            return {"status": "NOT_ADDRESSED", "record": {
                "constraint_id": cid, "desc": cdesc,
                "reason": "the plan involves neither entity of this comparison rule"}}

        if not subj or not obj:
            return {"status": "UNDEFINED", "record": {
                "constraint_id": cid, "desc": cdesc, "passed": False,
                "actual_summary": f"only one side of the entity pair appears in the plan"
                                  f" (subject {len(subj)} / object {len(obj)}), the order cannot be decided",
                "reason": "insufficient_evidence",
                "required_evidence": ["schedule[].task_tag",
                                      "schedule[].region_id", f"schedule[].{measure}"],
            }}

        s_vals = [v for v in (self._measure(t, measure) for t in subj) if v is not None]
        o_vals = [v for v in (self._measure(t, measure) for t in obj) if v is not None]
        if not s_vals or not o_vals:
            return {"status": "UNDEFINED", "record": {
                "constraint_id": cid, "desc": cdesc, "passed": False,
                "actual_summary": f"the schedule entry lacks the {measure} field, the order cannot be decided",
                "reason": "insufficient_evidence",
                "required_evidence": [f"schedule[].{measure}"],
            }}

        s_first, o_first = min(s_vals), min(o_vals)
        if s_first > o_first:
            return {"status": "FAIL", "record": self._violation(
                cid, cdesc, measure, s_first, f"≤{o_first}",
                "priority_violation")}
        return {"status": "PASS", "record": self._passed(
            cid, cdesc, {measure: f"subject {s_first} <= object {o_first}"})}

    def _eval_precedes_event(self, cid: str, cdesc: str, spec: Dict,
                              plan: Dict[str, Any], task_context: Dict
                              ) -> Dict[str, Any]:
        events = plan.get("events")
        if not isinstance(events, dict):
            return {"status": "UNDEFINED", "record": {
                "constraint_id": cid, "desc": cdesc, "passed": False,
                "actual_summary": "the plan carries no events evidence field (precondition event state), so the verdict cannot be made",
                "reason": "missing_evidence",
                "required_evidence": [f"events.{spec.get('event')}.done"],
            }}

        event_name = spec.get("event")
        ev = events.get(event_name)
        if ev is None:
            return {"status": "UNDEFINED", "record": {
                "constraint_id": cid, "desc": cdesc, "passed": False,
                "actual_summary": f"the plan does not report the state of precondition event {event_name}",
                "reason": "missing_evidence",
                "required_evidence": [f"events.{event_name}.done"],
            }}

        if isinstance(ev, dict):
            done = bool(ev.get("done"))
            ev_time = ev.get("time")
        else:
            done, ev_time = bool(ev), None

        measure = spec.get("measure", "start_time")
        actions = self._matching_tasks(plan, spec.get("action"), task_context)

        if not actions:
            return {"status": "PASS", "record": self._passed(
                cid, cdesc, {"events": f"{event_name}: the plan never started the constrained action"})}

        if not done:
            return {"status": "FAIL", "record": self._violation(
                cid, cdesc, f"events.{event_name}", "not completed",
                "must be completed before the action starts", "precondition_violation")}

        a_vals = [v for v in (self._measure(t, measure) for t in actions)
                  if v is not None]
        if isinstance(ev_time, (int, float)) and a_vals and ev_time > min(a_vals):
            return {"status": "FAIL", "record": self._violation(
                cid, cdesc, f"events.{event_name}.time", f"{ev_time}",
                f"≤{min(a_vals)}", "precondition_violation")}

        return {"status": "PASS", "record": self._passed(
            cid, cdesc, {"events": f"{event_name}: done={done}"})}

    def _passed(self, cid: str, cdesc: str, actual_data: Any) -> Dict:
        return {
            "constraint_id": cid,
            "desc": cdesc,
            "passed": True,
            "actual_summary": self._summarize(actual_data),
        }

    def _violation(self, cid: str, cdesc: str, field: str,
                    actual: Any, limit: Any, reason: str) -> Dict:
        deviation = None
        deviation_pct = None
        if isinstance(actual, (int, float)) and isinstance(limit, (int, float)) and limit != 0:
            deviation = actual - limit
            deviation_pct = round(deviation / abs(limit) * 100, 1)

        return {
            "constraint_id": cid,
            "desc": cdesc,
            "passed": False,
            "field": field,
            "actual_value": actual,
            "limit": limit,
            "deviation": deviation,
            "deviation_pct": deviation_pct,
            "reason": reason,
        }

    def _is_constraint_addressed(self, c: Dict, allocated: Dict,
                                   personnel: Dict) -> bool:
        params = c.get("params", {})

        constraint_keys = set()

        for section in ["resources", "limits", "ratios", "personnel"]:
            section_data = params.get(section, {})
            if isinstance(section_data, dict):
                constraint_keys.update(section_data.keys())

        if not constraint_keys:
            return True

        llm_keys = set(allocated.keys()) | set(personnel.keys())

        for ck in constraint_keys:
            if ck in llm_keys:
                return True
            for lk in llm_keys:
                if ck.startswith(lk) or lk.startswith(ck):
                    return True
                ck_words = set(ck.split("_"))
                lk_words = set(lk.split("_"))
                if len(ck_words & lk_words) >= 2:
                    return True

        return False

    def _find_key_in_dict(self, target_key: str, d: Dict) -> str:
        if target_key in d:
            return target_key
        target_words = set(target_key.split("_"))
        for k in d:
            k_words = set(k.split("_"))
            if target_words & k_words:
                return k
        return target_key

    def _as_number(self, value: Any) -> float:
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            return float(value.get("count", value.get("total", value.get("amount", 0))))
        return 0.0

    def _summarize(self, data: Any, max_len: int = 80) -> str:
        if isinstance(data, dict):
            items = [f"{k}={v}" for k, v in list(data.items())[:5]]
            s = ", ".join(items)
            return s[:max_len] + ("..." if len(s) > max_len else "")
        return str(data)[:max_len]

    def _get_population(self, task_context: Dict = None) -> int:
        if not task_context:
            return 0
        pop = task_context.get("total_population", 0)
        if pop > 0:
            return pop
        regions = task_context.get("regions", [])
        return sum(r.get("population", 0) for r in regions)

    def format_report(self, audit_result: Dict[str, Any],
                       verbose: bool = True) -> str:
        total = audit_result.get("total", 0)
        passed_count = len(audit_result.get("passed", []))
        violated_count = len(audit_result.get("violated", []))
        not_addressed_count = len(audit_result.get("not_addressed", []))
        undefined_count = len(audit_result.get("undefined", []))
        rate = audit_result.get("compliance_rate", 0)

        lines = [
            "constraint audit report",
            f"{'='*40}",
            f"total: {total} rules | passed: {passed_count} | violated: {violated_count} | not addressed: {not_addressed_count} | missing evidence: {undefined_count}",
            f"compliance rate: {rate:.1%} (over {passed_count} / {passed_count + violated_count + undefined_count} decidable rules)",
            "",
        ]

        if audit_result.get("undefined"):
            lines.append(f"--- missing evidence (counted as unsatisfied) ({undefined_count} rules) ---")
            for u in audit_result["undefined"]:
                lines.append(f"  [{u.get('constraint_id', '?')}] {u.get('desc', '')}")
                lines.append(f"    {u.get('actual_summary', '')}")
            lines.append("")

        if audit_result.get("violated"):
            lines.append(f"--- violation detail ({violated_count} rules) ---")
            for v in audit_result["violated"]:
                lines.append(
                    f"  [{v['constraint_id']}] {v['desc']}")
                lines.append(
                    f"    field: {v.get('field', '?')} | "
                    f"actual: {v.get('actual_value', '?')} | "
                    f"limit: {v.get('limit', '?')} | "
                    f"deviation: {v.get('deviation_pct', '?')}%")
                lines.append(f"    reason: {v.get('reason', '?')}")
                lines.append("")

        if audit_result.get("critical_violations"):
            lines.append(f"--- critical violations ({len(audit_result['critical_violations'])} rules) ---")
            for v in audit_result["critical_violations"]:
                lines.append(f"  ! [{v['constraint_id']}] {v['desc']}")

        return "\n".join(lines)


def create_auditor(constraints: List[Dict[str, Any]] = None,
                   llm_engine=None) -> ConstraintAuditor:
    return ConstraintAuditor(constraints=constraints, llm_engine=llm_engine)
