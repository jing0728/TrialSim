"""
phase2_atomize.py

Criterion Atomization — Phase II of the TrialSim-10k pipeline.

Final low-cost upgraded version:
- Rule-based atomization remains the default backbone
- GPT-4o-mini is used only for:
  1) rescuing criteria that would otherwise be dropped
  2) optionally improving violation metadata for extracted units

Pipeline position:
    parsed_pico.jsonl  ->  [this file]  ->  atomic_units.jsonl
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

INPUT_PATH = "data/raw/parsed_pico.jsonl"
OUTPUT_PATH = "data/raw/atomic_units.jsonl"

# ---------------------------------------------------------------------------
# OpenAI client
# ---------------------------------------------------------------------------

MODEL = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=API_KEY) if (OpenAI is not None and API_KEY) else None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

LOGICAL_ARCHITECT_PROMPT = """\
You are a Clinical Trial Architect specializing in regulatory compliance.
Your task is to identify the exact Pivot Point of a single eligibility criterion
and design the narrowest possible violation scenario.

Criterion text : {criterion_text}
Extracted triplet: <{variable}, {operator}, {threshold}>

Return ONLY valid JSON — no markdown, no explanation, no preamble:
{{
  "pivot_point": "{threshold}",
  "violation_value": "<value that is just barely outside the boundary>",
  "violation_rationale": "<one sentence: why this value is clinically marginal>",
  "clinical_unit": "<unit from the source text, e.g. years / mg/dL / % — or N/A>"
}}

Epsilon rules (use the SMALLEST clinically meaningful increment):
- Age: epsilon = 1 year
- Continuous lab values (HbA1c, eGFR, BP, creatinine, bilirubin): epsilon = 0.1
- Integer scores (ACT, NIHSS, ECOG): epsilon = 1
- Percentages: epsilon = 0.1
- Never invent units that do not appear in the criterion text.
- If the threshold cannot be determined, set violation_value to "N/A".
"""

TRIPLET_RESCUE_PROMPT = """\
You extract a single atomic eligibility triplet from one clinical-trial criterion.

Return ONLY valid JSON:
{{
  "variable": "...",
  "operator": "...",
  "threshold": "..."
}}

Rules:
- Allowed operators: >=, <=, >, <, between, =, N/A
- If the criterion is not numerically or categorically verifiable, return "N/A" for missing fields.
- "between" thresholds must be formatted as "X-Y"
- For exact status checks like "HIV negative", operator may be "=" and threshold may be "Negative"
- Do not add explanation.

Criterion:
{criterion_text}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_block(raw: str) -> str | None:
    """
    Extract the first plausible JSON object block from model output.
    Handles cases like:
      Sure! Here's the JSON:
      {...}
    """
    if not raw:
        return None

    raw = raw.strip()

    # Fast path: whole response is already JSON
    if raw.startswith("{") and raw.endswith("}"):
        return raw

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)

    return None


def _call_openai_json(prompt: str, max_output_tokens: int = 220, retries: int = 3) -> dict[str, Any] | None:
    """
    Call OpenAI Responses API and parse JSON-only output.
    Returns parsed dict on success, None on repeated failure.
    """
    if client is None:
        return None

    for attempt in range(retries):
        try:
            response = client.responses.create(
                model=MODEL,
                input=prompt,
                max_output_tokens=max_output_tokens,
            )

            raw = response.output_text.strip()
            json_text = _extract_json_block(raw)
            if not json_text:
                raise ValueError("No JSON found in model output")

            return json.loads(json_text)

        except Exception as exc:
            print(f"    [LLM retry {attempt + 1}/{retries}] {type(exc).__name__}")
            if attempt == retries - 1:
                return None
            time.sleep(1.5)

    return None


def _is_numeric_threshold(threshold: Any) -> bool:
    """
    True for:
      "18"
      "9.0"
      "18-65"
    False for:
      "Negative"
      "N/A"
      ""
    """
    if threshold in (None, "", "N/A"):
        return False

    text = str(threshold).strip()

    if "-" in text:
        parts = text.split("-")
        if len(parts) != 2:
            return False
        try:
            float(parts[0].strip())
            float(parts[1].strip())
            return True
        except ValueError:
            return False

    try:
        float(text)
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Tier 1: rule-based violation generator
# ---------------------------------------------------------------------------

def generate_violation_rule(operator: str, threshold: Any) -> str:
    """
    Cheap rule-based violation generation.

    Supports:
      >=, <=, >, <, between
    Returns "N/A" for non-numeric thresholds and unsupported operators.
    """
    try:
        op = str(operator).strip().lower()
        tau = str(threshold).strip()

        if op == "between" and "-" in tau:
            lo_s, hi_s = tau.split("-", 1)
            lo = float(lo_s.strip())
            hi = float(hi_s.strip())
            eps_lo = 1.0 if lo == int(lo) else 0.1
            eps_hi = 1.0 if hi == int(hi) else 0.1
            return f"{lo - eps_lo} or {hi + eps_hi}"

        val = float(tau)
        epsilon = 1.0 if val == int(val) else 0.1

        mapping = {
            ">=": str(val - epsilon),
            "≥": str(val - epsilon),
            "<=": str(val + epsilon),
            "≤": str(val + epsilon),
            ">": str(val),
            "<": str(val),
        }
        return mapping.get(operator, mapping.get(op, "N/A"))

    except Exception:
        return "N/A"


# ---------------------------------------------------------------------------
# Tier 2a: rescue dropped criteria
# ---------------------------------------------------------------------------

def rescue_triplet_with_llm(criterion_text: str, retries: int = 2) -> dict[str, str] | None:
    """
    Try to rescue a triplet for criteria that rule-based parsing left as N/A.
    Returns dict with variable/operator/threshold, or None if rescue failed.
    """
    data = _call_openai_json(
        TRIPLET_RESCUE_PROMPT.format(criterion_text=criterion_text),
        max_output_tokens=120,
        retries=retries,
    )
    if not data:
        return None

    variable = str(data.get("variable", "N/A")).strip() or "N/A"
    operator = str(data.get("operator", "N/A")).strip() or "N/A"
    threshold = str(data.get("threshold", "N/A")).strip() or "N/A"

    if variable == "N/A" or threshold == "N/A":
        return None

    return {
        "variable": variable,
        "operator": operator,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Tier 2b: improve violation metadata
# ---------------------------------------------------------------------------

def enhance_with_llm(unit: dict[str, Any], retries: int = 2) -> dict[str, Any]:
    """
    Improve violation metadata for already-extracted units.
    Falls back gracefully to the original unit.
    """
    if client is None or not _is_numeric_threshold(unit.get("threshold", "N/A")):
        return unit

    prompt = LOGICAL_ARCHITECT_PROMPT.format(
        criterion_text=unit.get("criterion_text", ""),
        variable=unit.get("variable", "N/A"),
        operator=unit.get("operator", "N/A"),
        threshold=unit.get("threshold", "N/A"),
    )

    data = _call_openai_json(prompt, max_output_tokens=220, retries=retries)
    if not data:
        return unit

    return {
        **unit,
        "pivot_point": data.get("pivot_point", unit.get("threshold", "N/A")),
        "violation_scenario": data.get("violation_value", unit.get("violation_scenario", "N/A")),
        "violation_rationale": data.get("violation_rationale", "N/A"),
        "clinical_unit": data.get("clinical_unit", "N/A"),
        "violation_source": "llm_v4",
    }


# ---------------------------------------------------------------------------
# Main atomizer
# ---------------------------------------------------------------------------

def extract_atomic_units(
    parsed: dict[str, Any],
    use_llm: bool = False,
    rescue_budget: list[int] | None = None,
    enhance_budget: list[int] | None = None,
) -> dict[str, Any]:
    """
    Convert parsed criteria into atomic units.

    Strategy:
    1) Use existing rule-based fields from phase1_parse.py
    2) If variable/threshold is missing, optionally rescue with GPT-4o-mini
    3) Generate rule-based violation
    4) Optionally upgrade violation metadata with GPT-4o-mini
    """
    units: list[dict[str, Any]] = []

    for crit_type in ("inclusion_criteria", "exclusion_criteria"):
        label = crit_type.split("_")[0]

        for crit in parsed.get(crit_type, []):
            recovered = None

            criterion_text = crit.get("criterion", "")
            variable = crit.get("variable", "N/A")
            operator = crit.get("operator", "N/A")
            threshold = crit.get("threshold", "N/A")

            rescued = False

            # ---------------------------------------------------------------
            # Rescue otherwise-dropped criteria
            # ---------------------------------------------------------------
            if (
                use_llm
                and client is not None
                and (variable == "N/A" or threshold == "N/A")
                and rescue_budget is not None
                and rescue_budget[0] > 0
            ):
                recovered = rescue_triplet_with_llm(criterion_text)

                if recovered:
                    rescue_budget[0] -= 1
                    variable = recovered["variable"]
                    operator = recovered["operator"]
                    threshold = recovered["threshold"]
                    rescued = True

            # Still unusable -> skip
            if variable == "N/A" or threshold == "N/A":
                continue

            unit = {
                "variable": variable,
                "operator": operator,
                "threshold": threshold,
                "type": label,
                "criterion_text": criterion_text,
                "violation_scenario": generate_violation_rule(operator, threshold),
                "violation_source": "rule_v1_rescued" if rescued else "rule_v1",
            }

            # Optional enhancement only for numeric-threshold units
            if (
                use_llm
                and client is not None
                and _is_numeric_threshold(threshold)
                and enhance_budget is not None
                and enhance_budget[0] > 0
            ):
                upgraded = enhance_with_llm(unit)

                if upgraded.get("violation_source") == "llm_v4":
                    enhance_budget[0] -= 1

                unit = upgraded

            units.append(unit)

    return {
        "nct_id": parsed.get("nct_id"),
        "condition": parsed.get("condition"),
        "demographics": parsed.get("demographics"),
        "atomic_units": units,
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_atomize(
    use_llm: bool = False,
    llm_rescue_limit: int = 10,
    llm_enhance_limit: int = 5,
    max_trials: int | None = 20,
) -> list[dict[str, Any]]:
    """
    Run atomization over parsed_pico.jsonl.

    Safe low-cost defaults:
    - rescue first 10 dropped criteria
    - enhance first 5 numeric units
    - process first 20 trials only

    Once confirmed working, scale these values upward.
    """
    results: list[dict[str, Any]] = []
    rescue_budget = [llm_rescue_limit]
    enhance_budget = [llm_enhance_limit]

    if use_llm and client is None:
        print("[WARN] OPENAI_API_KEY not set or OpenAI package unavailable.")
        print("[WARN] Falling back to rule-only mode.")

    with open(INPUT_PATH, encoding="utf-8") as f, open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for idx, line in enumerate(f, start=1):
            if max_trials is not None and idx > max_trials:
                break

            line = line.strip()
            if not line:
                continue

            parsed = json.loads(line)
            nct_id = parsed.get("nct_id", f"#{idx}")
            print(f"[{idx}] Processing {nct_id}...")

            atomized = extract_atomic_units(
                parsed,
                use_llm=use_llm,
                rescue_budget=rescue_budget,
                enhance_budget=enhance_budget,
            )

            print(
                f"    units={len(atomized.get('atomic_units', []))} | "
                f"rescue_left={rescue_budget[0]} | "
                f"enhance_left={enhance_budget[0]}"
            )

            out.write(json.dumps(atomized, ensure_ascii=False) + "\n")
            results.append(atomized)

    total_units = sum(len(r["atomic_units"]) for r in results)
    covered = sum(
        1
        for r in results
        for u in r["atomic_units"]
        if u.get("violation_scenario", "N/A") != "N/A"
    )
    llm_v4 = sum(
        1
        for r in results
        for u in r["atomic_units"]
        if u.get("violation_source") == "llm_v4"
    )
    rescued = sum(
        1
        for r in results
        for u in r["atomic_units"]
        if u.get("violation_source") == "rule_v1_rescued"
    )

    print(f"\nAtomization complete : {len(results)} trials processed")
    print(f"Total atomic units   : {total_units}")
    if total_units:
        print(f"Violation coverage   : {covered}/{total_units} ({covered / total_units * 100:.1f}%)")
    else:
        print("Violation coverage   : 0/0 (0.0%)")

    if use_llm:
        print(f"LLM rescued units    : {rescued}")
        print(f"LLM enhanced units   : {llm_v4}")
        print(f"Rescue budget left   : {rescue_budget[0]}")
        print(f"Enhance budget left  : {enhance_budget[0]}")

    print(f"Output saved to      : {OUTPUT_PATH}")

    if results:
        sample = next((r for r in results if r["atomic_units"]), results[0])
        print("\n=== Sample Output ===")
        print(f"NCT ID       : {sample.get('nct_id')}")
        print(f"Condition    : {sample.get('condition')}")
        print(f"Demographics : {sample.get('demographics')}")
        print(f"Atomic units : {len(sample.get('atomic_units', []))}")

        for unit in sample.get("atomic_units", [])[:5]:
            print()
            print(f"  Triplet    : <{unit.get('variable')}, {unit.get('operator')}, {unit.get('threshold')}>")
            print(f"  Type       : {unit.get('type')}")
            print(f"  Violation  : {unit.get('violation_scenario')}")
            print(f"  Source     : {unit.get('violation_source')}")

    return results


if __name__ == "__main__":
    run_atomize(
        use_llm=True,
        llm_rescue_limit=50,
        llm_enhance_limit=20,
        max_trials=100,
    )