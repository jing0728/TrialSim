"""
phase2_atomize.py

Criterion Atomization — Phase II of the TrialSim-10k pipeline.

Transforms structured PICO parse results into Atomic Logic Triplets of the
form <variable, operator, threshold>, then generates a violation scenario
(the "Negative Hard" pivot point) for each unit.

Two-tier violation generation:
    Tier 1 (rule-based)  : fast, zero API cost, covers well-formed numerics.
    Tier 2 (LLM V4)      : Clinical Trial Architect persona; produces a
                           clinically-grounded violation value, rationale,
                           and unit label for criteria that rule-based logic
                           cannot handle precisely.

Pipeline position:
    parsed_pico.jsonl  ->  [this file]  ->  atomic_units.jsonl
"""

import json
import time
import os
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = "data/raw/parsed_pico.jsonl"
OUTPUT_PATH = "data/raw/atomic_units.jsonl"

client = Anthropic()
MODEL  = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# V4 Logical Architect prompt
# ---------------------------------------------------------------------------
# Design rationale (mirrors paper Table IV evolution):
#   V1  Narrative      — free-form text, not parseable
#   V2  Structural     — structured but numerically unaware ("High BP" not "140 mmHg")
#   V3  Quant. Modeler — found thresholds but skipped violation scenarios
#   V4  Logical Architect (this) — enforces atomic decomposition, identifies
#       the exact Pivot Point, and computes a clinically marginal violation.
#
# Key prompt constraints:
#   • "Clinical Trial Architect" persona  → regulatory mindset, not conversational
#   • Explicit epsilon rules per data type → prevents epsilon=0 or epsilon=100
#   • "N/A" fallback requirement           → eliminates hallucinated values
#   • Strict JSON-only output              → safe to parse without repair logic
LOGICAL_ARCHITECT_PROMPT = """\
You are a Clinical Trial Architect specializing in regulatory compliance.
Your task is to identify the exact Pivot Point of a single eligibility \
criterion and design the narrowest possible violation scenario.

Criterion text : {criterion_text}
Extracted triplet: <{variable}, {operator}, {threshold}>

Return ONLY valid JSON — no markdown, no explanation, no preamble:
{{
  "pivot_point":          "{threshold}",
  "violation_value":      "<value that is just barely outside the boundary>",
  "violation_rationale":  "<one sentence: why this value is clinically marginal>",
  "clinical_unit":        "<unit from the source text, e.g. years / mg/dL / % — or N/A>"
}}

Epsilon rules (use the SMALLEST clinically meaningful increment):
  - Age                 : epsilon = 1 year
  - Continuous lab values (HbA1c, eGFR, BP …) : epsilon = 0.1
  - Integer scores (ACT, NIHSS …)             : epsilon = 1
  - Percentages                                : epsilon = 0.1 %
  - Never invent units that do not appear in the criterion text.
  - If the threshold cannot be determined, set violation_value to "N/A".\
"""

# ---------------------------------------------------------------------------
# Tier 1: rule-based violation generator
# ---------------------------------------------------------------------------

def generate_violation_rule(operator: str, threshold: str) -> str:
    """
    Fast, zero-cost violation value derived from operator + threshold alone.

    Covers:
        ">=" / "<="   : step one epsilon outside the bound
        ">"  / "<"    : the bound itself is the violation (exact equality fails)
        "between X-Y" : both boundary violations returned as "X-eps or Y+eps"

    Epsilon is 1 for integers, 0.1 for decimals — a deliberate simplification
    that suffices for age and most lab values at the cost of some clinical
    granularity (addressed by Tier 2).

    Returns "N/A" when the threshold cannot be parsed as a number.
    """
    try:
        if operator.lower() == "between" and "-" in str(threshold):
            lo, hi   = threshold.split("-")
            lo_f, hi_f = float(lo.strip()), float(hi.strip())
            eps_lo   = 1 if lo_f == int(lo_f) else 0.1
            eps_hi   = 1 if hi_f == int(hi_f) else 0.1
            return f"{lo_f - eps_lo} or {hi_f + eps_hi}"

        val     = float(threshold)
        epsilon = 1 if val == int(val) else 0.1

        return {
            ">=" : str(val - epsilon),   # just below lower bound → ineligible
            "≥"  : str(val - epsilon),
            "greater than or equal": str(val - epsilon),
            "<=" : str(val + epsilon),   # just above upper bound → ineligible
            "≤"  : str(val + epsilon),
            "less than or equal": str(val + epsilon),
            ">"  : str(val),             # exactly equal → fails strict ">"
            "greater than": str(val),
            "<"  : str(val),             # exactly equal → fails strict "<"
            "less than": str(val),
        }.get(operator, "N/A")

    except (ValueError, TypeError):
        return "N/A"

# ---------------------------------------------------------------------------
# Tier 2: LLM V4 Logical Architect enhancement
# ---------------------------------------------------------------------------

def enhance_with_llm(unit: dict, retries: int = 3) -> dict:
    """
    Call the Logical Architect (V4) to produce a clinically grounded
    violation value, a one-sentence rationale, and the correct clinical unit.

    The LLM output is merged on top of the existing unit dict so that all
    rule-based fields are preserved; only violation metadata is upgraded.

    Falls back gracefully: if the API call fails or returns unparseable JSON
    after all retries, the original unit is returned unchanged so the pipeline
    never loses a record.

    Rate-limit strategy: exponential back-off (1 s → 2 s → 4 s).
    """
    prompt = LOGICAL_ARCHITECT_PROMPT.format(
        criterion_text = unit.get("criterion_text", ""),
        variable       = unit.get("variable",  "N/A"),
        operator       = unit.get("operator",  "N/A"),
        threshold      = unit.get("threshold", "N/A"),
    )

    for attempt in range(retries):
        try:
            response = client.messages.create(
                model     = MODEL,
                max_tokens= 300,
                messages  = [{"role": "user", "content": prompt}],
            )
            raw = response.content[0].text.strip()

            # Strip accidental markdown fences (defensive — prompt forbids them)
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
                raw = raw.strip()

            enhancement = json.loads(raw)

            return {
                **unit,
                # Override rule-based violation with the LLM-computed value
                "violation_scenario":   enhancement.get("violation_value",     unit["violation_scenario"]),
                "violation_rationale":  enhancement.get("violation_rationale", "N/A"),
                "clinical_unit":        enhancement.get("clinical_unit",        "N/A"),
                "violation_source":     "llm_v4",
            }

        except json.JSONDecodeError:
            # LLM returned non-JSON; retry without modifying the unit
            pass
        except Exception as exc:
            if attempt == retries - 1:
                print(f"    [LLM enhance failed for '{unit.get('variable')}': {exc}]")
                break
            time.sleep(2 ** attempt)

    # All retries exhausted — return original unit with source tag
    return {**unit, "violation_source": "rule_v1_fallback"}

# ---------------------------------------------------------------------------
# Atomic unit extractor
# ---------------------------------------------------------------------------

def extract_atomic_units(parsed: dict, use_llm: bool = False) -> dict:
    """
    Convert one PICO-parsed trial record into a list of Atomic Logic Triplets.

    Each triplet:  ai = <variable, operator, threshold>

    Filtering rule: units with variable="N/A" or threshold="N/A" are dropped
    because they cannot be used to generate verifiable boundary cases.

    use_llm=True activates Tier 2 enhancement for every unit that has a
    non-"N/A" violation_scenario from Tier 1.  Set to False for fast/cheap
    bulk runs; enable selectively for high-value trials.
    """
    units: list[dict] = []

    for crit_type in ("inclusion_criteria", "exclusion_criteria"):
        label = crit_type.split("_")[0]   # "inclusion" | "exclusion"
        for crit in parsed.get(crit_type, []):
            v   = crit.get("variable",  "N/A")
            op  = crit.get("operator",  "N/A")
            tau = crit.get("threshold", "N/A")

            # Skip under-specified criteria — unusable for boundary generation
            if v == "N/A" or tau == "N/A":
                continue

            unit: dict = {
                "variable":         v,
                "operator":         op,
                "threshold":        tau,
                "type":             label,
                "criterion_text":   crit.get("criterion", ""),
                # Tier 1 baseline — always computed
                "violation_scenario": generate_violation_rule(op, tau),
                "violation_source": "rule_v1",
            }

            # Tier 2 upgrade — LLM adds clinical rationale and precise unit
            if use_llm and unit["violation_scenario"] != "N/A":
                unit = enhance_with_llm(unit)

            units.append(unit)

    return {
        "nct_id":       parsed.get("nct_id"),
        "condition":    parsed.get("condition"),
        "demographics": parsed.get("demographics"),
        "atomic_units": units,
    }

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_atomize(use_llm: bool = False, llm_limit: int = 50) -> list[dict]:
    """
    Process all records in parsed_pico.jsonl and write atomic_units.jsonl.

    Parameters
    ----------
    use_llm   : activate Tier 2 LLM enhancement (slower, costs API calls)
    llm_limit : max number of trials to send through LLM enhancement;
                ignored when use_llm=False.  Useful for partial upgrades
                on high-value disease categories (e.g. Oncology only).

    Quality signals printed at the end:
        - extraction rate  (how many units were kept vs dropped)
        - violation coverage (how many got a non-"N/A" violation value)
        - LLM upgrade rate   (only when use_llm=True)
    """
    results:    list[dict] = []
    llm_count:  int        = 0

    with open(INPUT_PATH, encoding="utf-8") as f, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out:

        for line in f:
            parsed = json.loads(line.strip())

            # Apply LLM only up to llm_limit to control cost during testing
            run_llm = use_llm and llm_count < llm_limit
            atomized = extract_atomic_units(parsed, use_llm=run_llm)

            if run_llm:
                llm_count += 1

            out.write(json.dumps(atomized, ensure_ascii=False) + "\n")
            results.append(atomized)

    # --- Quality report ---
    total_units      = sum(len(r["atomic_units"]) for r in results)
    covered          = sum(
        1 for r in results
        for u in r["atomic_units"]
        if u["violation_scenario"] != "N/A"
    )
    llm_upgraded     = sum(
        1 for r in results
        for u in r["atomic_units"]
        if u.get("violation_source") == "llm_v4"
    )

    print(f"Atomization complete : {len(results)} trials processed")
    print(f"Total atomic units   : {total_units}")
    print(f"Violation coverage   : {covered}/{total_units} "
          f"({covered / total_units * 100:.1f}%)" if total_units else "")
    if use_llm:
        print(f"LLM V4 upgrades      : {llm_upgraded}/{total_units}")
    print(f"Output saved to      : {OUTPUT_PATH}")

    # --- Sample output ---
    if results:
        print("\n=== Sample Output ===")
        ex = results[0]
        print(f"NCT ID       : {ex['nct_id']}")
        print(f"Condition    : {ex['condition']}")
        print(f"Demographics : {ex['demographics']}")
        print(f"Atomic units : {len(ex['atomic_units'])}")
        print()
        for u in ex["atomic_units"][:5]:
            print(f"  Triplet    : <{u['variable']}, {u['operator']}, {u['threshold']}>")
            print(f"  Type       : {u['type']}")
            print(f"  Violation  : {u['violation_scenario']}")
            if u.get("violation_rationale"):
                print(f"  Rationale  : {u['violation_rationale']}")
            if u.get("clinical_unit"):
                print(f"  Unit       : {u['clinical_unit']}")
            print(f"  Source     : {u.get('violation_source', 'rule_v1')}")
            print()

    return results


if __name__ == "__main__":
    # Quick run (rule-based only, free):
    #   run_atomize()
    #
    # LLM-enhanced run on first 50 trials:
    #   run_atomize(use_llm=True, llm_limit=50)
    run_atomize(use_llm=False)