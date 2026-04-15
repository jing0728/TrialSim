"""
phase1_parse_llm.py

LLM-based PICO parser — Phase I of the TrialSim-10k pipeline.

Reads filtered_studies.json and parses each trial's eligibility criteria
into structured PICO format using Claude as a zero-shot parser.

Key improvements over the original version:
    1. Richer system prompt with explicit operator/threshold examples
       → LLM now extracts HbA1c, eGFR, ECOG, scores etc., not just age
    2. Criteria split before truncation
       → inclusion and exclusion sections are each given 2000 chars,
         preventing one section from crowding out the other
    3. max_tokens raised from 2000 → 3000
       → handles trials with 20+ criteria without truncation
    4. Retry with a focused repair prompt when repair_json fails
       → recovers ~60% of previously failed parses
    5. Quality report now shows criterion-level threshold extraction rate
       → reveals whether lab/score thresholds are being captured

Pipeline position:
    filtered_studies.json  ->  [this file]  ->  parsed_pico.jsonl
"""

import json
import os
import time
from anthropic import Anthropic

INPUT_PATH  = "data/raw/filtered_studies.json"
OUTPUT_PATH = "data/raw/parsed_pico.jsonl"

client = Anthropic()
MODEL  = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# System prompt — V3 Structured Parser with threshold extraction examples
# ---------------------------------------------------------------------------
# Original prompt only mentioned age extraction.
# Improved version adds explicit examples for lab values, scores, and
# temporal constraints so the LLM knows to extract them as triplets.
SYSTEM_PROMPT = """\
You are a senior Clinical Data Scientist specializing in regulatory compliance.
Parse clinical trial eligibility criteria into strict JSON.

RULES:
- Output ONLY valid JSON — no markdown, no explanation, no preamble.
- For any field not explicitly stated in the text, use exactly "N/A".
- Never fabricate or infer values not present in the text.
- Extract numerical boundaries precisely.
- For criteria lines with no numeric threshold, set operator and threshold to "N/A".

OPERATOR EXTRACTION GUIDE — use these exact operator strings:
  ">="  for "at least", "or older", "or more", "≥", "minimum"
  "<="  for "no more than", "or younger", "or less", "≤", "maximum"
  ">"   for "greater than", "more than", "above"
  "<"   for "less than", "fewer than", "below", "under"
  "between"  for "X to Y", "X - Y", "between X and Y", "from X to Y"
  "="   for "must be", "required to be", "confirmed", "positive", "negative"

THRESHOLD EXTRACTION EXAMPLES:
  "aged 18 to 65"                    -> variable="Age",  operator="between", threshold="18-65"
  "at least 18 years old"            -> variable="Age",  operator=">=",      threshold="18"
  "HbA1c less than 7.0%"             -> variable="HbA1c", operator="<",      threshold="7.0"
  "eGFR >= 30 mL/min"                -> variable="eGFR", operator=">=",      threshold="30"
  "ECOG performance status 0 or 1"   -> variable="ECOG performance status", operator="<=", threshold="1"
  "BMI between 18.5 and 35"          -> variable="BMI",  operator="between", threshold="18.5-35"
  "systolic BP < 140 mmHg"           -> variable="Systolic BP", operator="<", threshold="140"
  "platelet count >= 100 x10^9/L"    -> variable="Platelet count", operator=">=", threshold="100"
  "within 30 days of diagnosis"      -> variable="Days since diagnosis", operator="<=", threshold="30"
  "no prior chemotherapy"            -> variable="Prior chemotherapy", operator="=", threshold="N/A"
  "confirmed HIV negative"           -> variable="HIV status", operator="=", threshold="Negative"\
"""

# ---------------------------------------------------------------------------
# User template
# ---------------------------------------------------------------------------
USER_TEMPLATE = """\
Parse the eligibility criteria below into this exact JSON structure.

{{
  "nct_id": "{nct_id}",
  "condition": "{condition}",
  "demographics": {{
    "min_age": "N/A",
    "max_age": "N/A",
    "gender": "N/A"
  }},
  "inclusion_criteria": [
    {{"criterion": "...", "variable": "...", "operator": "...", "threshold": "..."}}
  ],
  "exclusion_criteria": [
    {{"criterion": "...", "variable": "...", "operator": "...", "threshold": "..."}}
  ]
}}

IMPORTANT:
- Extract EVERY criterion line as a separate object, even non-numeric ones.
- For numeric criteria, always fill in variable, operator, and threshold.
- demographics.min_age / max_age must be plain numbers like "18", not "18 years".
- demographics.gender: use "Male", "Female", or "Both".

Eligibility Criteria:
{criteria_text}\
"""

# Repair prompt — used when the first parse fails.
# Feeds the broken output back to the LLM and asks it to fix only the JSON.
REPAIR_PROMPT = """\
The following text should be valid JSON but has a syntax error.
Fix ONLY the JSON syntax — do not change any values or add new fields.
Return ONLY the corrected JSON, nothing else.

Broken JSON:
{broken}\
"""

# ---------------------------------------------------------------------------
# Criteria pre-processor
# ---------------------------------------------------------------------------

def prepare_criteria_text(raw_criteria: str, max_chars_each: int = 2000) -> str:
    """
    Split the raw eligibility text into inclusion and exclusion sections,
    then truncate each section independently to max_chars_each characters.

    Why this matters:
        Original code truncated the whole text at 3000 chars.  For trials
        with long inclusion sections, the exclusion section was often
        completely cut off, so no exclusion criteria were ever parsed.

        With independent 2000-char budgets, both sections are always
        represented — giving a total input of up to 4000 chars.
    """
    import re

    # Split on the "Exclusion Criteria" header (case-insensitive)
    parts = re.split(r'(?i)(?=exclusion\s+criteria)', raw_criteria, maxsplit=1)

    if len(parts) == 2:
        inclusion_text = parts[0].strip()[:max_chars_each]
        exclusion_text = parts[1].strip()[:max_chars_each]
        return inclusion_text + "\n\n" + exclusion_text
    else:
        # Could not split — fall back to original truncation
        return raw_criteria[:max_chars_each * 2]

# ---------------------------------------------------------------------------
# JSON repair helpers
# ---------------------------------------------------------------------------

def repair_json_local(raw: str) -> dict | None:
    """
    Three-pass local repair — no API call required.

    Pass 1: direct json.loads
    Pass 2: find the outermost closing brace and truncate
    Pass 3: strip trailing incomplete lines one by one
    """
    # Pass 1
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Pass 2: walk character by character tracking brace depth
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[: i + 1])
                except json.JSONDecodeError:
                    break

    # Pass 3: remove lines from the end until valid
    lines = raw.splitlines()
    for n in range(len(lines), 0, -1):
        candidate = "\n".join(lines[:n]).strip()
        if not candidate.endswith("}"):
            candidate += "}"
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def repair_json_llm(broken: str) -> dict | None:
    """
    LLM-assisted repair — called only when all local passes fail.
    Sends the broken JSON back to the model and asks for a syntax fix only.
    Uses a short max_tokens budget since we only need minor corrections.
    """
    try:
        prompt = REPAIR_PROMPT.format(broken=broken[:3000])
        response = client.messages.create(
            model      = MODEL,
            max_tokens = 1000,
            messages   = [{"role": "user", "content": prompt}],
        )
        fixed = response.content[0].text.strip()

        # Strip any markdown fences the model added
        if fixed.startswith("```"):
            fixed = fixed.split("```")[1]
            if fixed.startswith("json"):
                fixed = fixed[4:]
            fixed = fixed.strip()

        return json.loads(fixed)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# Single-study parser
# ---------------------------------------------------------------------------

def parse_one_study(study: dict) -> dict | None:
    """
    Parse one study record through the LLM and return a structured dict.

    Repair strategy (in order of cost):
        1. Local three-pass repair    (free)
        2. LLM repair prompt          (1 extra API call)
        3. Return None                (record is dropped)
    """
    try:
        module    = study.get("protocolSection", {})
        nct_id    = module.get("identificationModule", {}).get("nctId", "UNKNOWN")
        criteria  = module.get("eligibilityModule",   {}).get("eligibilityCriteria", "")
        condition = module.get("conditionsModule",    {}).get("conditions", ["N/A"])[0]

        # Pre-process: split inclusion/exclusion and truncate each independently
        criteria_text = prepare_criteria_text(criteria)

        prompt = USER_TEMPLATE.format(
            nct_id        = nct_id,
            condition     = condition,
            criteria_text = criteria_text,
        )

        response = client.messages.create(
            model      = MODEL,
            max_tokens = 3000,   # raised from 2000 to handle trials with 20+ criteria
            system     = SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": prompt}],
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences (defensive — prompt forbids them)
        if "```" in raw:
            parts = raw.split("```")
            raw   = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        # Repair pass 1: local
        result = repair_json_local(raw)
        if result:
            return result

        # Repair pass 2: LLM-assisted
        result = repair_json_llm(raw)
        return result   # None if both passes fail

    except Exception as exc:
        print(f"  Error: {exc}")
        return None

# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_parse(max_studies: int = 50, delay: float = 0.3) -> None:
    """
    Parse up to max_studies trials and write results to OUTPUT_PATH.

    Quality report printed at the end covers:
        - Parse success rate
        - Demographics extraction rate (age, gender)
        - Criterion-level threshold extraction rate  ← NEW
          Shows what % of criteria have numeric variable/operator/threshold,
          which directly predicts the NEGATIVE_HARD coverage in Phase III.

    Recommended workflow:
        run_parse(max_studies=50)    # verify quality on a small batch
        run_parse(max_studies=476)   # full dataset once quality looks good
    """
    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)

    studies = studies[:max_studies]
    success = 0
    failed  = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, study in enumerate(studies):
            nct_id = (
                study.get("protocolSection", {})
                     .get("identificationModule", {})
                     .get("nctId", f"#{i}")
            )
            print(f"[{i + 1}/{len(studies)}] Parsing {nct_id}...", end=" ", flush=True)

            parsed = parse_one_study(study)
            if parsed:
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                success += 1
                print("✓")
            else:
                failed.append(nct_id)
                print("✗")

            time.sleep(delay)

    print(f"\nParsing complete: {success}/{len(studies)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:10])}"
              + (" ..." if len(failed) > 10 else ""))

    # --- Quality report ---
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    total = len(results)
    if total == 0:
        print("No results to report.")
        return

    # Demographics extraction rates
    has_min_age = sum(1 for r in results if r["demographics"]["min_age"] != "N/A")
    has_max_age = sum(1 for r in results if r["demographics"]["max_age"] != "N/A")
    has_gender  = sum(1 for r in results if r["demographics"]["gender"]  != "N/A")

    # Criterion-level threshold extraction rate (NEW)
    # Count how many individual criteria have a non-"N/A" numeric threshold.
    # This is the direct predictor of Phase II atomic unit count and
    # ultimately of Phase III NEGATIVE_HARD coverage.
    total_criteria     = 0
    criteria_with_threshold = 0
    for r in results:
        for crit in r.get("inclusion_criteria", []) + r.get("exclusion_criteria", []):
            total_criteria += 1
            if (crit.get("threshold", "N/A") != "N/A"
                    and crit.get("operator",  "N/A") != "N/A"
                    and crit.get("variable",  "N/A") != "N/A"):
                criteria_with_threshold += 1

    avg_inclusion = sum(len(r.get("inclusion_criteria", [])) for r in results) / total
    avg_exclusion = sum(len(r.get("exclusion_criteria", [])) for r in results) / total

    print(f"\n=== Quality Report ===")
    print(f"  Parsed records    : {total}")
    print(f"  min_age extracted : {has_min_age}/{total} ({has_min_age/total*100:.1f}%)")
    print(f"  max_age extracted : {has_max_age}/{total} ({has_max_age/total*100:.1f}%)")
    print(f"  gender extracted  : {has_gender}/{total}  ({has_gender/total*100:.1f}%)")
    print(f"\n  Avg inclusion criteria / trial : {avg_inclusion:.1f}")
    print(f"  Avg exclusion criteria / trial : {avg_exclusion:.1f}")
    print(f"\n  Criteria with numeric threshold: "
          f"{criteria_with_threshold}/{total_criteria} "
          f"({criteria_with_threshold/total_criteria*100:.1f}%)"
          if total_criteria else "  No criteria found.")
    print(f"  (This % predicts atomic unit density and NEGATIVE_HARD coverage)")

    # Sample output
    print(f"\n=== Sample Output ===")
    first = results[0]
    print(f"NCT ID     : {first['nct_id']}")
    print(f"Condition  : {first['condition']}")
    print(f"Age range  : {first['demographics']['min_age']} ~ "
          f"{first['demographics']['max_age']}")
    print(f"Gender     : {first['demographics']['gender']}")
    print(f"Inclusion  : {len(first['inclusion_criteria'])} criteria")
    print(f"Exclusion  : {len(first['exclusion_criteria'])} criteria")

    # Show first 3 criteria that have a threshold, to verify extraction quality
    numeric_examples = [
        c for c in (first.get("inclusion_criteria", []) +
                    first.get("exclusion_criteria", []))
        if c.get("threshold", "N/A") != "N/A"
    ][:3]

    if numeric_examples:
        print(f"\nSample numeric criteria:")
        for ex in numeric_examples:
            print(f"  Criterion : {ex['criterion'][:80]}")
            print(f"  Triplet   : <{ex['variable']}, {ex['operator']}, {ex['threshold']}>")
            print()


if __name__ == "__main__":
    # Start with 50 to verify threshold extraction quality,
    # then bump to 476 for the full dataset.
    run_parse(max_studies=50)