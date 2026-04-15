"""
phase1_parse_llm.py

Final LLM-based Phase I parser for TrialSim.

Goal:
    Convert filtered_studies.json into parsed_pico.jsonl with richer criterion extraction,
    including:
      - numeric constraints
      - categorical constraints
      - status / presence / absence constraints

Pipeline position:
    filtered_studies.json  ->  [this file]  ->  parsed_pico.jsonl
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

INPUT_PATH = "data/raw/filtered_studies.json"
OUTPUT_PATH = "data/raw/parsed_pico.jsonl"
CHECKPOINT_PATH = OUTPUT_PATH + ".done"

MODEL = "gpt-4o-mini"
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=API_KEY) if (OpenAI is not None and API_KEY) else None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a senior Clinical Data Scientist specializing in regulatory compliance.
Parse clinical trial eligibility criteria into strict JSON.

You must extract ALL verifiable constraints, not only numeric ones.

Allowed operator values:
- ">="
- "<="
- ">"
- "<"
- "between"
- "="
- "presence"
- "absence"
- "N/A"

Rules:
- Output ONLY valid JSON.
- No markdown, no explanation, no preamble.
- Never fabricate values.
- If a field is not explicitly stated, use "N/A".
- Extract every criterion line as a separate object.
- For demographics.gender, use "Male", "Female", or "Both".

Examples:
- "aged 18 to 65" -> variable="Age", operator="between", threshold="18-65"
- "at least 18 years old" -> variable="Age", operator=">=", threshold="18"
- "HbA1c less than 7.0%" -> variable="HbA1c", operator="<", threshold="7.0"
- "ECOG performance status 0 or 1" -> variable="ECOG performance status", operator="between", threshold="0-1"
- "confirmed HIV negative" -> variable="HIV status", operator="=", threshold="Negative"
- "informed consent signed" -> variable="Informed consent", operator="presence", threshold="Signed"
- "no prior chemotherapy" -> variable="Prior chemotherapy", operator="absence", threshold="Chemotherapy"
- "histologically confirmed urothelial carcinoma" -> variable="Cancer type", operator="=", threshold="Urothelial carcinoma"
"""

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
- Extract EVERY criterion line as a separate object, including numeric, categorical, and status constraints.
- For numeric criteria, always fill variable/operator/threshold when present.
- For categorical criteria, use operator "=".
- For status constraints like signed/confirmed/present, use operator "presence".
- For "no X" constraints, use operator "absence".
- demographics.min_age / max_age must be plain numbers like "18", not "18 years".

Eligibility Criteria:
{criteria_text}
"""

REPAIR_PROMPT = """\
The following text should be valid JSON but has a syntax error.
Fix ONLY the JSON syntax. Do not change any values or add new fields.
Return ONLY corrected JSON.

Broken JSON:
{broken}
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_block(raw: str) -> str | None:
    if not raw:
        return None

    raw = raw.strip()

    if raw.startswith("{") and raw.endswith("}"):
        return raw

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        return match.group(0)

    return None


def _call_openai_text(
    prompt: str,
    *,
    system: str | None = None,
    max_output_tokens: int = 3000,
    retries: int = 4,
) -> str | None:
    if client is None:
        return None

    for attempt in range(retries):
        try:
            if system:
                response = client.responses.create(
                    model=MODEL,
                    instructions=system,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                )
            else:
                response = client.responses.create(
                    model=MODEL,
                    input=prompt,
                    max_output_tokens=max_output_tokens,
                )
            return response.output_text.strip()

        except Exception as exc:
            print(f"    [LLM retry {attempt + 1}/{retries}] {type(exc).__name__}")
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)

    return None


def repair_json_local(raw: str) -> dict | None:
    """
    Three-pass local repair.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # outermost brace truncate
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

    # strip trailing lines
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
    fixed = _call_openai_text(
        REPAIR_PROMPT.format(broken=broken[:4000]),
        max_output_tokens=1200,
        retries=2,
    )
    if not fixed:
        return None

    block = _extract_json_block(fixed)
    if not block:
        return None

    try:
        return json.loads(block)
    except json.JSONDecodeError:
        return None


def prepare_criteria_text(raw_criteria: str, max_chars_each: int = 2200) -> str:
    """
    Split into inclusion / exclusion and truncate each independently.
    """
    if not raw_criteria:
        return ""

    parts = re.split(r'(?i)(?=exclusion\s+criteria)', raw_criteria, maxsplit=1)

    if len(parts) == 2:
        inclusion_text = parts[0].strip()[:max_chars_each]
        exclusion_text = parts[1].strip()[:max_chars_each]
        return inclusion_text + "\n\n" + exclusion_text

    return raw_criteria[: max_chars_each * 2]


def normalize_record(data: dict[str, Any], nct_id: str, condition: str) -> dict[str, Any]:
    """
    Ensure the final schema is complete and safe.
    """
    demo = data.get("demographics", {}) if isinstance(data, dict) else {}

    normalized = {
        "nct_id": data.get("nct_id", nct_id) if isinstance(data, dict) else nct_id,
        "condition": data.get("condition", condition) if isinstance(data, dict) else condition,
        "demographics": {
            "min_age": str(demo.get("min_age", "N/A")) if demo else "N/A",
            "max_age": str(demo.get("max_age", "N/A")) if demo else "N/A",
            "gender": str(demo.get("gender", "N/A")) if demo else "N/A",
        },
        "inclusion_criteria": [],
        "exclusion_criteria": [],
    }

    for key in ("inclusion_criteria", "exclusion_criteria"):
        items = data.get(key, []) if isinstance(data, dict) else []
        if not isinstance(items, list):
            items = []

        cleaned = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cleaned.append({
                "criterion": str(item.get("criterion", "N/A")).strip() or "N/A",
                "variable": str(item.get("variable", "N/A")).strip() or "N/A",
                "operator": str(item.get("operator", "N/A")).strip() or "N/A",
                "threshold": str(item.get("threshold", "N/A")).strip() or "N/A",
            })
        normalized[key] = cleaned

    return normalized


def parse_one_study(study: dict, max_retries: int = 4) -> dict | None:
    module = study.get("protocolSection", {})
    nct_id = module.get("identificationModule", {}).get("nctId", "UNKNOWN")
    criteria = module.get("eligibilityModule", {}).get("eligibilityCriteria", "")
    condition = module.get("conditionsModule", {}).get("conditions", ["N/A"])[0]

    if not criteria:
        return None

    criteria_text = prepare_criteria_text(criteria)
    prompt = USER_TEMPLATE.format(
        nct_id=nct_id,
        condition=condition,
        criteria_text=criteria_text,
    )

    for attempt in range(max_retries):
        raw = _call_openai_text(
            prompt,
            system=SYSTEM_PROMPT,
            max_output_tokens=3200,
            retries=2,
        )
        if not raw:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            continue

        block = _extract_json_block(raw)
        if not block:
            if attempt == max_retries - 1:
                return None
            time.sleep(2 ** attempt)
            continue

        parsed = repair_json_local(block)
        if parsed is None:
            parsed = repair_json_llm(block)

        if parsed is not None:
            return normalize_record(parsed, nct_id=nct_id, condition=condition)

        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)

    return None


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_quality_report(records: list[dict]) -> None:
    total = len(records)
    if total == 0:
        print("\n=== Quality Report ===")
        print("  No parsed records.")
        return

    min_age_count = sum(1 for r in records if r["demographics"].get("min_age") != "N/A")
    max_age_count = sum(1 for r in records if r["demographics"].get("max_age") != "N/A")
    gender_count = sum(1 for r in records if r["demographics"].get("gender") != "N/A")

    inc_counts = [len(r.get("inclusion_criteria", [])) for r in records]
    exc_counts = [len(r.get("exclusion_criteria", [])) for r in records]

    all_criteria = sum(inc_counts) + sum(exc_counts)

    numeric_ops = {">=", "<=", ">", "<", "between"}
    numeric_threshold_count = 0
    categorical_or_status_count = 0

    for r in records:
        for key in ("inclusion_criteria", "exclusion_criteria"):
            for item in r.get(key, []):
                op = item.get("operator", "N/A")
                threshold = item.get("threshold", "N/A")
                if op in numeric_ops and threshold != "N/A":
                    numeric_threshold_count += 1
                elif op in {"=", "presence", "absence"} and threshold != "N/A":
                    categorical_or_status_count += 1

    print("\n=== Quality Report ===")
    print(f"  Parsed records    : {total}")
    print(f"  min_age extracted : {min_age_count}/{total} ({min_age_count / total * 100:.1f}%)")
    print(f"  max_age extracted : {max_age_count}/{total} ({max_age_count / total * 100:.1f}%)")
    print(f"  gender extracted  : {gender_count}/{total} ({gender_count / total * 100:.1f}%)")
    print()
    print(f"  Avg inclusion criteria / trial : {sum(inc_counts) / total:.1f}")
    print(f"  Avg exclusion criteria / trial : {sum(exc_counts) / total:.1f}")
    print()
    print(f"  Criteria with numeric threshold     : {numeric_threshold_count}/{all_criteria} ({numeric_threshold_count / all_criteria * 100:.1f}%)")
    print(f"  Categorical / status constraints    : {categorical_or_status_count}/{all_criteria} ({categorical_or_status_count / all_criteria * 100:.1f}%)")
    print(f"  Total structured constraints        : {(numeric_threshold_count + categorical_or_status_count)}/{all_criteria} ({(numeric_threshold_count + categorical_or_status_count) / all_criteria * 100:.1f}%)")

    sample = records[0]
    print("\n=== Sample Output ===")
    print(f"NCT ID     : {sample.get('nct_id')}")
    print(f"Condition  : {sample.get('condition')}")
    print(f"Age range  : {sample['demographics'].get('min_age')} ~ {sample['demographics'].get('max_age')}")
    print(f"Gender     : {sample['demographics'].get('gender')}")
    print(f"Inclusion  : {len(sample.get('inclusion_criteria', []))} criteria")
    print(f"Exclusion  : {len(sample.get('exclusion_criteria', []))} criteria")

    shown = 0
    print("\nSample structured criteria:")
    for key in ("inclusion_criteria", "exclusion_criteria"):
        for item in sample.get(key, [])[:10]:
            print(f"  Criterion : {item['criterion'][:90]}")
            print(f"  Triplet   : <{item['variable']}, {item['operator']}, {item['threshold']}>")
            print()
            shown += 1
            if shown >= 3:
                return


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_parse(max_studies: int | None = None, overwrite: bool = False) -> list[dict]:
    if client is None:
        raise RuntimeError("OpenAI client unavailable. Install openai and set OPENAI_API_KEY.")

    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)

    if max_studies is not None:
        studies = studies[:max_studies]

    if overwrite:
        if os.path.exists(OUTPUT_PATH):
            os.remove(OUTPUT_PATH)
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)

    done_ids: set[str] = set()
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, encoding="utf-8") as f:
            done_ids = {line.strip() for line in f if line.strip()}

    pending = []
    for s in studies:
        sid = (
            s.get("protocolSection", {})
             .get("identificationModule", {})
             .get("nctId", "")
        )
        if sid not in done_ids:
            pending.append(s)

    print(f"Resuming — {len(done_ids)} records already done, {len(pending)} remaining.")

    parsed_records: list[dict] = []

    # load already-parsed records for reporting
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        parsed_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    out_mode = "a" if os.path.exists(OUTPUT_PATH) and not overwrite else "w"

    with open(OUTPUT_PATH, out_mode, encoding="utf-8") as out, \
         open(CHECKPOINT_PATH, "a", encoding="utf-8") as ckpt:

        total = len(studies)
        completed = len(done_ids)

        for study in pending:
            module = study.get("protocolSection", {})
            nct_id = module.get("identificationModule", {}).get("nctId", "UNKNOWN")

            print(f"[{completed + 1}/{total}] Parsing {nct_id}...", end=" ")

            parsed = parse_one_study(study)
            if parsed is not None:
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                out.flush()

                ckpt.write(nct_id + "\n")
                ckpt.flush()

                parsed_records.append(parsed)
                completed += 1
                print("✓")
            else:
                print("✗")

    print(f"\nParsing complete: {completed}/{len(studies)} succeeded")
    print_quality_report(parsed_records)
    return parsed_records


if __name__ == "__main__":
    # Set overwrite=True if you want to rebuild parsed_pico.jsonl from scratch.
    run_parse(max_studies=None, overwrite=True)