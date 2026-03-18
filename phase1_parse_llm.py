import json, os, time
from anthropic import Anthropic

INPUT_PATH  = "data/raw/filtered_studies.json"
OUTPUT_PATH = "data/raw/parsed_pico.jsonl"

client = Anthropic()

SYSTEM_PROMPT = """You are a senior Clinical Data Scientist specializing in regulatory compliance.
Parse clinical trial eligibility criteria into strict JSON.

RULES:
- Output ONLY valid JSON, no markdown, no explanation, no preamble.
- For any field not explicitly stated in the text, use exactly "N/A".
- Never fabricate or infer values not present in the text.
- Extract numerical boundaries precisely (e.g. "aged 18 to 65" -> min_age: "18", max_age: "65").
- For criteria lines that have no numeric threshold, set operator and threshold to "N/A"."""

USER_TEMPLATE = """Parse the eligibility criteria below into this exact JSON structure:

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

Eligibility Criteria:
{criteria_text}"""


def repair_json(raw: str) -> dict | None:
    """
    Attempt to parse JSON, and if it fails due to truncation or
    special characters, find the outermost closing brace and truncate.
    """
    # First attempt: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second attempt: find outermost closing brace
    depth = 0
    for i, ch in enumerate(raw):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[:i + 1])
                except json.JSONDecodeError:
                    break

    # Third attempt: strip trailing incomplete lines
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


def parse_one_study(study: dict) -> dict | None:
    try:
        module    = study.get("protocolSection", {})
        nct_id    = module.get("identificationModule", {}).get("nctId", "UNKNOWN")
        criteria  = module.get("eligibilityModule", {}).get("eligibilityCriteria", "")
        condition = module.get("conditionsModule", {}).get("conditions", ["N/A"])[0]

        prompt = USER_TEMPLATE.format(
            nct_id=nct_id,
            condition=condition,
            criteria_text=criteria[:3000]
        )

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        return repair_json(raw)

    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_parse(max_studies: int = 50):
    """
    Parse studies using LLM.
    Start with max_studies=50 to verify quality, then increase to full dataset.
    """
    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)

    studies = studies[:max_studies]
    success = 0
    failed  = []

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, study in enumerate(studies):
            nct_id = (study.get("protocolSection", {})
                          .get("identificationModule", {})
                          .get("nctId", f"#{i}"))
            print(f"[{i+1}/{len(studies)}] Parsing {nct_id}...", end=" ", flush=True)

            parsed = parse_one_study(study)
            if parsed:
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                success += 1
                print("✓")
            else:
                failed.append(nct_id)
                print("✗")

            time.sleep(0.3)

    print(f"\nComplete: {success}/{len(studies)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed)}")

    # Quality check: age extraction rate
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        results = [json.loads(l) for l in f]

    has_min_age = sum(1 for r in results if r["demographics"]["min_age"] != "N/A")
    has_max_age = sum(1 for r in results if r["demographics"]["max_age"] != "N/A")
    has_gender  = sum(1 for r in results if r["demographics"]["gender"]  != "N/A")
    total       = len(results)

    print(f"\n=== Quality Report ===")
    print(f"  min_age extracted : {has_min_age}/{total} ({has_min_age/total*100:.1f}%)")
    print(f"  max_age extracted : {has_max_age}/{total} ({has_max_age/total*100:.1f}%)")
    print(f"  gender extracted  : {has_gender}/{total}  ({has_gender/total*100:.1f}%)")

    # Sample output
    print(f"\n=== Sample Output ===")
    first = results[0]
    print(f"NCT ID     : {first['nct_id']}")
    print(f"Condition  : {first['condition']}")
    print(f"Age range  : {first['demographics']['min_age']} ~ {first['demographics']['max_age']}")
    print(f"Gender     : {first['demographics']['gender']}")
    print(f"Inclusion  : {len(first['inclusion_criteria'])} criteria")
    print(f"Exclusion  : {len(first['exclusion_criteria'])} criteria")
    if first["inclusion_criteria"]:
        ex = first["inclusion_criteria"][0]
        print(f"\nFirst inclusion criterion:")
        print(f"  Text     : {ex['criterion'][:80]}...")
        print(f"  Variable : {ex['variable']}")
        print(f"  Operator : {ex['operator']}")
        print(f"  Threshold: {ex['threshold']}")


if __name__ == "__main__":
    run_parse(max_studies=476)