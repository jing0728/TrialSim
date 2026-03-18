import json, re, os, time
from anthropic import Anthropic

INPUT_PATH  = "data/raw/filtered_studies.json"
OUTPUT_PATH = "data/raw/parsed_pico.jsonl"

client = Anthropic()

SYSTEM_PROMPT = """You are a senior Clinical Data Scientist specializing in regulatory compliance.
Parse clinical trial eligibility criteria into strict JSON.

RULES:
- Output ONLY valid JSON, no markdown, no explanation.
- For any field not explicitly stated, use exactly "N/A".
- Never fabricate or infer values not in the text.
- Extract numerical boundaries precisely."""

USER_TEMPLATE = """Parse the eligibility criteria below into this exact JSON:

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
            model="claude-haiku-4-5-20251001",  # cheapest model, good enough for parsing
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        return json.loads(raw.strip())

    except Exception as e:
        print(f"  Error: {e}")
        return None


def run_parse(max_studies=50):
    """Start with 50 studies to test quality before running full dataset."""
    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)

    studies = studies[:max_studies]
    success = 0

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, study in enumerate(studies):
            nct_id = (study.get("protocolSection", {})
                          .get("identificationModule", {})
                          .get("nctId", f"#{i}"))
            print(f"[{i+1}/{len(studies)}] Parsing {nct_id}...", end=" ")

            parsed = parse_one_study(study)
            if parsed:
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                success += 1
                print("✓")
            else:
                print("✗")

            time.sleep(0.3)

    print(f"\nComplete: {success}/{len(studies)} succeeded")

    # Check age extraction improvement
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        results = [json.loads(l) for l in f]

    has_age = sum(1 for r in results if r["demographics"]["min_age"] != "N/A")
    print(f"Age extracted: {has_age}/{len(results)} ({has_age/len(results)*100:.1f}%)")


if __name__ == "__main__":
    run_parse(max_studies=50)