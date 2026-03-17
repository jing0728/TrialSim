import json, re, os

INPUT_PATH  = "data/raw/filtered_studies.json"
OUTPUT_PATH = "data/raw/parsed_pico.jsonl"


def extract_demographics(text: str) -> dict:
    """Extract age range and gender from eligibility text."""
    demo = {"min_age": "N/A", "max_age": "N/A", "gender": "N/A"}

    # Pattern 1: "between the ages of 18 and 55"
    m = re.search(r'between\s+the\s+ages?\s+of\s+(\d+)\s+and\s+(\d+)', text, re.I)
    if m:
        demo["min_age"] = m.group(1)
        demo["max_age"] = m.group(2)

    # Pattern 2: "aged 18 to 65" / "ages 18-65"
    if demo["min_age"] == "N/A":
        m = re.search(r'age[sd]?\s+(\d+)\s*(?:to|-|and)\s*(\d+)', text, re.I)
        if m:
            demo["min_age"] = m.group(1)
            demo["max_age"] = m.group(2)

    # Pattern 3: "between 18 and 65 years"
    if demo["min_age"] == "N/A":
        m = re.search(r'between\s+(\d+)\s+and\s+(\d+)\s*year', text, re.I)
        if m:
            demo["min_age"] = m.group(1)
            demo["max_age"] = m.group(2)

    # Pattern 4: "Any child aged 5 years and below" -> max_age only
    if demo["max_age"] == "N/A":
        m = re.search(r'aged?\s+(\d+)\s+years?\s+and\s+below', text, re.I)
        if m:
            demo["max_age"] = m.group(1)

    # Pattern 5: "Age 75 years and over" / "18 years and over/above"
    if demo["min_age"] == "N/A":
        m = re.search(r'age[d]?\s+(\d+)\s+years?\s+and\s+(?:over|above)', text, re.I)
        if not m:
            m = re.search(r'(\d+)\s+years?\s+and\s+(?:over|above)', text, re.I)
        if m:
            demo["min_age"] = m.group(1)

    # Pattern 6: escaped operators "\\< 13 years" / "\\> 18 years" (from exclusion)
    # These appear in exclusion criteria, so we map them to the opposite bound
    if demo["min_age"] == "N/A":
        # "\\> X years old" in exclusion means min_age = X
        m = re.search(r'[\\>]+\s*(\d+)\s*years?\s*old', text, re.I)
        if m:
            demo["min_age"] = m.group(1)
    if demo["max_age"] == "N/A":
        # "\\< X years old" in exclusion means max_age = X
        m = re.search(r'[\\<]+\s*(\d+)\s*years?\s*old', text, re.I)
        if m:
            demo["max_age"] = m.group(1)
    # Pattern 6b: "Subject >= 20 years of age" / ">= 18 years of age"
    if demo["min_age"] == "N/A":
        m = re.search(r'[≥>=]+\s*(\d+)\s+years?\s+of\s+age', text, re.I)
        if m:
            demo["min_age"] = m.group(1)

    # Pattern 6c: "18 years old or above" / "18 years old or older"
    if demo["min_age"] == "N/A":
        m = re.search(r'(\d+)\s+years?\s+old\s+or\s+(?:above|over|older)', text, re.I)
        if m:
            demo["min_age"] = m.group(1)
# Pattern 6d: "65 years of age and older/above"
    if demo["min_age"] == "N/A":
        m = re.search(r'(\d+)\s+years?\s+of\s+age\s+and\s+(?:older|over|above)', text, re.I)
        if m:
            demo["min_age"] = m.group(1)

    # Pattern 6e: "aged 5 years and below" (with \- bullet prefix)
    if demo["max_age"] == "N/A":
        m = re.search(r'aged?\s+(\d+)\s+years?\s+and\s+below', text, re.I)
        if m:
            demo["max_age"] = m.group(1)
    # Pattern 7: lower bound only — ">= 18" / "at least 18" / "18 years or older"
    if demo["min_age"] == "N/A":
        m = re.search(r'(?:age[d]?\s*[≥>=]+\s*|at least\s*)(\d+)', text, re.I)
        if not m:
            m = re.search(r'(\d+)\s*years?\s+or\s+older', text, re.I)
        if m:
            demo["min_age"] = m.group(1)

    # Pattern 8: upper bound only — "<= 75" / "no older than 75"
    if demo["max_age"] == "N/A":
        m = re.search(r'(?:age[d]?\s*[≤<=]+\s*|no older than\s*|up to\s*)(\d+)', text, re.I)
        if m:
            demo["max_age"] = m.group(1)

    # Pattern 9: "minimum age X" / "maximum age X"
    if demo["min_age"] == "N/A":
        m = re.search(r'minimum\s+age\s+(?:of\s+)?(\d+)', text, re.I)
        if m:
            demo["min_age"] = m.group(1)
    if demo["max_age"] == "N/A":
        m = re.search(r'maximum\s+age\s+(?:of\s+)?(\d+)', text, re.I)
        if m:
            demo["max_age"] = m.group(1)

    # Pattern 10: "X year old" / "X-year-old" as age cutoff
    if demo["min_age"] == "N/A":
        m = re.search(r'(\d+)[- ]year[- ]old\s+(?:and\s+)?(?:over|above|older)', text, re.I)
        if m:
            demo["min_age"] = m.group(1)

    # Gender
    text_l = text.lower()
    if "male" in text_l and "female" in text_l:
        demo["gender"] = "Both"
    elif "female" in text_l or "women" in text_l:
        demo["gender"] = "Female"
    elif "male" in text_l or "men" in text_l:
        demo["gender"] = "Male"

    return demo


def split_inclusion_exclusion(text: str):
    """Split eligibility text into inclusion and exclusion sections."""
    parts = re.split(r'(?=exclusion criteria)', text, flags=re.I)
    if len(parts) == 2:
        return parts[0], parts[1]
    return text, ""


def extract_criteria_lines(section: str) -> list[str]:
    """Split a section into individual criterion lines by bullet or numbering."""
    section = re.sub(r'^.*?criteria[:\s]*', '', section, flags=re.I)
    lines = re.split(r'\n\s*(?:\\-|[-•\*]|\d+[\.\)])\s*', section)
    return [l.strip() for l in lines if len(l.strip()) > 10]


def parse_criterion(line: str) -> dict:
    """
    Attempt to extract <variable, operator, threshold> from a single criterion line.
    Covers numeric operators AND plain age/BMI range patterns.
    """
    result = {
        "criterion": line,
        "variable":  "N/A",
        "operator":  "N/A",
        "threshold": "N/A"
    }

    # Pattern A: "between the ages of 18 and 55"
    m = re.search(r'between\s+the\s+ages?\s+of\s+(\d+)\s+and\s+(\d+)', line, re.I)
    if m:
        result["variable"]  = "Age"
        result["operator"]  = "between"
        result["threshold"] = f"{m.group(1)}-{m.group(2)}"
        return result

    # Pattern B: "aged X to Y" / "ages X-Y"
    m = re.search(r'age[sd]?\s+(\d+)\s*(?:to|-)\s*(\d+)', line, re.I)
    if m:
        result["variable"]  = "Age"
        result["operator"]  = "between"
        result["threshold"] = f"{m.group(1)}-{m.group(2)}"
        return result

    # Pattern C: "X years or older/younger"
    m = re.search(r'(\d+)\s*years?\s+or\s+(older|younger)', line, re.I)
    if m:
        result["variable"]  = "Age"
        result["operator"]  = ">=" if m.group(2).lower() == "older" else "<="
        result["threshold"] = m.group(1)
        return result

    # Pattern D: "X years and over/above"
    m = re.search(r'(\d+)\s+years?\s+and\s+(?:over|above)', line, re.I)
    if m:
        result["variable"]  = "Age"
        result["operator"]  = ">="
        result["threshold"] = m.group(1)
        return result

    # Pattern E: "aged X years and below"
    m = re.search(r'aged?\s+(\d+)\s+years?\s+and\s+below', line, re.I)
    if m:
        result["variable"]  = "Age"
        result["operator"]  = "<="
        result["threshold"] = m.group(1)
        return result

    # Pattern F: escaped operators "\\< 13 years old" / "\\> 18 years old"
    m = re.search(r'[\\]+([<>])\s*(\d+)\s*years?\s*old', line, re.I)
    if m:
        result["variable"]  = "Age"
        result["operator"]  = m.group(1)
        result["threshold"] = m.group(2)
        return result

    # Pattern G: BMI range — "BMI 18.5 to 30"
    m = re.search(r'bmi\s*(?:of\s*)?([\d.]+)\s*(?:to|-)\s*([\d.]+)', line, re.I)
    if m:
        result["variable"]  = "BMI"
        result["operator"]  = "between"
        result["threshold"] = f"{m.group(1)}-{m.group(2)}"
        return result

    # Pattern H: generic operator — "HbA1c < 7.0" / "eGFR >= 30 ml/min"
    m = re.search(
        r'([\w][\w\s]{2,25}?)\s*([<>≤≥=]+)\s*([\d.]+\s*(?:%|mg|ml|mmol|ng|pg|iu|g)?)',
        line, re.I
    )
    if m:
        result["variable"]  = m.group(1).strip()
        result["operator"]  = m.group(2).strip()
        result["threshold"] = m.group(3).strip()
        return result

    return result


def parse_study(study: dict) -> dict | None:
    """Parse a single study record into structured PICO format."""
    try:
        module    = study.get("protocolSection", {})
        nct_id    = module.get("identificationModule", {}).get("nctId", "UNKNOWN")
        criteria  = module.get("eligibilityModule", {}).get("eligibilityCriteria", "")
        condition = module.get("conditionsModule", {}).get("conditions", ["N/A"])[0]

        inc_text, exc_text = split_inclusion_exclusion(criteria)
        inc_lines = extract_criteria_lines(inc_text)
        exc_lines = extract_criteria_lines(exc_text)

        return {
            "nct_id":             nct_id,
            "condition":          condition,
            "demographics":       extract_demographics(criteria),
            "inclusion_criteria": [parse_criterion(l) for l in inc_lines],
            "exclusion_criteria": [parse_criterion(l) for l in exc_lines],
        }
    except Exception as e:
        print(f"  Parse error: {e}")
        return None


def run_parse():
    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)

    success = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for i, study in enumerate(studies):
            parsed = parse_study(study)
            if parsed:
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                success += 1

    print(f"Parsing complete: {success}/{len(studies)} studies succeeded")
    print(f"Output saved to: {OUTPUT_PATH}")

    # Print one example to verify output quality
    with open(OUTPUT_PATH, encoding="utf-8") as f:
        first = json.loads(f.readline())

    print("\n=== Sample Output ===")
    print(f"NCT ID     : {first['nct_id']}")
    print(f"Condition  : {first['condition']}")
    print(f"Age range  : {first['demographics']['min_age']} ~ {first['demographics']['max_age']}")
    print(f"Gender     : {first['demographics']['gender']}")
    print(f"Inclusion criteria count : {len(first['inclusion_criteria'])}")
    print(f"Exclusion criteria count : {len(first['exclusion_criteria'])}")

    if first['inclusion_criteria']:
        ex = first['inclusion_criteria'][0]
        print(f"\nFirst inclusion criterion:")
        print(f"  Text     : {ex['criterion'][:80]}...")
        print(f"  Variable : {ex['variable']}")
        print(f"  Operator : {ex['operator']}")
        print(f"  Threshold: {ex['threshold']}")


if __name__ == "__main__":
    run_parse()