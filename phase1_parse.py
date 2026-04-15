import json
import os
import re

INPUT_PATH = "data/raw/filtered_studies.json"
OUTPUT_PATH = "data/raw/parsed_pico.jsonl"


def split_inclusion_exclusion(criteria: str) -> tuple[str, str]:
    if not criteria:
        return "", ""

    parts = re.split(r'(?i)exclusion\s+criteria[:\s]*', criteria, maxsplit=1)
    if len(parts) == 2:
        inc = re.sub(r'(?i)inclusion\s+criteria[:\s]*', '', parts[0]).strip()
        exc = parts[1].strip()
        return inc, exc

    # fallback
    text = re.sub(r'(?i)inclusion\s+criteria[:\s]*', '', criteria).strip()
    return text, ""


def extract_criteria_lines(text: str) -> list[str]:
    if not text:
        return []

    lines = []
    for raw in text.splitlines():
        line = raw.strip().lstrip("-•*0123456789. )(").strip()
        if len(line) >= 3:
            lines.append(line)
    return lines


def extract_demographics(text: str) -> dict:
    demo = {"min_age": "N/A", "max_age": "N/A", "gender": "N/A"}
    if not text:
        return demo

    t = text.lower()

    # gender
    if "male" in t and "female" in t:
        demo["gender"] = "Both"
    elif "male" in t:
        demo["gender"] = "Male"
    elif "female" in t:
        demo["gender"] = "Female"

    # age patterns
    patterns = [
        (r'between\s+the\s+ages?\s+of\s+(\d+)\s+and\s+(\d+)', "between"),
        (r'age[sd]?\s+(\d+)\s*(?:to|-|and)\s*(\d+)', "between"),
        (r'between\s+(\d+)\s+and\s+(\d+)\s*year', "between"),
    ]
    for pat, kind in patterns:
        m = re.search(pat, text, re.I)
        if m:
            demo["min_age"] = m.group(1)
            demo["max_age"] = m.group(2)
            return demo

    m = re.search(r'(?:at least|>=|≥)\s*(\d+)\s*(?:years?|yrs?)', text, re.I)
    if m:
        demo["min_age"] = m.group(1)

    m = re.search(r'(?:up to|<=|≤|no older than)\s*(\d+)\s*(?:years?|yrs?)', text, re.I)
    if m:
        demo["max_age"] = m.group(1)

    m = re.search(r'(\d+)\s+years?\s+(?:or older|and older|and over|or above)', text, re.I)
    if m and demo["min_age"] == "N/A":
        demo["min_age"] = m.group(1)

    return demo


def parse_criterion(line: str) -> dict:
    result = {
        "criterion": line,
        "variable": "N/A",
        "operator": "N/A",
        "threshold": "N/A",
    }

    # between
    m = re.search(r'(.+?)\s+(?:between|from)\s+(\d+(?:\.\d+)?)\s*(?:and|to|-)\s*(\d+(?:\.\d+)?)', line, re.I)
    if m:
        result["variable"] = m.group(1).strip(" ,:;-")
        result["operator"] = "between"
        result["threshold"] = f"{m.group(2)}-{m.group(3)}"
        return result

    # >= <= > <
    m = re.search(r'(.+?)\s*(>=|≤|<=|≥|>|<)\s*(\d+(?:\.\d+)?)', line, re.I)
    if m:
        result["variable"] = m.group(1).strip(" ,:;-")
        result["operator"] = m.group(2)
        result["threshold"] = m.group(3)
        return result

    # textual comparisons
    text_patterns = [
        (r'(.+?)\s+(?:at least|minimum of)\s+(\d+(?:\.\d+)?)', ">="),
        (r'(.+?)\s+(?:no more than|maximum of|up to)\s+(\d+(?:\.\d+)?)', "<="),
        (r'(.+?)\s+(?:less than|below|under)\s+(\d+(?:\.\d+)?)', "<"),
        (r'(.+?)\s+(?:greater than|more than|above)\s+(\d+(?:\.\d+)?)', ">"),
    ]
    for pat, op in text_patterns:
        m = re.search(pat, line, re.I)
        if m:
            result["variable"] = m.group(1).strip(" ,:;-")
            result["operator"] = op
            result["threshold"] = m.group(2)
            return result

    return result


def parse_study(study: dict) -> dict | None:
    try:
        module = study.get("protocolSection", {})
        nct_id = module.get("identificationModule", {}).get("nctId", "UNKNOWN")
        criteria = module.get("eligibilityModule", {}).get("eligibilityCriteria", "")
        condition = module.get("conditionsModule", {}).get("conditions", ["N/A"])[0]

        inc_text, exc_text = split_inclusion_exclusion(criteria)
        inc_lines = extract_criteria_lines(inc_text)
        exc_lines = extract_criteria_lines(exc_text)

        return {
            "nct_id": nct_id,
            "condition": condition,
            "demographics": extract_demographics(criteria),
            "inclusion_criteria": [parse_criterion(l) for l in inc_lines],
            "exclusion_criteria": [parse_criterion(l) for l in exc_lines],
        }
    except Exception as e:
        print(f"  Parse error: {e}")
        return None


def run_parse(max_studies: int | None = None) -> None:
    checkpoint_path = OUTPUT_PATH + ".done"

    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)

    if max_studies is not None:
        studies = studies[:max_studies]

    done_ids = set()
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as f:
            done_ids = {line.strip() for line in f if line.strip()}
        print(f"Resuming — {len(done_ids)} already done, {len(studies) - len(done_ids)} remaining.")

    pending = [
        s for s in studies
        if s.get("protocolSection", {}).get("identificationModule", {}).get("nctId", "") not in done_ids
    ]

    out_mode = "a" if done_ids else "w"
    success = len(done_ids)
    failed = []

    with open(OUTPUT_PATH, out_mode, encoding="utf-8") as out, \
         open(checkpoint_path, "a", encoding="utf-8") as ckpt:

        for i, study in enumerate(pending, start=1):
            nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId", f"#{i}")
            print(f"[{success + 1}/{len(studies)}] Parsing {nct_id}...", end=" ")

            parsed = parse_study(study)
            if parsed:
                out.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                out.flush()
                ckpt.write(nct_id + "\n")
                ckpt.flush()
                success += 1
                print("✓")
            else:
                failed.append(nct_id)
                print("✗")

    print(f"\nParsing complete: {success}/{len(studies)} succeeded")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join(failed[:10])}")


if __name__ == "__main__":
    run_parse(max_studies=500)