import json, os

INPUT_PATH  = "data/raw/raw_studies.json"
OUTPUT_PATH = "data/raw/filtered_studies.json"

def has_valid_criteria(study: dict) -> bool:
    try:
        criteria = (study
                    .get("protocolSection", {})
                    .get("eligibilityModule", {})
                    .get("eligibilityCriteria", ""))
    except AttributeError:
        return False
    
    if not criteria or len(criteria) < 100:
        return False
    
    c_lower = criteria.lower()
    has_inclusion = "inclusion" in c_lower
    has_exclusion = "exclusion" in c_lower
    return has_inclusion and has_exclusion

def run_filter():
    with open(INPUT_PATH, encoding="utf-8") as f:
        studies = json.load(f)
    
    print(f"Datas: {len(studies)}")
    filtered = [s for s in studies if has_valid_criteria(s)]
    print(f"After pre filter: {len(filtered)}")
    print(f"Saving rate: {len(filtered)/len(studies)*100:.1f}%")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    print(f"Save to  {OUTPUT_PATH}")
    return filtered

if __name__ == "__main__":
    run_filter()