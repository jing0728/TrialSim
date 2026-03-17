import json, os

INPUT_PATH  = "data/raw/raw_studies.json"
OUTPUT_PATH = "data/raw/filtered_studies.json"

def has_valid_criteria(study: dict) -> bool:
    """
    过滤条件（对应论文 Phase I Logical Pruning）：
    1. 必须有 eligibilityCriteria 字段
    2. 必须同时包含 inclusion 和 exclusion 关键词
    3. 长度不能太短（过滤掉无内容的占位符）
    """
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
    
    print(f"原始记录数: {len(studies)}")
    filtered = [s for s in studies if has_valid_criteria(s)]
    print(f"过滤后记录数: {len(filtered)}")
    print(f"保留率: {len(filtered)/len(studies)*100:.1f}%")
    
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)
    
    print(f"已保存到 {OUTPUT_PATH}")
    return filtered

if __name__ == "__main__":
    run_filter()