import json, os
from collections import defaultdict

INPUT_PATH  = "data/dialogues/dialogues.jsonl"
OUTPUT_PATH = "data/dialogues/validated.jsonl"


# ============================================================
# Validation checks (mirrors paper Phase IV logic)
# ============================================================

def check_role_reversal(dialogue: list[dict]) -> bool:
    """
    Reject if patient turn contains clinical/technical keywords
    that should only appear in recruiter speech (persona hallucination).
    """
    clinical_keywords = [
        "eligibility criteria", "inclusion", "exclusion",
        "NCT", "protocol", "clinical trial registry",
        "randomized", "placebo", "IRB"
    ]
    for turn in dialogue:
        if turn["role"] == "patient":
            text_lower = turn["text"].lower()
            if any(kw.lower() in text_lower for kw in clinical_keywords):
                return False
    return True


def check_internal_consistency(record: dict) -> bool:
    """
    Reject if patient profile values contradict the dialogue content.
    Checks age mentioned in dialogue matches patient_profile.
    """
    profile = record.get("patient_profile", {})
    age     = profile.get("age")
    if age is None:
        return True

    for turn in record.get("dialogue", []):
        if turn["role"] == "patient" and str(age) in turn["text"]:
            return True  # Age mentioned correctly at least once

    # If no age appears in patient turns at all, flag as inconsistent
    age_mentioned = any(
        str(age) in t["text"]
        for t in record["dialogue"]
        if t["role"] == "patient"
    )
    return age_mentioned


def check_dialogue_length(dialogue: list[dict]) -> bool:
    """
    Reject if dialogue is too short (degenerate) or has no patient turns.
    Paper minimum: enough turns to contain all atomic units.
    """
    patient_turns = [t for t in dialogue if t["role"] == "patient"]
    return len(patient_turns) >= 1 and len(dialogue) >= 3


def check_label_logic(record: dict) -> bool:
    """
    For NEGATIVE_HARD: must have a violated_unit.
    For POSITIVE: violated_unit must be None.
    """
    label   = record.get("label")
    violated = record.get("violated_unit")

    if label == "NEGATIVE_HARD" and violated is None:
        return False
    if label == "POSITIVE" and violated is not None:
        return False
    return True


# ============================================================
# Main validation pipeline
# ============================================================

def run_validation():
    with open(INPUT_PATH, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    validated  = []
    rejected   = []
    fail_stats = defaultdict(int)

    for record in records:
        dialogue = record.get("dialogue", [])
        failures = []

        if not check_role_reversal(dialogue):
            failures.append("role_reversal")

        if not check_internal_consistency(record):
            failures.append("internal_inconsistency")

        if not check_dialogue_length(dialogue):
            failures.append("dialogue_degeneracy")

        if not check_label_logic(record):
            failures.append("label_logic_error")

        if failures:
            for f in failures:
                fail_stats[f] += 1
            rejected.append({**record, "rejection_reasons": failures})
        else:
            validated.append(record)

    # Save validated output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for r in validated:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    total      = len(records)
    n_valid    = len(validated)
    n_rejected = len(rejected)

    print(f"Validation complete")
    print(f"  Total input      : {total}")
    print(f"  Passed           : {n_valid}  ({n_valid/total*100:.1f}%)")
    print(f"  Rejected         : {n_rejected}  ({n_rejected/total*100:.1f}%)")
    print(f"\nRejection breakdown:")
    for reason, count in sorted(fail_stats.items(), key=lambda x: -x[1]):
        print(f"  {reason:<30}: {count}")
    print(f"\nValidated output saved to: {OUTPUT_PATH}")

    # Label distribution
    pos  = sum(1 for r in validated if r["label"] == "POSITIVE")
    neg  = sum(1 for r in validated if r["label"] == "NEGATIVE_HARD")
    print(f"\nLabel distribution (validated):")
    print(f"  POSITIVE      : {pos}")
    print(f"  NEGATIVE_HARD : {neg}")

    # Persona distribution
    persona_counts = defaultdict(int)
    for r in validated:
        persona_counts[r["persona"]] += 1
    print(f"\nPersona distribution:")
    for persona, count in sorted(persona_counts.items()):
        print(f"  {persona:<15}: {count}")

    # Print one validated sample
    if validated:
        ex = validated[0]
        print(f"\n=== Sample Validated Record ===")
        print(f"NCT ID   : {ex['nct_id']}")
        print(f"Condition: {ex['condition']}")
        print(f"Label    : {ex['label']}")
        print(f"Persona  : {ex['persona']}")
        print(f"Violated : {ex['violated_unit']}")
        print(f"Turns    : {ex['turn_count']}")
        for turn in ex["dialogue"]:
            role = turn["role"].upper().ljust(10)
            print(f"  [{role}] {turn['text']}")


if __name__ == "__main__":
    run_validation()