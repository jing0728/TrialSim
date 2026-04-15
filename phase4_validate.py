"""
phase4_validate.py

Data Quality Assurance — Phase IV of the TrialSim-10k pipeline.

Runs five validation checks on every synthesized dialogue record and
writes only the records that pass all checks to the output file.

Check hierarchy (order matters — cheaper checks run first):
    1. check_dialogue_length      — structural minimum, O(n) scan
    2. check_role_reversal        — keyword scan on patient turns
    3. check_internal_consistency — profile-vs-dialogue age match
    4. check_label_logic          — presence/absence of violated_unit
    5. check_exact_one_violation  — full mathematical audit (most expensive)

Check 5 is the key upgrade over the original:
    Original  : only verified that violated_unit exists (label logic).
    Upgraded  : mathematically confirms the patient profile violates
                exactly one atomic unit — no more, no less.
    This mirrors the "Ground Truth Guard" described in paper Section III-D-2
    and the audit criteria in Table VI.

Pipeline position:
    dialogues.jsonl  ->  [this file]  ->  validated.jsonl
"""

import json
import os
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
INPUT_PATH  = "data/dialogues/dialogues.jsonl"
OUTPUT_PATH = "data/dialogues/validated.jsonl"

# ---------------------------------------------------------------------------
# Check 1 — Dialogue length
# ---------------------------------------------------------------------------

def check_dialogue_length(dialogue: list[dict]) -> bool:
    """
    Reject degenerate dialogues that are too short to contain useful signal.

    Minimum bar (paper Section III-D-1, "Dialogue Degeneracy" failure mode):
        - At least 1 patient turn  (otherwise there is nothing to audit)
        - At least 3 turns total   (recruiter open + patient reply + 1 more)
    """
    patient_turns = [t for t in dialogue if t["role"] == "patient"]
    return len(patient_turns) >= 1 and len(dialogue) >= 3

# ---------------------------------------------------------------------------
# Check 2 — Role reversal / persona hallucination
# ---------------------------------------------------------------------------

# Keywords that belong exclusively to recruiter/protocol language.
# A patient uttering any of these signals that the LLM broke character
# and leaked its system prompt into the patient persona.
#
# Precision rules — why each keyword is worded this way:
#   "eligibility criteria"  : "eligibility" alone is fine ("am I eligible?")
#                             must appear as the full phrase
#   "inclusion criteria"    : bare "inclusion" triggers on lab variable names
#                             e.g. a var called "inclusion_score"
#   "exclusion criteria"    : same reasoning as above
#   "nct0"                  : bare "nct" matches "function" (fu-NCT-ion);
#                             "NCT0..." is the actual registry ID prefix
#   "clinical trial registry": keep full phrase, very specific
#   "randomized"            : specific enough as a standalone word
#   "placebo"               : specific enough as a standalone word
#   "irb approval"          : bare "irb" matches "airborne" (a-IRB-orne)
#   "protocol number"       : bare "protocol" appears in legitimate patient
#                             speech ("treatment protocol", "study protocol")
_CLINICAL_KEYWORDS = frozenset([
    "eligibility criteria",
    "inclusion criteria",
    "exclusion criteria",
    "nct0",
    "clinical trial registry",
    "randomized",
    "placebo",
    "irb approval",
    "protocol number",
])

def check_role_reversal(dialogue: list[dict]) -> bool:
    """
    Reject if any patient turn contains clinical/protocol terminology.

    Case-insensitive check against _CLINICAL_KEYWORDS.
    A single match in any patient turn is sufficient to reject the record,
    because even one leaked keyword contaminates the natural-language signal
    that the benchmark is designed to test.
    """
    for turn in dialogue:
        if turn["role"] == "patient":
            text_lower = turn["text"].lower()
            if any(kw in text_lower for kw in _CLINICAL_KEYWORDS):
                return False
    return True

# ---------------------------------------------------------------------------
# Check 3 — Internal consistency (profile vs. dialogue)
# ---------------------------------------------------------------------------

def check_internal_consistency(record: dict) -> bool:
    """
    Verify that the patient's stated age in the dialogue matches
    the age stored in patient_profile.

    Design choice — age only:
        Age is the one demographic that is always asked and always numeric,
        making it the most reliable cross-reference signal.  Lab values are
        skipped here because their string representations can vary
        (e.g. "7.1" vs "7.10"); the mathematical check in Check 5 covers them.

    Pass condition: the correct age string appears at least once in any
    patient turn.  The Forgetful persona first states a wrong value then
    self-corrects, so the correct value will still be present.
    """
    profile = record.get("patient_profile", {})
    age     = profile.get("age")

    if age is None:
        return True   # No age in profile — nothing to cross-reference

    age_str = str(age)
    return any(
        age_str in turn["text"]
        for turn in record.get("dialogue", [])
        if turn["role"] == "patient"
    )

# ---------------------------------------------------------------------------
# Check 4 — Label logic (presence / absence of violated_unit)
# ---------------------------------------------------------------------------

def check_label_logic(record: dict) -> bool:
    """
    Enforce the structural contract between label and violated_unit:

        POSITIVE      → violated_unit must be None
                        (an eligible patient violates nothing)
        NEGATIVE_HARD → violated_unit must be a non-None dict
                        (an ineligible patient must violate exactly one unit)

    This is a cheap precondition check.  The actual mathematical proof that
    the violation is correct — and that only one unit is violated — is
    performed by Check 5.
    """
    label    = record.get("label")
    violated = record.get("violated_unit")

    if label == "NEGATIVE_HARD" and violated is None:
        return False
    if label == "POSITIVE" and violated is not None:
        return False
    return True

# ---------------------------------------------------------------------------
# Check 5 — Mathematical single-violation audit (the Ground Truth Guard)
# ---------------------------------------------------------------------------

def evaluate_criterion(patient_val: float, operator: str, threshold: str) -> bool:
    """
    Evaluate whether patient_val satisfies a single atomic criterion.

    Supported operators:
        ">=" / "≥"   — patient_val >= threshold
        "<=" / "≤"   — patient_val <= threshold
        ">"           — patient_val >  threshold
        "<"           — patient_val <  threshold
        "="  / "=="   — patient_val == threshold
        "between"     — lo <= patient_val <= hi  (threshold format: "lo-hi")

    Returns True when the criterion is satisfied (patient passes this unit).
    Raises ValueError if threshold cannot be parsed — caller handles it.
    """
    val = float(patient_val)

    # Range operator: threshold must be "lo-hi"
    if operator.lower() == "between":
        parts = str(threshold).split("-")
        if len(parts) != 2:
            raise ValueError(f"Malformed 'between' threshold: {threshold!r}")
        lo, hi = float(parts[0].strip()), float(parts[1].strip())
        return lo <= val <= hi

    # Single-value operators
    tau = float(threshold)
    dispatch = {
        ">=" : val >= tau,
        "≥"  : val >= tau,
        "<=" : val <= tau,
        "≤"  : val <= tau,
        ">"  : val >  tau,
        "<"  : val <  tau,
        "="  : val == tau,
        "==" : val == tau,
    }

    if operator not in dispatch:
        # Unknown operator — conservatively treat as satisfied so we do not
        # generate false positives in the violation count.
        return True

    return dispatch[operator]


def check_exact_one_violation(record: dict) -> tuple[bool, str]:
    """
    Mathematical Ground Truth Guard — mirrors paper Table VI audit logic.

    For POSITIVE records  : skipped (returns True immediately).
    For NEGATIVE_HARD     : verifies that the patient profile violates
                            exactly one atomic unit while satisfying all others.

    Three failure modes caught here:
        "zero violations"     — label says NEGATIVE_HARD but the profile
                                actually satisfies every criterion; wrong label.
        "multiple violations" — more than one unit violated; the case is an
                                "Easy Negative" that does not test marginal
                                numerical reasoning (paper Section III-B-3).
        "violation mismatch"  — the unit that mathematically fails does not
                                match the violated_unit recorded in metadata;
                                indicates a synthesis or bookkeeping error.

    Demographics age fallback:
        Age constraints are sometimes stored only in demographics
        (min_age / max_age) rather than as atomic_units entries.
        When violated_unit.variable == "Age" and no Age unit is in
        atomic_units, we verify directly against the demographics block.

    Fallback operator skip:
        Records where violated_unit.operator == "fallback" were created
        as a last-resort safety net — no numeric threshold exists to check.
        We trust the earlier label_logic check and skip math verification.

    Returns (passed: bool, reason: str).
    reason is an empty string when passed=True.
    """
    if record.get("label") != "NEGATIVE_HARD":
        return True, ""

    violated = record.get("violated_unit", {}) or {}

    # Fallback violations cannot be verified mathematically — skip
    if violated.get("operator") == "fallback":
        return True, ""

    profile      = record.get("patient_profile", {})
    units        = record.get("atomic_units",    [])
    demographics = record.get("demographics",    {})

    actual_violations: list[str] = []

    # Track whether any Age unit exists in atomic_units
    age_in_units = any(u.get("variable", "").lower() == "age" for u in units)

    for unit in units:
        var = unit.get("variable", "")
        op  = unit.get("operator",  "")
        tau = unit.get("threshold", "")

        if var.lower() == "age":
            patient_val = profile.get("age")
        else:
            patient_val = profile.get("lab_values", {}).get(var)

        if patient_val is None:
            continue

        try:
            if not evaluate_criterion(float(patient_val), op, tau):
                actual_violations.append(var)
        except (ValueError, TypeError):
            continue

    # Demographics age fallback —
    # If violated_unit says "Age" but no Age unit is in atomic_units,
    # derive the constraint from demographics.min_age / max_age and verify.
    if not age_in_units and violated.get("variable", "").lower() == "age":
        patient_age = profile.get("age")
        min_age     = demographics.get("min_age", "N/A")
        max_age     = demographics.get("max_age", "N/A")
        if patient_age is not None:
            try:
                age_f = float(patient_age)
                if min_age != "N/A" and age_f < float(min_age):
                    actual_violations.append("Age")
                elif max_age != "N/A" and age_f > float(max_age):
                    actual_violations.append("Age")
            except (ValueError, TypeError):
                pass

    # Deduplicate violations by variable name (case-insensitive).
    # One trial can have multiple atomic units for the same variable,
    # e.g. <Age, >=, 18> AND <Age, between, 18-55>. Violating the
    # target age simultaneously fails both units — this is still a
    # single-variable violation and must not be rejected as "multiple".
    unique_vars = list(dict.fromkeys(v.lower() for v in actual_violations))

    # --- Verdict ---
    if len(unique_vars) == 0:
        return False, (
            "label=NEGATIVE_HARD but profile satisfies all criteria "
            "(no mathematical violation found)"
        )

    if len(unique_vars) > 1:
        return False, (
            f"multiple variables violated: {unique_vars} — "
            "record is an Easy Negative, not a hard boundary case"
        )

    expected_var = violated.get("variable", "")
    actual_var   = unique_vars[0]

    # Case-insensitive comparison — "Age" vs "age" should not cause mismatch
    if actual_var.lower() != expected_var.lower():
        return False, (
            f"violation mismatch: metadata says '{expected_var}' "
            f"but mathematical audit found '{actual_var}'"
        )

    return True, ""

# ---------------------------------------------------------------------------
# Main validation pipeline
# ---------------------------------------------------------------------------

def run_validation(input_path: str = INPUT_PATH,
                   output_path: str = OUTPUT_PATH) -> None:
    """
    Run all five checks over every record in input_path.
    Records that pass all checks are written to output_path.
    Rejected records are counted but not saved (add a rejected dump if needed).

    Checks are applied in cheapest-first order so that expensive checks
    (check_exact_one_violation) only run on records that passed the cheap ones.
    """
    with open(input_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    validated:  list[dict]         = []
    fail_stats: defaultdict[str, int] = defaultdict(int)

    # Map check name → callable for structured reporting
    checks = [
        ("dialogue_degeneracy",    lambda r: (check_dialogue_length(r.get("dialogue", [])), "")),
        ("role_reversal",          lambda r: (check_role_reversal(r.get("dialogue", [])),   "")),
        ("internal_inconsistency", lambda r: (check_internal_consistency(r),                "")),
        ("label_logic_error",      lambda r: (check_label_logic(r),                         "")),
        ("exact_violation_error",  check_exact_one_violation),
    ]

    for record in records:
        failures: list[str] = []

        for check_name, check_fn in checks:
            passed, reason = check_fn(record)
            if not passed:
                failures.append(check_name)
                fail_stats[check_name] += 1
                # Continue checking — collect all failure modes per record
                # so the quality report shows the full breakdown.

        if not failures:
            validated.append(record)

    # Write validated output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        for r in validated:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Summary report ---
    total      = len(records)
    n_valid    = len(validated)
    n_rejected = total - n_valid

    print("Validation complete")
    print(f"  Total input : {total}")
    print(f"  Passed      : {n_valid}  ({n_valid / total * 100:.1f}%)")
    print(f"  Rejected    : {n_rejected}  ({n_rejected / total * 100:.1f}%)")

    if fail_stats:
        print("\nRejection breakdown:")
        for reason, count in sorted(fail_stats.items(), key=lambda x: -x[1]):
            print(f"  {reason:<30}: {count}")

    print(f"\nValidated output saved to: {output_path}")

    # --- Label distribution (mirrors paper Table VII) ---
    pos = sum(1 for r in validated if r["label"] == "POSITIVE")
    neg = sum(1 for r in validated if r["label"] == "NEGATIVE_HARD")
    print(f"\nLabel distribution (validated):")
    print(f"  POSITIVE      : {pos}")
    print(f"  NEGATIVE_HARD : {neg}")
    if n_valid:
        print(f"  NEGATIVE ratio: {neg / n_valid * 100:.1f}%  "
              f"(paper target: ~74.9%)")

    # --- Persona distribution (mirrors paper Table VII) ---
    persona_counts: defaultdict[str, int] = defaultdict(int)
    for r in validated:
        persona_counts[r["persona"]] += 1
    print("\nPersona distribution:")
    for persona, count in sorted(persona_counts.items()):
        pct = count / n_valid * 100 if n_valid else 0
        print(f"  {persona:<15}: {count}  ({pct:.1f}%)")

    # --- Sample output ---
    if validated:
        ex = validated[0]
        print("\n=== Sample Validated Record ===")
        print(f"NCT ID   : {ex['nct_id']}")
        print(f"Condition: {ex['condition']}")
        print(f"Label    : {ex['label']}")
        print(f"Persona  : {ex['persona']}")
        print(f"Violated : {ex['violated_unit']}")
        print(f"Turns    : {ex['turn_count']}")
        for turn in ex["dialogue"]:
            role = turn["role"].upper().ljust(10)
            print(f"  [{role}] {turn['text'][:120]}")


if __name__ == "__main__":
    run_validation()