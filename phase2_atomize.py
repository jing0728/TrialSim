import json

INPUT_PATH  = "data/raw/parsed_pico.jsonl"
OUTPUT_PATH = "data/raw/atomic_units.jsonl"


def extract_atomic_units(parsed: dict) -> dict:
    """
    Convert PICO parse results into atomic logic triplet format.
    Each unit: ai = <variable, operator, threshold>
    Also generates a Negative Hard violation scenario (Pivot Point +/- epsilon).
    """
    units = []

    for crit in parsed.get("inclusion_criteria", []):
        v   = crit.get("variable",  "N/A")
        op  = crit.get("operator",  "N/A")
        tau = crit.get("threshold", "N/A")

        if v == "N/A" or tau == "N/A":
            continue

        unit = {
            "variable":       v,
            "operator":       op,
            "threshold":      tau,
            "type":           "inclusion",
            "criterion_text": crit.get("criterion", "")
        }
        unit["violation_scenario"] = generate_violation(op, tau)
        units.append(unit)

    for crit in parsed.get("exclusion_criteria", []):
        v   = crit.get("variable",  "N/A")
        op  = crit.get("operator",  "N/A")
        tau = crit.get("threshold", "N/A")

        if v == "N/A" or tau == "N/A":
            continue

        unit = {
            "variable":       v,
            "operator":       op,
            "threshold":      tau,
            "type":           "exclusion",
            "criterion_text": crit.get("criterion", "")
        }
        unit["violation_scenario"] = generate_violation(op, tau)
        units.append(unit)

    return {
        "nct_id":       parsed.get("nct_id"),
        "condition":    parsed.get("condition"),
        "demographics": parsed.get("demographics"),
        "atomic_units": units
    }


def generate_violation(operator: str, threshold: str) -> str:
    """
    Generate a Negative Hard scenario: threshold +/- epsilon.

    Supported formats:
      - Single value:  ">=", "<=", ">", "<"  e.g. threshold = "18"
      - Range value:   "between"              e.g. threshold = "18-55"

    Examples:
      Age >= 18        -> violation = 17   (just below lower bound)
      Age between 18-55 -> violation = 17  (just below lower bound)
                          violation_high = 56 (just above upper bound)
    """
    try:
        # --- Range: "between" with threshold like "18-55" ---
        if operator.lower() == "between" and "-" in str(threshold):
            parts = threshold.split("-")
            low  = float(parts[0].strip())
            high = float(parts[1].strip())
            eps_low  = 1 if low  == int(low)  else 0.1
            eps_high = 1 if high == int(high) else 0.1
            # Two boundary violations: just below low, just above high
            return f"{low - eps_low} or {high + eps_high}"

        # --- Single value ---
        val     = float(threshold)
        epsilon = 1 if val == int(val) else 0.1

        if operator in [">=", "≥", "greater than or equal"]:
            return str(val - epsilon)   # Below lower bound -> ineligible
        elif operator in ["<=", "≤", "less than or equal"]:
            return str(val + epsilon)   # Above upper bound -> ineligible
        elif operator in [">", "greater than"]:
            return str(val)             # Exactly equal -> ineligible
        elif operator in ["<", "less than"]:
            return str(val)             # Exactly equal -> ineligible
        else:
            return "N/A"

    except (ValueError, TypeError):
        return "N/A"


def run_atomize():
    results = []

    with open(INPUT_PATH, encoding="utf-8") as f, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as out:

        for line in f:
            parsed   = json.loads(line.strip())
            atomized = extract_atomic_units(parsed)
            out.write(json.dumps(atomized, ensure_ascii=False) + "\n")
            results.append(atomized)

    print(f"Atomization complete: {len(results)} trials processed")
    print(f"Output saved to: {OUTPUT_PATH}")

    # Print a sample to verify output quality
    if results:
        print("\n=== Sample Output ===")
        ex = results[0]
        print(f"NCT ID   : {ex['nct_id']}")
        print(f"Condition: {ex['condition']}")
        print(f"Demographics: {ex['demographics']}")
        print(f"Total atomic units: {len(ex['atomic_units'])}")
        print()
        for u in ex["atomic_units"][:5]:
            print(f"  Atomic unit : <{u['variable']}, {u['operator']}, {u['threshold']}>")
            print(f"  Type        : {u['type']}")
            print(f"  Violation   : {u['violation_scenario']}")
            print()

    return results


if __name__ == "__main__":
    run_atomize()