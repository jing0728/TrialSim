import json, random, os

INPUT_PATH  = "data/raw/atomic_units.jsonl"
OUTPUT_PATH = "data/dialogues/dialogues.jsonl"
os.makedirs("data/dialogues", exist_ok=True)

# ============================================================
# Five personas from the paper
# ============================================================

PERSONAS = {
    "COOPERATIVE": {
        "description": "Answers concisely and directly.",
        "age_template":    "I am {age} years old.",
        "gender_template": "I am {gender}.",
        "lab_template":    "My {variable} is {value}.",
    },
    "CHATTY": {
        "description": "Buries facts in long narratives.",
        "age_template":    "Well, you know, my birthday was just last month — I just turned {age}, can you believe it?",
        "gender_template": "I am {gender}, by the way, my whole family came to visit recently.",
        "lab_template":    "Oh, my doctor mentioned something about {variable}, I think it was around {value} or so, he said it was fine.",
    },
    "FORGETFUL": {
        "description": "Initially uncertain, then self-corrects.",
        "age_template":    "I think I am... maybe {age_wrong}? No wait, sorry, I just turned {age} actually.",
        "gender_template": "I am {gender}.",
        "lab_template":    "Hmm, I am not sure about my {variable}... I think it was {value_wrong}? Actually no, my last result was {value}.",
    },
    "ANXIOUS": {
        "description": "Asks questions, needs reassurance.",
        "age_template":    "I am {age} years old. Is that going to be a problem?",
        "gender_template": "I am {gender}. Does that affect my eligibility?",
        "lab_template":    "My {variable} was {value}. Is that value okay for this trial?",
    },
    "RELUCTANT": {
        "description": "Gives minimal answers, needs prompting.",
        "age_template":    "{age}.",
        "gender_template": "{gender}.",
        "lab_template":    "{value}.",
    },
}

# ============================================================
# Recruiter question templates
# ============================================================

RECRUITER_QUESTIONS = {
    "age":    "Could you please tell me your age?",
    "gender": "Could you confirm your biological sex or gender?",
    "lab":    "Do you know your most recent {variable} result?",
}


# ============================================================
# FIX 2: Improved criteria detection
# ============================================================

def detect_real_criteria(trial: dict) -> bool:
    """
    Determine whether this trial has enough verifiable thresholds to
    generate a NEGATIVE_HARD case.

    Original logic only checked demographics.min_age / max_age and
    non-age atomic_units.  This missed trials where age constraints
    live inside atomic_units (variable = "Age" or "age") rather than
    the demographics block — a common result of the LLM parser.

    Extended checks (in order):
        1. demographics.min_age or max_age is set         (original)
        2. any non-age atomic_unit has a numeric threshold (original)
        3. any atomic_unit with variable="age" has a
           numeric threshold                               (NEW — fixes gap)
    """
    demo = trial.get("demographics", {})

    # Check 1: age range in demographics block
    has_demo_age = (
        demo.get("min_age", "N/A") != "N/A"
        or demo.get("max_age", "N/A") != "N/A"
    )
    if has_demo_age:
        return True

    units = trial.get("atomic_units", [])

    # Check 2: non-age lab/numeric unit with a parseable threshold
    has_lab = any(
        u.get("threshold", "N/A") != "N/A"
        for u in units
        if u.get("variable", "").lower() != "age"
    )
    if has_lab:
        return True

    # Check 3 (NEW): age constraint stored as an atomic_unit instead of
    # in the demographics block — common when the LLM parser is used
    has_unit_age = any(
        u.get("threshold", "N/A") != "N/A"
        for u in units
        if u.get("variable", "").lower() == "age"
    )
    return has_unit_age


# ============================================================
# Patient profile generator
# ============================================================

def generate_patient_profile(trial: dict, label: str) -> tuple:
    """
    Generate a virtual patient profile.
    label = 'POSITIVE' (meets all criteria) or 'NEGATIVE_HARD' (violates exactly one unit).

    Age sourcing order (handles both parser outputs):
        1. demographics.min_age / max_age   (rule-based parser output)
        2. atomic_units where variable="Age" (LLM parser output)   <-- NEW fallback
    Returns (profile dict, violated_unit dict or None).
    """
    demo     = trial.get("demographics", {})
    units    = trial.get("atomic_units", [])
    profile  = {}
    violated = None

    # --- Demographics: Age ---
    min_age = demo.get("min_age", "N/A")
    max_age = demo.get("max_age", "N/A")

    # FIX: if demographics block has no age, fall back to atomic_units
    if min_age == "N/A" and max_age == "N/A":
        for u in units:
            if u.get("variable", "").lower() == "age" and u.get("threshold", "N/A") != "N/A":
                op  = u.get("operator", "")
                tau = u.get("threshold", "")
                # Map atomic unit operator back to min/max age fields
                if op in [">=", "≥", "between"]:
                    try:
                        if op == "between" and "-" in str(tau):
                            parts   = tau.split("-")
                            min_age = parts[0].strip()
                            max_age = parts[1].strip()
                        else:
                            min_age = str(int(float(tau)))
                    except (ValueError, TypeError):
                        pass
                elif op in ["<=", "≤"]:
                    try:
                        max_age = str(int(float(tau)))
                    except (ValueError, TypeError):
                        pass
                break   # Only use the first age unit found

    if min_age != "N/A" and max_age != "N/A":
        try:
            lo = int(min_age)
            hi = int(max_age)
            if label == "POSITIVE":
                profile["age"] = random.randint(lo, hi)
            else:
                profile["age"] = lo - 1 if random.random() < 0.5 else hi + 1
                violated = {
                    "variable":      "Age",
                    "operator":      "between",
                    "threshold":     f"{lo}-{hi}",
                    "patient_value": profile["age"]
                }
        except ValueError:
            profile["age"] = 30
    elif min_age != "N/A":
        try:
            lo = int(min_age)
            if label == "POSITIVE":
                profile["age"] = random.randint(lo, lo + 30)
            else:
                profile["age"] = lo - 1
                violated = {
                    "variable":      "Age",
                    "operator":      ">=",
                    "threshold":     str(lo),
                    "patient_value": profile["age"]
                }
        except ValueError:
            profile["age"] = 30
    elif max_age != "N/A":
        try:
            hi = int(max_age)
            if label == "POSITIVE":
                profile["age"] = random.randint(max(1, hi - 20), hi)
            else:
                profile["age"] = hi + 1
                violated = {
                    "variable":      "Age",
                    "operator":      "<=",
                    "threshold":     str(hi),
                    "patient_value": profile["age"]
                }
        except ValueError:
            profile["age"] = 30
    else:
        profile["age"] = 30

    # --- Demographics: Gender ---
    gender = demo.get("gender", "Both")
    if gender == "Male":
        profile["gender"] = "Male"
    elif gender == "Female":
        profile["gender"] = "Female"
    else:
        profile["gender"] = random.choice(["Male", "Female"])

    # --- Lab / numeric units ---
    lab_values = {}
    for unit in units:
        var = unit["variable"]
        op  = unit["operator"]
        tau = unit["threshold"]

        if var.lower() == "age":
            continue

        try:
            if op == "between" and "-" in str(tau):
                lo_s, hi_s = tau.split("-")
                lo_v, hi_v = float(lo_s), float(hi_s)
                if label == "POSITIVE" or violated is not None:
                    lab_values[var] = round(random.uniform(lo_v, hi_v), 1)
                else:
                    val = lo_v - 1 if random.random() < 0.5 else hi_v + 1
                    lab_values[var] = round(val, 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

            elif op in [">=", "≥"]:
                thresh = float(tau)
                if label == "POSITIVE" or violated is not None:
                    lab_values[var] = round(thresh + random.uniform(0.1, 5), 1)
                else:
                    lab_values[var] = round(thresh - random.uniform(0.1, 2), 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

            elif op in ["<=", "≤"]:
                thresh = float(tau)
                if label == "POSITIVE" or violated is not None:
                    lab_values[var] = round(thresh - random.uniform(0.1, 5), 1)
                else:
                    lab_values[var] = round(thresh + random.uniform(0.1, 2), 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

            elif op == "<":
                thresh = float(tau)
                if label == "POSITIVE" or violated is not None:
                    lab_values[var] = round(thresh - random.uniform(0.1, 5), 1)
                else:
                    lab_values[var] = round(thresh + random.uniform(0.1, 2), 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

            elif op == ">":
                thresh = float(tau)
                if label == "POSITIVE" or violated is not None:
                    lab_values[var] = round(thresh + random.uniform(0.1, 5), 1)
                else:
                    lab_values[var] = round(thresh - random.uniform(0.1, 2), 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

        except (ValueError, TypeError):
            continue

    profile["lab_values"] = lab_values
    return profile, violated


# ============================================================
# Dialogue generator
# ============================================================

def generate_dialogue(trial: dict, profile: dict, persona_name: str) -> list:
    """
    Generate a multi-turn recruiter/patient dialogue.
    Returns a list of {"role": "recruiter"/"patient", "text": "..."} turns.
    """
    persona = PERSONAS[persona_name]
    turns   = []

    # Opening
    turns.append({
        "role": "recruiter",
        "text": (f"Hello, thank you for your interest in this clinical trial for "
                 f"{trial.get('condition', 'the condition')}. "
                 f"I need to ask you a few questions to check your eligibility.")
    })

    # Age
    turns.append({"role": "recruiter", "text": RECRUITER_QUESTIONS["age"]})
    age = profile.get("age", 30)
    if persona_name == "FORGETFUL":
        wrong_age = age + random.choice([-2, 2, 3])
        response  = persona["age_template"].format(age=age, age_wrong=wrong_age)
    else:
        response = persona["age_template"].format(age=age)
    turns.append({"role": "patient", "text": response})

    # Gender
    turns.append({"role": "recruiter", "text": RECRUITER_QUESTIONS["gender"]})
    gender   = profile.get("gender", "N/A")
    response = persona["gender_template"].format(gender=gender)
    turns.append({"role": "patient", "text": response})

    # Lab values
    for var, val in profile.get("lab_values", {}).items():
        q = RECRUITER_QUESTIONS["lab"].format(variable=var)
        turns.append({"role": "recruiter", "text": q})
        if persona_name == "FORGETFUL":
            wrong_val = round(val + random.choice([-1, 1, 0.5]), 1)
            response  = persona["lab_template"].format(
                variable=var, value=val, value_wrong=wrong_val)
        else:
            response = persona["lab_template"].format(variable=var, value=val)
        turns.append({"role": "patient", "text": response})

    return turns


# ============================================================
# SCT Protocol: strip recruiter conclusion turns
# ============================================================

def apply_sct(turns: list) -> list:
    """
    Strict Conclusion Truncation:
    Keep only up to the last patient disclosure turn.
    Removes any recruiter summary or eligibility hint at the end.
    """
    last_patient_idx = -1
    for i, turn in enumerate(turns):
        if turn["role"] == "patient":
            last_patient_idx = i
    if last_patient_idx == -1:
        return turns
    return turns[:last_patient_idx + 1]


# ============================================================
# Main pipeline
# ============================================================

def run_dialogue_synthesis():
    with open(INPUT_PATH, encoding="utf-8") as f:
        trials = [json.loads(line) for line in f]

    all_dialogues = []
    persona_names = list(PERSONAS.keys())
    skipped_neg   = 0

    for trial in trials:
        # FIX 2: use improved detection that also checks atomic_units age fields
        has_real_criteria = detect_real_criteria(trial)

        labels = ["POSITIVE", "NEGATIVE_HARD"] if has_real_criteria else ["POSITIVE"]
        if not has_real_criteria:
            skipped_neg += 1

        for label in labels:
            persona_name      = random.choice(persona_names)
            profile, violated = generate_patient_profile(trial, label)
            turns             = generate_dialogue(trial, profile, persona_name)
            turns_sct         = apply_sct(turns)

            record = {
                "nct_id":          trial.get("nct_id"),
                "condition":       trial.get("condition"),
                "label":           label,
                "persona":         persona_name,
                "patient_profile": profile,
                "violated_unit":   violated,
                "dialogue":        turns_sct,
                "turn_count":      len(turns_sct),
                # FIX 1: carry atomic_units forward so quality_report.py
                # can compute Section 4 (atomic unit density)
                "atomic_units":    trial.get("atomic_units", []),
            }
            all_dialogues.append(record)

    # Save output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for d in all_dialogues:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Summary
    total     = len(all_dialogues)
    positive  = sum(1 for d in all_dialogues if d["label"] == "POSITIVE")
    negative  = sum(1 for d in all_dialogues if d["label"] == "NEGATIVE_HARD")
    avg_turns = sum(d["turn_count"] for d in all_dialogues) / total

    print(f"Dialogue synthesis complete")
    print(f"  Total dialogues        : {total}")
    print(f"  POSITIVE               : {positive}  ({positive/total*100:.1f}%)")
    print(f"  NEGATIVE_HARD          : {negative}  ({negative/total*100:.1f}%)")
    print(f"  Skipped (no threshold) : {skipped_neg}")
    print(f"  Avg turns (SCT)        : {avg_turns:.1f}")
    print(f"  Output saved to        : {OUTPUT_PATH}")

    # Sample
    ex = all_dialogues[0]
    print(f"\n=== Sample Dialogue ===")
    print(f"NCT ID        : {ex['nct_id']}")
    print(f"Condition     : {ex['condition']}")
    print(f"Label         : {ex['label']}")
    print(f"Persona       : {ex['persona']}")
    print(f"Violated      : {ex['violated_unit']}")
    print(f"Turns         : {ex['turn_count']}")
    print(f"Atomic units  : {len(ex['atomic_units'])}")
    print()
    for turn in ex["dialogue"]:
        role = turn["role"].upper().ljust(10)
        print(f"  [{role}] {turn['text']}")


if __name__ == "__main__":
    run_dialogue_synthesis()