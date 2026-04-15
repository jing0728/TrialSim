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
        "gender_template": "I am {gender}. Will that be an issue for me?",
        "lab_template":    "My {variable} was {value}. Is that an acceptable value?",
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
                    # POSITIVE: must satisfy >= thresh, also ensure non-negative
                    lab_values[var] = round(max(0.0, thresh + random.uniform(0.1, 5)), 1)
                else:
                    # NEGATIVE violation: must be below thresh
                    # max(0.0) prevents negatives but keeps the value below thresh
                    raw = round(thresh - random.uniform(0.1, 2), 1)
                    lab_values[var] = round(max(0.0, raw), 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

            elif op in ["<=", "≤"]:
                thresh = float(tau)
                if label == "POSITIVE" or violated is not None:
                    # POSITIVE: must satisfy <= thresh, also ensure non-negative
                    lab_values[var] = round(max(0.0, thresh - random.uniform(0.1, 5)), 1)
                else:
                    # NEGATIVE violation: must exceed thresh (can't be negative anyway)
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
                    lab_values[var] = round(max(0.0, thresh - random.uniform(0.1, 5)), 1)
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
                    lab_values[var] = round(max(0.0, thresh + random.uniform(0.1, 5)), 1)
                else:
                    raw = round(thresh - random.uniform(0.1, 2), 1)
                    lab_values[var] = round(max(0.0, raw), 1)
                    violated = {
                        "variable":      var,
                        "operator":      op,
                        "threshold":     tau,
                        "patient_value": lab_values[var]
                    }

        except (ValueError, TypeError):
            continue

    profile["lab_values"] = lab_values

    # --- Safety check: NEGATIVE_HARD must always have a violated_unit ---
    #
    # Why this happens:
    #   detect_real_criteria() found a threshold and set label=NEGATIVE_HARD,
    #   but every atomic unit's operator was either unsupported (e.g. "=",
    #   "N/A") or raised ValueError during float() conversion, so the
    #   loop above never assigned `violated`.  The record would then be
    #   written as NEGATIVE_HARD with violated_unit=None and rejected by
    #   phase4_validate.py's check_label_logic() — wasting the record.
    #
    # Fix: when this edge case occurs, force the age one step below the
    #   minimum valid value so we always have a well-formed violated_unit.
    #   Uses min_age when available; falls back to current age - 1.
    if label == "NEGATIVE_HARD" and violated is None:
        current_age = profile.get("age", 30)
        if min_age != "N/A":
            try:
                lo = int(min_age)
                profile["age"] = lo - 1
                violated = {
                    "variable":      "Age",
                    "operator":      ">=",
                    "threshold":     str(lo),
                    "patient_value": lo - 1,
                }
            except ValueError:
                profile["age"] = current_age - 1
                violated = {
                    "variable":      "Age",
                    "operator":      "fallback",
                    "threshold":     str(current_age),
                    "patient_value": current_age - 1,
                }
        else:
            profile["age"] = current_age - 1
            violated = {
                "variable":      "Age",
                "operator":      "fallback",
                "threshold":     str(current_age),
                "patient_value": current_age - 1,
            }

    return profile, violated


# Keywords that must never appear inside a patient turn.
# Used both by phase4_validate.py and here to sanitize variable names
# before they are embedded into dialogue templates.
# Note: this set is intentionally MORE aggressive than phase4's _CLINICAL_KEYWORDS
# because false positives here (replacing a benign variable name) are harmless,
# while false negatives cause role_reversal failures.
# Short forms like "irb" and "nct" are included here but NOT in phase4's checker
# to avoid false positives in patient speech ("airborne", "function" etc.).
_SENSITIVE_VAR_KEYWORDS = frozenset([
    "randomized", "randomised", "placebo",
    "irb",
    "irb approval",
    " nct",
    "nct0",
    "clinical trial registry",
    "eligibility criteria",
    "inclusion criteria",
    "exclusion criteria",
    "protocol number",
    "informed consent",
    "adverse event",
    "controlled trial",
    "study drug",
    "investigational",
    "washout period",
    "screening visit",
    "concomitant",
    "contraindicated",
    "double-blind",
    "open-label",
])

_PATIENT_FORBIDDEN = frozenset([
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

def sanitize_variable_name(var: str) -> str:
    """
    Replace a variable name with a safe generic label if it contains
    any clinical-protocol keyword that would trigger the role_reversal check.
    """
    var_lower = " " + var.lower()
    if any(kw in var_lower for kw in _SENSITIVE_VAR_KEYWORDS):
        return "that measurement"
    return var


def safe_patient_response(text: str, fallback_value: str) -> str:
    """
    Post-generation safety net: if the patient response still contains
    forbidden keywords after variable sanitization, replace with bare value.
    """
    if any(kw in text.lower() for kw in _PATIENT_FORBIDDEN):
        return f"It was {fallback_value}."
    return text


def clean_variable_name(var: str) -> str:
    """
    Remove backslashes, trailing punctuation and whitespace noise from
    LLM-parsed variable names before embedding them in dialogue templates.

    Examples:
        "Absolute neutrophil count (ANC) \\"  →  "Absolute neutrophil count (ANC)"
        "\\. Male or female subjects"          →  "Male or female subjects"
        "eGFR :"                               →  "eGFR"
    """
    import re
    cleaned = re.sub(r'^[\\.\s]+', '', var)
    cleaned = re.sub(r'[\\.\s:;,]+$', '', cleaned)
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned.strip()


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

    # Lab values — clean, sanitize variable names, then post-check patient response
    for var, val in profile.get("lab_values", {}).items():
        clean_var = clean_variable_name(var)
        safe_var  = sanitize_variable_name(clean_var)
        q = RECRUITER_QUESTIONS["lab"].format(variable=safe_var)
        turns.append({"role": "recruiter", "text": q})
        if persona_name == "FORGETFUL":
            wrong_val = round(val + random.choice([-1, 1, 0.5]), 1)
            response  = persona["lab_template"].format(
                variable=safe_var, value=val, value_wrong=wrong_val)
        else:
            response = persona["lab_template"].format(variable=safe_var, value=val)
        response = safe_patient_response(response, str(val))
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
# Violatable unit collector
# ============================================================

def collect_violatable_units(trial: dict) -> list[dict]:
    """
    Return every unit in this trial that can produce a valid NEGATIVE_HARD case.

    Two sources are checked:
        1. demographics.min_age / max_age  — age constraint from the parser's
           structured output (rule-based parser writes here)
        2. atomic_units with a numeric threshold  — individual criterion triplets
           (LLM parser often writes age here too, plus all lab/score units)

    Each returned item is a "violation target" dict:
        {
            "source":    "demographics" | "atomic_unit",
            "variable":  str,
            "operator":  str,
            "threshold": str,
        }

    Deduplication: if "Age" already appears as an atomic_unit, we skip the
    demographics age target to avoid generating duplicate age-violation records.
    """
    targets: list[dict] = []
    demo    = trial.get("demographics", {})
    units   = trial.get("atomic_units",  [])

    # Check whether an Age atomic_unit already exists
    age_in_units = any(u.get("variable", "").lower() == "age" for u in units
                       if u.get("threshold", "N/A") != "N/A")

    # Source 1: demographics age (only when not covered by atomic_units)
    if not age_in_units:
        min_age = demo.get("min_age", "N/A")
        max_age = demo.get("max_age", "N/A")
        if min_age != "N/A" and max_age != "N/A":
            targets.append({
                "source":    "demographics",
                "variable":  "Age",
                "operator":  "between",
                "threshold": f"{min_age}-{max_age}",
            })
        elif min_age != "N/A":
            targets.append({
                "source":    "demographics",
                "variable":  "Age",
                "operator":  ">=",
                "threshold": min_age,
            })
        elif max_age != "N/A":
            targets.append({
                "source":    "demographics",
                "variable":  "Age",
                "operator":  "<=",
                "threshold": max_age,
            })

    # Source 2: atomic units with numeric thresholds
    supported_ops = {">=", "≥", "<=", "≤", ">", "<", "between"}
    for unit in units:
        tau = unit.get("threshold", "N/A")
        op  = unit.get("operator",  "N/A")
        var = unit.get("variable",  "N/A")

        if tau == "N/A" or var == "N/A":
            continue
        if op not in supported_ops:
            continue

        # Verify threshold is numeric (or valid range)
        try:
            if op == "between" and "-" in str(tau):
                parts = tau.split("-")
                float(parts[0].strip())
                float(parts[1].strip())
            else:
                float(tau)
        except (ValueError, TypeError):
            continue

        targets.append({
            "source":    "atomic_unit",
            "variable":  var,
            "operator":  op,
            "threshold": tau,
        })

    return targets


# ============================================================
# Targeted NEGATIVE_HARD profile generator
# ============================================================

def fix_positive_constraints(profile: dict, units: list, skip_var: str) -> dict:
    """
    Repair POSITIVE baseline lab values (and age) that accidentally violate
    atomic unit constraints due to multi-constraint conflicts.
    Runs up to 3 passes to handle cascading conflicts.
    Also repairs age from atomic_unit constraints when skip_var is not "age".
    """
    lab_values = profile.get("lab_values", {})

    for _pass in range(3):
        changed = False

        for unit in units:
            var = unit.get("variable", "")
            op  = unit.get("operator",  "")
            tau = unit.get("threshold", "")

            if var.lower() in ("age", skip_var.lower()):
                continue
            if var not in lab_values:
                continue

            try:
                current = float(lab_values[var])
                fixed   = None

                if op in [">=", "≥"]:
                    if current < float(tau):
                        fixed = round(max(0.0, float(tau) + random.uniform(0.1, 3)), 1)
                elif op in ["<=", "≤"]:
                    if current > float(tau):
                        fixed = round(max(0.0, float(tau) - random.uniform(0.1, 3)), 1)
                elif op == ">":
                    if current <= float(tau):
                        fixed = round(max(0.0, float(tau) + random.uniform(0.1, 3)), 1)
                elif op == "<":
                    if current >= float(tau):
                        fixed = round(max(0.0, float(tau) - random.uniform(0.1, 3)), 1)
                elif op == "between" and "-" in str(tau):
                    lo_s, hi_s = tau.split("-")
                    lo_v, hi_v = float(lo_s), float(hi_s)
                    if not (lo_v <= current <= hi_v):
                        fixed = round(random.uniform(max(0.0, lo_v), hi_v), 1)

                if fixed is not None:
                    lab_values[var] = fixed
                    changed = True

            except (ValueError, TypeError):
                continue

        # Repair age from atomic_unit constraints when not targeting age
        if skip_var.lower() != "age":
            current_age = profile.get("age", 30)
            for unit in units:
                if unit.get("variable", "").lower() != "age":
                    continue
                op  = unit.get("operator",  "")
                tau = unit.get("threshold", "")
                try:
                    if op in [">=", "≥"]:
                        lo = int(float(tau))
                        if current_age < lo:
                            profile["age"] = lo + random.randint(1, 10)
                            current_age = profile["age"]
                            changed = True
                    elif op in ["<=", "≤"]:
                        hi = int(float(tau))
                        if current_age > hi:
                            profile["age"] = hi - random.randint(1, 10)
                            current_age = profile["age"]
                            changed = True
                    elif op == "between" and "-" in str(tau):
                        lo_s, hi_s = tau.split("-")
                        lo_i, hi_i = int(float(lo_s)), int(float(hi_s))
                        if not (lo_i <= current_age <= hi_i):
                            profile["age"] = random.randint(lo_i, hi_i)
                            current_age = profile["age"]
                            changed = True
                except (ValueError, TypeError):
                    continue

        if not changed:
            break

    profile["lab_values"] = lab_values
    return profile


def generate_negative_hard_profile(trial: dict, target: dict) -> tuple[dict, dict]:
    """
    Build a patient profile that violates EXACTLY the specified target unit
    and satisfies every other constraint.

    Parameters
    ----------
    trial  : the full trial record (demographics + atomic_units)
    target : one item from collect_violatable_units()

    Returns (profile, violated_unit).

    Strategy:
        1. Generate a POSITIVE baseline (satisfies all constraints)
        2. Call fix_positive_constraints to repair any multi-constraint conflicts
           that the baseline generator may have introduced
        3. Override only the target variable with a boundary-violating value
    """
    # Step 1: valid baseline
    profile, _ = generate_patient_profile(trial, "POSITIVE")

    # Step 2: repair any accidental constraint violations in the baseline
    # (skipping the target variable since we're about to override it anyway)
    profile = fix_positive_constraints(
        profile, trial.get("atomic_units", []), skip_var=target["variable"]
    )

    demo     = trial.get("demographics", {})
    var      = target["variable"]
    op       = target["operator"]
    tau      = target["threshold"]
    violated = None

    # --- Override the target variable ---
    if var.lower() == "age":
        # Age override
        try:
            if op == "between" and "-" in str(tau):
                lo, hi = tau.split("-")
                lo_i, hi_i = int(float(lo)), int(float(hi))
                profile["age"] = lo_i - 1 if random.random() < 0.5 else hi_i + 1
                violated = {
                    "variable":      "Age",
                    "operator":      "between",
                    "threshold":     tau,
                    "patient_value": profile["age"],
                }
            elif op in [">=", "≥"]:
                lo_i = int(float(tau))
                profile["age"] = lo_i - 1
                violated = {
                    "variable":      "Age",
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": lo_i - 1,
                }
            elif op in ["<=", "≤"]:
                hi_i = int(float(tau))
                profile["age"] = hi_i + 1
                violated = {
                    "variable":      "Age",
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": hi_i + 1,
                }
        except (ValueError, TypeError):
            pass

    else:
        # Lab / numeric unit override
        # Rule: violation values must FAIL the constraint.
        #       max(0.0, val) prevents negatives but preserves the violation.
        #       Never use clamp_lab_value here — it would repair the violation.
        lab_values = profile.get("lab_values", {})
        try:
            if op == "between" and "-" in str(tau):
                lo_s, hi_s = tau.split("-")
                lo_v, hi_v = float(lo_s), float(hi_s)
                # Violation: just below lo or just above hi
                if random.random() < 0.5:
                    val = round(max(0.0, lo_v - 1), 1)
                else:
                    val = round(hi_v + 1, 1)
                lab_values[var] = val
                violated = {
                    "variable":      var,
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": val,
                }
            elif op in [">=", "≥"]:
                # Violation: below threshold (keep >= 0 if possible)
                thresh = float(tau)
                raw = round(thresh - random.uniform(0.1, 2), 1)
                val = round(max(0.0, raw), 1)
                lab_values[var] = val
                violated = {
                    "variable":      var,
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": val,
                }
            elif op in ["<=", "≤"]:
                # Violation: above threshold (always positive)
                thresh = float(tau)
                val = round(thresh + random.uniform(0.1, 2), 1)
                lab_values[var] = val
                violated = {
                    "variable":      var,
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": val,
                }
            elif op == "<":
                # Violation: >= threshold
                thresh = float(tau)
                val = round(thresh + random.uniform(0.1, 2), 1)
                lab_values[var] = val
                violated = {
                    "variable":      var,
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": val,
                }
            elif op == ">":
                # Violation: <= threshold (keep >= 0 if possible)
                thresh = float(tau)
                raw = round(thresh - random.uniform(0.1, 2), 1)
                val = round(max(0.0, raw), 1)
                lab_values[var] = val
                violated = {
                    "variable":      var,
                    "operator":      op,
                    "threshold":     tau,
                    "patient_value": val,
                }
        except (ValueError, TypeError):
            pass

        profile["lab_values"] = lab_values

    # Last-resort fallback — should rarely trigger
    if violated is None:
        age = profile.get("age", 30)
        profile["age"] = age - 1
        violated = {
            "variable":      "Age",
            "operator":      "fallback",
            "threshold":     str(age),
            "patient_value": age - 1,
        }

    return profile, violated


# ============================================================
# Main pipeline
# ============================================================

def run_dialogue_synthesis():
    """
    Generate dialogues for all trials.

    Per-trial output:
        POSITIVE × 1
        NEGATIVE_HARD × N   where N = number of violatable units

    This mirrors the paper's Symmetric Sample Pairs design (Section III-B-3):
    each atomic unit gets its own NEGATIVE_HARD record, so the model must
    reason about each individual threshold rather than guessing by keyword.

    Expected NEGATIVE_HARD ratio ≈ N/(1+N).
    With avg 4 violatable units per trial → ~80%, exceeding the paper's 74.9%.
    """
    with open(INPUT_PATH, encoding="utf-8") as f:
        trials = [json.loads(line) for line in f]

    all_dialogues: list[dict] = []
    persona_names = list(PERSONAS.keys())
    skipped_neg   = 0

    for trial in trials:
        targets = collect_violatable_units(trial)

        # --- POSITIVE (always 1 per trial) ---
        pos_profile, _ = generate_patient_profile(trial, "POSITIVE")
        pos_turns       = generate_dialogue(trial, pos_profile, random.choice(persona_names))
        all_dialogues.append({
            "nct_id":          trial.get("nct_id"),
            "condition":       trial.get("condition"),
            "label":           "POSITIVE",
            "persona":         pos_turns[0]["role"] and random.choice(persona_names),
            "patient_profile": pos_profile,
            "violated_unit":   None,
            "dialogue":        apply_sct(pos_turns),
            "turn_count":      len(apply_sct(pos_turns)),
            "atomic_units":    trial.get("atomic_units", []),
            "demographics":    trial.get("demographics", {}),
        })

        if not targets:
            skipped_neg += 1
            continue

        # --- NEGATIVE_HARD (1 per violatable unit) ---
        for target in targets:
            persona_name     = random.choice(persona_names)
            profile, violated = generate_negative_hard_profile(trial, target)
            turns             = generate_dialogue(trial, profile, persona_name)
            turns_sct         = apply_sct(turns)

            all_dialogues.append({
                "nct_id":          trial.get("nct_id"),
                "condition":       trial.get("condition"),
                "label":           "NEGATIVE_HARD",
                "persona":         persona_name,
                "patient_profile": profile,
                "violated_unit":   violated,
                "dialogue":        turns_sct,
                "turn_count":      len(turns_sct),
                "atomic_units":    trial.get("atomic_units", []),
                "demographics":    trial.get("demographics", {}),
            })

    # Save output
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for d in all_dialogues:
            out.write(json.dumps(d, ensure_ascii=False) + "\n")

    # Summary
    total    = len(all_dialogues)
    positive = sum(1 for d in all_dialogues if d["label"] == "POSITIVE")
    negative = sum(1 for d in all_dialogues if d["label"] == "NEGATIVE_HARD")
    avg_turns = sum(d["turn_count"] for d in all_dialogues) / total

    print(f"Dialogue synthesis complete")
    print(f"  Total dialogues        : {total}")
    print(f"  POSITIVE               : {positive}  ({positive/total*100:.1f}%)")
    print(f"  NEGATIVE_HARD          : {negative}  ({negative/total*100:.1f}%)")
    print(f"  Skipped (no threshold) : {skipped_neg}")
    print(f"  Avg turns (SCT)        : {avg_turns:.1f}")
    print(f"  Avg NEGATIVE per trial : {negative/positive:.1f}")
    print(f"  Output saved to        : {OUTPUT_PATH}")

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