"""
quality_report.py

Comprehensive quality report for the TrialSim-10k pipeline output.

Reads the validated dialogue file and computes all metrics needed to
verify that the generated dataset matches the statistical targets
described in the paper.  Designed to be run after phase4_validate.py.

Paper reference targets (used as benchmarks throughout):
    Table VII  — Label distribution  : ~74.9% NEGATIVE_HARD
    Table VII  — Persona distribution: COOPERATIVE/RELUCTANT/ANXIOUS/CHATTY/FORGETFUL
    Table IX   — Linguistic stats    : avg 5.66 turns, 330 words overall
                                       CHATTY ~482 words, RELUCTANT ~233 words
    Section III-F — Logic density    : avg 10.85 atomic units per trial

Usage:
    python quality_report.py                          # default path
    python quality_report.py --path custom/path.jsonl # custom path
    python quality_report.py --json                   # dump full JSON report
"""

import json
import argparse
import sys
from collections import Counter, defaultdict
from statistics  import mean, stdev
from pathlib     import Path

# ---------------------------------------------------------------------------
# Default path — matches phase4_validate.py output
# ---------------------------------------------------------------------------
DEFAULT_PATH = "data/dialogues/validated.jsonl"

# ---------------------------------------------------------------------------
# Paper benchmark targets (Section III / Tables VII, IX)
# ---------------------------------------------------------------------------
PAPER_TARGETS = {
    "negative_hard_ratio_pct": 74.9,
    "avg_turns":               5.66,
    "avg_words_overall":       330.36,
    "avg_atomic_units":        10.85,
    "persona_avg_words": {
        "CHATTY":      482.68,
        "FORGETFUL":   357.30,
        "ANXIOUS":     332.70,
        "COOPERATIVE": 320.63,
        "RELUCTANT":   233.30,
    },
}

# ---------------------------------------------------------------------------
# Helper: delta indicator
# ---------------------------------------------------------------------------

def _delta(actual: float, target: float, unit: str = "") -> str:
    """
    Format the difference between actual and paper target as a signed string.
    E.g.  _delta(76.2, 74.9, "%")  ->  "+1.3%  [target 74.9%]"
    """
    diff = actual - target
    sign = "+" if diff >= 0 else ""
    return f"{sign}{diff:.1f}{unit}  [paper target: {target}{unit}]"

# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_records(path: str) -> list[dict]:
    """Load a JSONL file and return a list of record dicts."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with open(p, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    if not records:
        print(f"[ERROR] File is empty: {path}", file=sys.stderr)
        sys.exit(1)
    return records

# ---------------------------------------------------------------------------
# Section 1: Dataset overview
# ---------------------------------------------------------------------------

def section_overview(records: list[dict]) -> dict:
    """
    Basic counts: total records, unique trials, label split.
    """
    total       = len(records)
    unique_ncts = len({r.get("nct_id") for r in records})
    pos         = sum(1 for r in records if r.get("label") == "POSITIVE")
    neg         = sum(1 for r in records if r.get("label") == "NEGATIVE_HARD")
    neg_ratio   = neg / total * 100 if total else 0

    print("=" * 60)
    print("SECTION 1 — Dataset Overview")
    print("=" * 60)
    print(f"  Total validated records : {total}")
    print(f"  Unique NCT IDs          : {unique_ncts}")
    print(f"  POSITIVE                : {pos}  ({pos / total * 100:.1f}%)")
    print(f"  NEGATIVE_HARD           : {neg}  ({neg / total * 100:.1f}%)")
    print(f"  NEGATIVE ratio          : {_delta(neg_ratio, PAPER_TARGETS['negative_hard_ratio_pct'], '%')}")

    return {"total": total, "unique_ncts": unique_ncts,
            "positive": pos, "negative_hard": neg,
            "negative_hard_ratio_pct": round(neg_ratio, 2)}

# ---------------------------------------------------------------------------
# Section 2: Persona distribution (paper Table VII)
# ---------------------------------------------------------------------------

def section_persona_distribution(records: list[dict]) -> dict:
    """
    Count and percentage breakdown by persona.
    Matches paper Table VII right column.
    """
    total  = len(records)
    counts = Counter(r.get("persona", "UNKNOWN") for r in records)

    print("\n" + "=" * 60)
    print("SECTION 2 — Persona Distribution  (paper Table VII)")
    print("=" * 60)
    print(f"  {'Persona':<15}  {'Count':>6}  {'%':>6}")
    print(f"  {'-'*15}  {'-'*6}  {'-'*6}")
    for persona, count in sorted(counts.items(), key=lambda x: -x[1]):
        pct = count / total * 100
        print(f"  {persona:<15}  {count:>6}  {pct:>5.1f}%")

    return dict(counts)

# ---------------------------------------------------------------------------
# Section 3: Linguistic complexity (paper Table IX)
# ---------------------------------------------------------------------------

def _word_count(record: dict) -> int:
    """Total word count across all turns in one dialogue."""
    return sum(len(t["text"].split()) for t in record.get("dialogue", []))

def section_linguistic_complexity(records: list[dict]) -> dict:
    """
    Per-persona averages for turn count and word count.
    Matches paper Table IX.
    """
    print("\n" + "=" * 60)
    print("SECTION 3 — Linguistic Complexity  (paper Table IX)")
    print("=" * 60)

    all_turns = [r["turn_count"] for r in records]
    all_words = [_word_count(r) for r in records]

    print(f"  Overall avg turns : {mean(all_turns):.2f}  "
          f"{_delta(mean(all_turns), PAPER_TARGETS['avg_turns'])}")
    print(f"  Overall avg words : {mean(all_words):.0f}  "
          f"{_delta(mean(all_words), PAPER_TARGETS['avg_words_overall'])}")
    print()

    personas   = sorted({r.get("persona") for r in records})
    per_persona: dict = {}

    print(f"  {'Persona':<15}  {'Avg turns':>10}  {'Avg words':>10}  {'Paper words':>12}  Delta")
    print(f"  {'-'*15}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")

    for persona in personas:
        subset     = [r for r in records if r.get("persona") == persona]
        avg_turns  = mean(r["turn_count"]  for r in subset)
        avg_words  = mean(_word_count(r)   for r in subset)
        paper_wds  = PAPER_TARGETS["persona_avg_words"].get(persona, None)
        delta_str  = (f"{avg_words - paper_wds:+.0f}"
                      if paper_wds is not None else "n/a")

        print(f"  {persona:<15}  {avg_turns:>10.2f}  {avg_words:>10.0f}  "
              f"{str(paper_wds) if paper_wds else 'n/a':>12}  {delta_str}")

        per_persona[persona] = {
            "avg_turns": round(avg_turns, 2),
            "avg_words": round(avg_words, 1),
        }

    return {
        "overall_avg_turns": round(mean(all_turns), 2),
        "overall_avg_words": round(mean(all_words), 1),
        "per_persona":       per_persona,
    }

# ---------------------------------------------------------------------------
# Section 4: Atomic unit density (paper Section III-F)
# ---------------------------------------------------------------------------

def section_atomic_density(records: list[dict]) -> dict:
    """
    Average number of atomic units per trial.
    Paper reports 10.85 atomic units per trial on average.

    Note: atomic_units may not be present in every record depending on
    which phase generated the dialogues.  Records without the field are
    skipped with a warning.
    """
    print("\n" + "=" * 60)
    print("SECTION 4 — Atomic Unit Density  (paper Section III-F)")
    print("=" * 60)

    densities = [
        len(r["atomic_units"])
        for r in records
        if "atomic_units" in r
    ]
    missing = len(records) - len(densities)

    if not densities:
        print("  [WARNING] No records contain 'atomic_units' field.")
        print("  Run phase2_atomize.py and re-validate to populate this field.")
        return {}

    if missing:
        print(f"  [WARNING] {missing} records missing 'atomic_units' — skipped.")

    avg    = mean(densities)
    sd     = stdev(densities) if len(densities) > 1 else 0.0
    minval = min(densities)
    maxval = max(densities)

    print(f"  Avg atomic units/trial : {avg:.2f}  "
          f"{_delta(avg, PAPER_TARGETS['avg_atomic_units'])}")
    print(f"  Std dev                : {sd:.2f}")
    print(f"  Min / Max              : {minval} / {maxval}")

    # Distribution buckets: 0-5, 6-10, 11-15, 16+
    buckets = {"0-5": 0, "6-10": 0, "11-15": 0, "16+": 0}
    for d in densities:
        if   d <= 5:  buckets["0-5"]   += 1
        elif d <= 10: buckets["6-10"]  += 1
        elif d <= 15: buckets["11-15"] += 1
        else:         buckets["16+"]   += 1

    print("\n  Density distribution:")
    for bucket, count in buckets.items():
        bar = "█" * (count * 30 // len(densities))
        print(f"    {bucket:>5} units : {count:>5}  {bar}")

    return {
        "avg":     round(avg, 2),
        "stdev":   round(sd,  2),
        "min":     minval,
        "max":     maxval,
        "buckets": buckets,
    }

# ---------------------------------------------------------------------------
# Section 5: Therapeutic area coverage (paper Table VIII)
# ---------------------------------------------------------------------------

def section_disease_coverage(records: list[dict]) -> dict:
    """
    Top-N disease conditions and their percentage share.
    Paper Table VIII shows a long-tail distribution with Breast Neoplasms
    at 6.1% as the most frequent category.
    """
    print("\n" + "=" * 60)
    print("SECTION 5 — Therapeutic Area Coverage  (paper Table VIII)")
    print("=" * 60)

    total    = len(records)
    counts   = Counter(r.get("condition", "Unknown") for r in records)
    top_n    = 10

    print(f"  Unique conditions : {len(counts)}")
    print(f"  Top {top_n}:")
    print(f"  {'Condition':<40}  {'Count':>6}  {'%':>5}")
    print(f"  {'-'*40}  {'-'*6}  {'-'*5}")

    for condition, count in counts.most_common(top_n):
        pct = count / total * 100
        # Flag if any single category exceeds 10% (paper shows ~6% max)
        flag = "  ← high concentration" if pct > 10 else ""
        cond_display = (condition[:37] + "...") if len(condition) > 40 else condition
        print(f"  {cond_display:<40}  {count:>6}  {pct:>4.1f}%{flag}")

    return {
        "unique_conditions": len(counts),
        "top_10": {k: v for k, v in counts.most_common(10)},
    }

# ---------------------------------------------------------------------------
# Section 6: Violation quality (NEGATIVE_HARD records only)
# ---------------------------------------------------------------------------

def section_violation_quality(records: list[dict]) -> dict:
    """
    For NEGATIVE_HARD records, analyse which variables are most commonly
    the violated unit and what the violation source breakdown is.

    violation_source field (added by phase2_atomize.py V4):
        "rule_v1"          — computed by rule-based epsilon logic
        "llm_v4"           — upgraded by Logical Architect prompt
        "rule_v1_fallback" — LLM was attempted but fell back to rules
    """
    print("\n" + "=" * 60)
    print("SECTION 6 — Violation Quality  (NEGATIVE_HARD records)")
    print("=" * 60)

    neg_records = [r for r in records if r.get("label") == "NEGATIVE_HARD"]
    if not neg_records:
        print("  No NEGATIVE_HARD records found.")
        return {}

    # Most-violated variables
    violated_vars = Counter(
        r["violated_unit"]["variable"]
        for r in neg_records
        if r.get("violated_unit") and r["violated_unit"].get("variable")
    )

    print(f"  Total NEGATIVE_HARD records : {len(neg_records)}")
    print(f"\n  Most violated variables (top 10):")
    for var, count in violated_vars.most_common(10):
        pct = count / len(neg_records) * 100
        print(f"    {var:<30}  {count:>5}  ({pct:.1f}%)")

    # Violation source breakdown (only present if phase2 V4 was used)
    all_units = [
        u
        for r in records
        for u in r.get("atomic_units", [])
        if "violation_source" in u
    ]
    if all_units:
        source_counts = Counter(u["violation_source"] for u in all_units)
        total_units   = len(all_units)
        print(f"\n  Violation source breakdown ({total_units} units with source tag):")
        for source, count in source_counts.most_common():
            pct = count / total_units * 100
            print(f"    {source:<25}  {count:>6}  ({pct:.1f}%)")
    else:
        print("\n  [INFO] No 'violation_source' tags found.")
        print("  Run phase2_atomize.py with use_llm=True to populate V4 metadata.")

    return {
        "total_negative_hard": len(neg_records),
        "top_violated_vars":   dict(violated_vars.most_common(10)),
    }

# ---------------------------------------------------------------------------
# Section 7: SCT effectiveness
# ---------------------------------------------------------------------------

def section_sct_effectiveness(records: list[dict]) -> dict:
    """
    Verify that SCT (Strict Conclusion Truncation) is working correctly
    by checking that no dialogue ends on a recruiter turn.

    After SCT, the last turn in every dialogue must be a patient turn.
    Any record ending on a recruiter turn means SCT was not applied or
    failed, and that record may contain answer leakage.
    """
    print("\n" + "=" * 60)
    print("SECTION 7 — SCT Effectiveness  (anti-leakage check)")
    print("=" * 60)

    sct_ok      = 0
    sct_fail    = 0
    fail_examples: list[str] = []

    for r in records:
        dialogue = r.get("dialogue", [])
        if not dialogue:
            continue
        last_role = dialogue[-1]["role"]
        if last_role == "patient":
            sct_ok += 1
        else:
            sct_fail += 1
            if len(fail_examples) < 3:
                fail_examples.append(r.get("nct_id", "unknown"))

    total_checked = sct_ok + sct_fail
    print(f"  Dialogues checked        : {total_checked}")
    print(f"  Ends on patient turn     : {sct_ok}  ({sct_ok / total_checked * 100:.1f}%)")
    print(f"  Ends on recruiter turn   : {sct_fail}  ({sct_fail / total_checked * 100:.1f}%)")

    if sct_fail > 0:
        print(f"\n  [WARNING] {sct_fail} records may have answer leakage.")
        print(f"  Example NCT IDs: {fail_examples}")
        print("  Re-run phase3 with apply_sct() to fix.")
    else:
        print("\n  [OK] All dialogues end on a patient disclosure turn.")

    return {
        "sct_pass": sct_ok,
        "sct_fail": sct_fail,
        "sct_pass_rate_pct": round(sct_ok / total_checked * 100, 2) if total_checked else 0,
    }

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_quality_report(path: str, dump_json: bool = False) -> dict:
    """
    Run all seven quality sections and optionally dump a JSON summary.

    Parameters
    ----------
    path      : path to the validated JSONL file
    dump_json : if True, write a machine-readable summary to
                data/dialogues/quality_report.json

    Returns the full summary dict (useful for automated testing).
    """
    print(f"\nTrialSim-10k Quality Report")
    print(f"Input: {path}")

    records = load_records(path)
    summary = {"input_path": path}

    summary["overview"]             = section_overview(records)
    summary["persona_distribution"] = section_persona_distribution(records)
    summary["linguistic_complexity"]= section_linguistic_complexity(records)
    summary["atomic_density"]       = section_atomic_density(records)
    summary["disease_coverage"]     = section_disease_coverage(records)
    summary["violation_quality"]    = section_violation_quality(records)
    summary["sct_effectiveness"]    = section_sct_effectiveness(records)

    print("\n" + "=" * 60)
    print("Report complete.")
    print("=" * 60)

    if dump_json:
        out_path = Path(path).parent / "quality_report.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"JSON summary saved to: {out_path}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TrialSim-10k quality report generator"
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help=f"Path to validated JSONL file (default: {DEFAULT_PATH})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Also dump a machine-readable JSON summary alongside the input file",
    )
    args = parser.parse_args()
    generate_quality_report(path=args.path, dump_json=args.json)