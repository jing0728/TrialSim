"""
run_pipeline.py

One-click runner for TrialSim-10k Phase II → III → IV → Quality Report.

Usage:
    python run_pipeline.py                  # template dialogue (fast, free)
    python run_pipeline.py --llm            # LLM dialogue    (slow, costs API)
    python run_pipeline.py --llm --start 3  # resume from a specific step
    python run_pipeline.py --step 4         # run only one step

Steps:
    2  phase2_atomize.py          — atomize parsed_pico.jsonl
    3  phase3_dialogue.py         — template-based dialogue synthesis
    3L phase3_dialogue_llm.py     — LLM dual-agent synthesis (--llm flag)
    4  phase4_validate.py         — quality assurance + validation
    5  quality_report.py          — compare against paper benchmarks
"""

import subprocess
import sys
import time
import argparse
from pathlib import Path

# ---------------------------------------------------------------------------
# Step definitions
# ---------------------------------------------------------------------------

STEPS_TEMPLATE = [
    {
        "id":     2,
        "label":  "Phase II — Atomization",
        "script": "phase2_atomize.py",
        "desc":   "Convert PICO parse results into atomic logic triplets",
    },
    {
        "id":     3,
        "label":  "Phase III — Dialogue Synthesis (template)",
        "script": "phase3_dialogue.py",
        "desc":   "Generate dialogues using hardcoded persona templates",
    },
    {
        "id":     4,
        "label":  "Phase IV — Validation",
        "script": "phase4_validate.py",
        "desc":   "Run 5-check quality assurance pipeline",
    },
    {
        "id":     5,
        "label":  "Quality Report",
        "script": "quality_report.py",
        "desc":   "Compare dataset statistics against paper benchmarks",
    },
]

STEPS_LLM = [
    {
        "id":     2,
        "label":  "Phase II — Atomization",
        "script": "phase2_atomize.py",
        "desc":   "Convert PICO parse results into atomic logic triplets",
    },
    {
        "id":     "3L",
        "label":  "Phase III — Dialogue Synthesis (LLM)",
        "script": "phase3_dialogue_llm.py",
        "desc":   "Generate dialogues using LLM dual-agent framework",
    },
    {
        "id":     4,
        "label":  "Phase IV — Validation",
        "script": "phase4_validate.py",
        "desc":   "Run 5-check quality assurance pipeline",
    },
    {
        "id":     5,
        "label":  "Quality Report",
        "script": "quality_report.py",
        "desc":   "Compare dataset statistics against paper benchmarks",
    },
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def print_banner(text: str, char: str = "=", width: int = 60) -> None:
    print("\n" + char * width)
    print(f"  {text}")
    print(char * width)


def print_step_header(step: dict, index: int, total: int) -> None:
    print_banner(
        f"Step {index}/{total} — [{step['id']}] {step['label']}",
        char="─",
    )
    print(f"  Script : {step['script']}")
    print(f"  Goal   : {step['desc']}")
    print()


def check_script_exists(script: str) -> bool:
    if not Path(script).exists():
        print(f"  [ERROR] {script} not found in current directory.")
        print(f"  Make sure you are running from the TrialSim project root.")
        return False
    return True


def run_step(step: dict, index: int, total: int) -> bool:
    """
    Run a single pipeline step as a subprocess.
    Returns True if the step succeeded (exit code 0), False otherwise.
    """
    print_step_header(step, index, total)

    script = step["script"]
    if not check_script_exists(script):
        return False

    start_time = time.time()

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,   # let stdout/stderr stream live to terminal
    )

    elapsed = time.time() - start_time
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    if result.returncode == 0:
        print(f"\n  ✓  {step['label']} completed in {time_str}")
        return True
    else:
        print(f"\n  ✗  {step['label']} FAILED (exit code {result.returncode})")
        print(f"     Fix the error above before continuing.")
        return False


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TrialSim-10k pipeline runner (Phase II → V)"
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Use LLM dual-agent dialogue synthesis (phase3_dialogue_llm.py) "
             "instead of the template version",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        metavar="STEP_ID",
        help="Resume from this step ID (e.g. --start 3 or --start 3L). "
             "Skips all earlier steps.",
    )
    parser.add_argument(
        "--step",
        type=str,
        default=None,
        metavar="STEP_ID",
        help="Run only a single step by ID and exit.",
    )
    args = parser.parse_args()
    mode = "LLM dual-agent" if args.llm else "template"
    steps = STEPS_LLM if args.llm else STEPS_TEMPLATE
    if not args.llm:
        print("  Free mode: no LLM calls in phases 2-5.")
        print("  Make sure parsed_pico.jsonl was generated by phase1_parse.py, not phase1_parse_llm.py.")

    print_banner(f"TrialSim-10k Pipeline Runner  [{mode} mode]")
    print(f"  Steps to run : {' → '.join(str(s['id']) for s in steps)}")
    if args.llm:
        print("  Note: LLM mode calls the Anthropic API for every dialogue.")
        print("        Estimated cost for 500 trials: ~$2–5 USD.")
    print()

    # Filter steps based on --step or --start flags
    if args.step is not None:
        target_id = args.step
        steps = [s for s in steps if str(s["id"]) == target_id]
        if not steps:
            print(f"[ERROR] No step with ID '{target_id}' found.")
            sys.exit(1)

    elif args.start is not None:
        start_id  = args.start
        step_ids  = [str(s["id"]) for s in steps]
        if start_id not in step_ids:
            print(f"[ERROR] Step ID '{start_id}' not found. "
                  f"Available: {', '.join(step_ids)}")
            sys.exit(1)
        start_idx = step_ids.index(start_id)
        steps     = steps[start_idx:]
        print(f"  Resuming from step {start_id}. "
              f"Skipping: {', '.join(step_ids[:start_idx])}")

    # Run each step
    total      = len(steps)
    results    = {}
    wall_start = time.time()

    for i, step in enumerate(steps, start=1):
        ok = run_step(step, i, total)
        results[str(step["id"])] = "✓ passed" if ok else "✗ failed"

        if not ok:
            print_banner("Pipeline stopped — fix the error above and re-run", char="!")
            print(f"  To resume from this step: python run_pipeline.py --start {step['id']}"
                  + (" --llm" if args.llm else ""))
            _print_summary(results, time.time() - wall_start)
            sys.exit(1)

    # All steps passed
    wall_elapsed = time.time() - wall_start
    _print_summary(results, wall_elapsed)
    print_banner("All steps completed successfully ✓")


def _print_summary(results: dict, elapsed: float) -> None:
    minutes, seconds = divmod(int(elapsed), 60)
    time_str = f"{minutes}m {seconds}s" if minutes else f"{seconds}s"

    print_banner("Run Summary", char="─")
    for step_id, status in results.items():
        print(f"  Step {step_id:<4} : {status}")
    print(f"\n  Total time : {time_str}")


if __name__ == "__main__":
    main()