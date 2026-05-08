"""
Run all lab steps sequentially.

Usage:
  python run_all.py          # Run all steps
  python run_all.py --step 3 # Run only step 3
"""

import argparse
import sys
import subprocess

STEPS = [
    ("01_langsmith_rag_pipeline",  "Step 1: LangSmith RAG Pipeline"),
    ("02_prompt_hub_ab_routing",   "Step 2: Prompt Hub A/B Routing"),
    ("03_ragas_evaluation",         "Step 3: RAGAS Evaluation"),
    ("04_guardrails_validator",      "Step 4: Guardrails AI Validators"),
]


def run_step(step_num: int):
    module = STEPS[step_num - 1][0]
    label = STEPS[step_num - 1][1]
    print(f"\n{'=' * 60}")
    print(f"  Running: {label}")
    print(f"{'=' * 60}\n")
    result = subprocess.run([sys.executable, f"{module}.py"])
    if result.returncode != 0:
        print(f"❌ {label} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    print(f"✅ {label} completed successfully")


def main():
    parser = argparse.ArgumentParser(description="Run Day 22 lab steps")
    parser.add_argument(
        "--step", type=int, choices=[1, 2, 3, 4],
        help="Run only a specific step (1-4)"
    )
    args = parser.parse_args()

    if args.step:
        run_step(args.step)
    else:
        for i in range(1, len(STEPS) + 1):
            run_step(i)
        print(f"\n{'=' * 60}")
        print("  ✅ All steps completed successfully!")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
