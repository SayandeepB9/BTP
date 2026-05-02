"""
Evaluate correctness of model solutions for HumanEval, MBPP, and LiveCodeBench v6.

Uses evalplus for HumanEval/MBPP and lcb_runner for LiveCodeBench.

Usage:
    python src/evaluate.py data/benchmarks/Qwen--Qwen3-8B

Reads each dataset JSON from the given folder (if present), converts solutions
to the format expected by each evaluation harness, runs evaluation, and writes
results back into each JSON as a "passed" boolean per entry plus a summary.
"""

import argparse
import json
import os
import re
import sys
import tempfile
from pathlib import Path

LCB_REPO = "/tmp/LiveCodeBench"

BENCHMARKS_DIR = Path(__file__).resolve().parents[1] / "data" / "benchmarks"


def _strip_python_tags(solution: str) -> str:
    """Remove ```python and ``` tags to get raw code."""
    code = solution.strip()
    if code.startswith("```python"):
        code = code[len("```python"):]
    if code.endswith("```"):
        code = code[:-3]
    return code.strip()


def _rename_function(code: str, expected_name: str) -> str:
    """Rename the model's main function to the expected entry_point.

    Finds top-level 'def' statements, picks the last one as the main function,
    and renames all occurrences of that name to expected_name.
    """
    if not code.strip():
        return code

    # Find all top-level function definitions (no leading whitespace)
    top_level_defs = re.findall(r'^def\s+(\w+)\s*\(', code, re.MULTILINE)
    if not top_level_defs:
        return code

    # If the expected name is already defined, nothing to do
    if expected_name in top_level_defs:
        return code

    # Use the last top-level def as the "main" function
    model_name = top_level_defs[-1]

    # Replace all occurrences (def, calls, recursive refs) using word boundary
    return re.sub(r'\b' + re.escape(model_name) + r'\b', expected_name, code)


# ---------- HumanEval / MBPP via evalplus ----------

def _evalplus_task_id(entry, dataset_name: str) -> str:
    """Convert a dataset entry's task_id to the format evalplus expects.

    HumanEval task_ids are already strings like 'HumanEval/0'.
    MBPP task_ids are integers like 11 and need to become 'Mbpp/11'.
    """
    tid = entry["task_id"]
    if dataset_name == "mbpp" and isinstance(tid, int):
        return f"Mbpp/{tid}"
    return str(tid)


def eval_evalplus(
    folder: Path,
    filename: str,
    dataset_name: str,
    only_missing_verdict: bool = True,
) -> dict | None:
    """Evaluate HumanEval or MBPP using evalplus.

    dataset_name: "humaneval" or "mbpp" (as expected by evalplus).
    """
    filepath = folder / filename
    if not filepath.exists():
        print(f"  {dataset_name}: {filename} not found, skipping")
        return None

    with open(filepath) as f:
        data = json.load(f)

    forced_fail_indices = []
    touched_rows = False

    # For MBPP, only entries whose task_id exists in evalplus can be evaluated
    if dataset_name == "mbpp":
        from evalplus.data import get_mbpp_plus
        valid_ids = set(get_mbpp_plus().keys())
        eval_indices = []
        for i, entry in enumerate(data):
            if _evalplus_task_id(entry, dataset_name) in valid_ids:
                eval_indices.append(i)
            else:
                forced_fail_indices.append(i)
                if entry.get("passed") is not False or entry.get("verdict") != 0:
                    touched_rows = True
                entry["passed"] = False
                entry["verdict"] = 0

        print(f"  {dataset_name}: {len(data)} entries total, {len(eval_indices)} match evalplus MBPP+")
        if forced_fail_indices:
            print(
                f"  {dataset_name}: marked {len(forced_fail_indices)} non-evalplus entries as failed (verdict=0)"
            )
    else:
        eval_indices = list(range(len(data)))

    # Skip entries without solutions
    no_solution = sum(1 for i in eval_indices if not data[i].get("solution"))
    eval_indices = [i for i in eval_indices if data[i].get("solution")]
    total = len(eval_indices)
    if no_solution:
        print(f"  {dataset_name}: skipping {no_solution} entries with no solution")

    if only_missing_verdict:
        pending_indices = [i for i in eval_indices if "verdict" not in data[i]]
        already_done = total - len(pending_indices)
        if already_done:
            print(f"  {dataset_name}: reusing {already_done}/{total} existing verdicts")
        print(f"  {dataset_name}: evaluating {len(pending_indices)} pending entries")
    else:
        pending_indices = list(eval_indices)
        print(f"  {dataset_name}: evaluating {len(pending_indices)} entries")

    if not pending_indices:
        num_passed = sum(1 for i in eval_indices if data[i].get("verdict") == 1)
        total_with_forced = total + len(forced_fail_indices)
        pass_rate = num_passed / total_with_forced * 100 if total_with_forced else 0
        display = "HumanEval" if dataset_name == "humaneval" else "MBPP"
        print(f"  {display}: {num_passed}/{total_with_forced} passed ({pass_rate:.1f}%)")

        if touched_rows:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, default=str)

        return {
            "dataset_key": dataset_name,
            "dataset": display,
            "total": total_with_forced,
            "passed": num_passed,
            "pass_rate": round(pass_rate, 2),
        }

    # Build evalplus JSONL: {task_id, solution} per entry
    # evalplus requires ALL problems to be present; provide empty code for missing ones
    our_solutions = {}
    for i in pending_indices:
        entry = data[i]
        raw_solution = entry.get("solution", "")
        code = _strip_python_tags(raw_solution) if raw_solution else ""
        ep_tid = _evalplus_task_id(entry, dataset_name)
        our_solutions[ep_tid] = code

    if dataset_name == "mbpp":
        from evalplus.data import get_mbpp_plus
        all_ids = sorted(get_mbpp_plus().keys())
    elif dataset_name == "humaneval":
        from evalplus.data import get_human_eval_plus
        all_ids = sorted(get_human_eval_plus().keys())
    else:
        all_ids = sorted(our_solutions.keys())

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, dir=folder
    ) as tmp:
        tmp_path = tmp.name
        for tid in all_ids:
            code = our_solutions.get(tid, "")
            tmp.write(json.dumps({"task_id": tid, "solution": code}) + "\n")

    results_path = tmp_path.replace(".jsonl", "_eval_results.json")

    try:
        from evalplus.evaluate import evaluate

        # Remove stale results file if it exists to avoid overwrite prompt
        if os.path.exists(results_path):
            os.unlink(results_path)

        evaluate(
            dataset=dataset_name,
            samples=tmp_path,
            base_only=False,
            parallel=min(os.cpu_count() or 4, 16),
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if os.path.exists(results_path):
        with open(results_path) as f:
            eval_results = json.load(f)

        eval_detail = eval_results.get("eval", {})
        # Mark only the entries evaluated in this run.
        for i in pending_indices:
            entry = data[i]
            ep_tid = _evalplus_task_id(entry, dataset_name)
            task_results = eval_detail.get(ep_tid, [])
            passed = any(
                r.get("base_status") == "pass" and r.get("plus_status") == "pass"
                for r in task_results
            )
            entry["passed"] = passed
            entry["verdict"] = 1 if passed else 0

        os.unlink(results_path)
    else:
        for i in pending_indices:
            data[i]["passed"] = False
            data[i]["verdict"] = 0

    num_passed = sum(1 for i in eval_indices if data[i].get("verdict") == 1)
    total_with_forced = total + len(forced_fail_indices)
    pass_rate = num_passed / total_with_forced * 100 if total_with_forced else 0
    display = "HumanEval" if dataset_name == "humaneval" else "MBPP"
    print(f"  {display}: {num_passed}/{total_with_forced} passed ({pass_rate:.1f}%)")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return {
        "dataset_key": dataset_name,
        "dataset": display,
        "total": total_with_forced,
        "passed": num_passed,
        "pass_rate": round(pass_rate, 2),
    }


# ---------- LiveCodeBench v6 via lcb_runner ----------

def eval_livecodebench(folder: Path, only_missing_verdict: bool = True) -> dict | None:
    filepath = folder / "livecodebench_v6.json"
    if not filepath.exists():
        print(f"  LiveCodeBench v6: livecodebench_v6.json not found, skipping")
        return None

    # Ensure lcb_runner is importable
    if LCB_REPO not in sys.path:
        sys.path.insert(0, LCB_REPO)

    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.evaluation.compute_code_generation_metrics import check_correctness

    # Load original dataset to build CodeGenerationProblem objects (need raw fields)
    src_path = BENCHMARKS_DIR / "livecodebench_v6.json"
    with open(src_path) as f:
        src_data = json.load(f)

    with open(filepath) as f:
        data = json.load(f)

    # Only evaluate entries that have solutions
    eval_pairs = [
        (i, src_entry, entry)
        for i, (src_entry, entry) in enumerate(zip(src_data, data))
        if entry.get("solution")
    ]

    if only_missing_verdict:
        pending_pairs = [pair for pair in eval_pairs if "verdict" not in pair[2]]
        already_done = len(eval_pairs) - len(pending_pairs)
        if already_done:
            print(
                f"  LiveCodeBench v6: reusing {already_done}/{len(eval_pairs)} existing verdicts"
            )
    else:
        pending_pairs = list(eval_pairs)

    no_solution = len(data) - len(eval_pairs)
    if no_solution:
        print(f"  LiveCodeBench v6: skipping {no_solution} entries with no solution")
    print(f"  LiveCodeBench v6: evaluating {len(pending_pairs)} pending entries")

    for idx, (_, src_entry, entry) in enumerate(pending_pairs):
        raw_solution = entry.get("solution", "")
        code = _strip_python_tags(raw_solution) if raw_solution else ""

        try:
            problem = CodeGenerationProblem(**src_entry)
            sample = problem.get_evaluation_sample()
            result, _ = check_correctness(sample, code, timeout=6, debug=False)
            entry["passed"] = all(r == True for r in result)
            entry["verdict"] = 1 if entry["passed"] else 0
        except Exception as e:
            entry["passed"] = False
            entry["verdict"] = 0
            print(f"  [{idx + 1}/{len(pending_pairs)}] error: {e}")
            continue

        status = "passed" if entry["passed"] else "failed"
        print(f"  [{idx + 1}/{len(pending_pairs)}] {status}")

    total = len(eval_pairs)
    num_passed = sum(1 for _, _, e in eval_pairs if e.get("verdict") == 1)
    pass_rate = num_passed / total * 100 if total else 0
    print(f"  LiveCodeBench v6: {num_passed}/{total} passed ({pass_rate:.1f}%)")

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2, default=str)

    return {
        "dataset_key": "livecodebench",
        "dataset": "LiveCodeBench v6",
        "total": total,
        "passed": num_passed,
        "pass_rate": round(pass_rate, 2),
    }


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Evaluate model solutions")
    parser.add_argument("folder", type=Path, help="Model results folder")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["humaneval", "mbpp", "livecodebench"],
        default=["humaneval", "mbpp", "livecodebench"],
        help="Datasets to evaluate (default: all)",
    )
    parser.add_argument(
        "--recheck-all",
        action="store_true",
        help="Re-evaluate all entries with solutions (default: evaluate only missing verdicts)",
    )
    args = parser.parse_args()

    folder = args.folder.resolve()
    if not folder.is_dir():
        print(f"Error: {folder} is not a directory")
        sys.exit(1)

    only_missing_verdict = not args.recheck_all
    selected = set(args.datasets)

    print(f"Evaluating: {folder}\n")
    summary_order = ["humaneval", "mbpp", "livecodebench"]
    summary_by_key = {}

    summary_path = folder / "results.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                existing_summary = json.load(f)
            for row in existing_summary:
                label = row.get("dataset")
                if label == "HumanEval":
                    summary_by_key["humaneval"] = row
                elif label == "MBPP":
                    summary_by_key["mbpp"] = row
                elif label == "LiveCodeBench v6":
                    summary_by_key["livecodebench"] = row
        except Exception:
            pass

    # HumanEval
    if "humaneval" in selected:
        print("=== HumanEval ===")
        r = eval_evalplus(
            folder,
            "humaneval.json",
            "humaneval",
            only_missing_verdict=only_missing_verdict,
        )
        if r:
            summary_by_key["humaneval"] = {
                "dataset": r["dataset"],
                "total": r["total"],
                "passed": r["passed"],
                "pass_rate": r["pass_rate"],
            }
        print()

    # MBPP
    if "mbpp" in selected:
        print("=== MBPP ===")
        r = eval_evalplus(
            folder,
            "mbpp.json",
            "mbpp",
            only_missing_verdict=only_missing_verdict,
        )
        if r:
            summary_by_key["mbpp"] = {
                "dataset": r["dataset"],
                "total": r["total"],
                "passed": r["passed"],
                "pass_rate": r["pass_rate"],
            }
        print()

    # LiveCodeBench v6
    if "livecodebench" in selected:
        print("=== LiveCodeBench v6 ===")
        r = eval_livecodebench(folder, only_missing_verdict=only_missing_verdict)
        if r:
            summary_by_key["livecodebench"] = {
                "dataset": r["dataset"],
                "total": r["total"],
                "passed": r["passed"],
                "pass_rate": r["pass_rate"],
            }
        print()

    # Summary
    summary = [summary_by_key[k] for k in summary_order if k in summary_by_key]
    if summary:
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print("=" * 50)
        print(f"{'Dataset':<20} {'Passed':>8} {'Total':>8} {'Rate':>8}")
        print("-" * 50)
        for r in summary:
            print(f"{r['dataset']:<20} {r['passed']:>8} {r['total']:>8} {r['pass_rate']:>7.1f}%")
        print("=" * 50)
        print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
