from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAFE_DELETE_PATHS = [
    "results/metrics/*.csv",
    "results/metrics/*.txt",
    "results/figures/*.png",
    "results/checkpoints/*.pt",
    "logs/*.log",
    "reports/*.md",
    "docs/final_preprocessing_decision.md",
    "docs/final_benchmark_selection.md",
    "docs/*decision*.md",
    "docs/*selection*.md",
    "docs/*report*.md",
]

KEEP_DIRECTORIES = [
    "results/metrics",
    "results/figures",
    "results/checkpoints",
    "logs",
    "reports",
]

PROTECTED_PREFIXES = [
    PROJECT_ROOT / "data",
    PROJECT_ROOT / "original_baseline",
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "experiments",
]
PROTECTED_FILES = {
    PROJECT_ROOT / "README.md",
    PROJECT_ROOT / "docs" / "project_plan.md",
    PROJECT_ROOT / "requirements.txt",
    PROJECT_ROOT / ".gitignore",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Clean generated experiment artifacts safely.")
    parser.add_argument("--dry-run", action="store_true", help="List files that would be deleted.")
    parser.add_argument("--execute", action="store_true", help="Delete the matched generated files.")
    return parser.parse_args()


def _is_protected(path: Path) -> bool:
    resolved = path.resolve()
    if resolved in PROTECTED_FILES:
        return True
    return any(prefix.resolve() == resolved or prefix.resolve() in resolved.parents for prefix in PROTECTED_PREFIXES)


def _collect_paths() -> list[Path]:
    candidates: dict[Path, None] = {}
    for pattern in SAFE_DELETE_PATHS:
        for path in PROJECT_ROOT.glob(pattern):
            if path.is_file():
                candidates[path.resolve()] = None

    safe_paths: list[Path] = []
    for path in sorted(candidates.keys()):
        if _is_protected(path):
            raise RuntimeError(f"Protected path matched cleanup patterns unexpectedly: {path}")
        safe_paths.append(path)
    return safe_paths


def _ensure_keep_directories() -> None:
    for relative_dir in KEEP_DIRECTORIES:
        (PROJECT_ROOT / relative_dir).mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.dry_run == args.execute:
        print("Specify exactly one of --dry-run or --execute.")
        print("Examples:")
        print("  python experiments/00_clean_generated_artifacts.py --dry-run")
        print("  python experiments/00_clean_generated_artifacts.py --execute")
        return

    targets = _collect_paths()
    mode = "DRY RUN" if args.dry_run else "EXECUTE"
    print(f"[{mode}] Generated artifact cleanup targets:")
    for path in targets:
        print(f"- {path}")

    if args.dry_run:
        print(f"\nDry-run count: {len(targets)} files")
        return

    deleted_paths: list[Path] = []
    for path in targets:
        path.unlink(missing_ok=True)
        deleted_paths.append(path)

    _ensure_keep_directories()
    print(f"\nDeleted {len(deleted_paths)} files:")
    for path in deleted_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
