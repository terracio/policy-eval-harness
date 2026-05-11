from __future__ import annotations

import argparse
import compileall
import subprocess
import sys
from pathlib import Path

MAX_PYTHON_LINES = 350
EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "src/policy_eval_harness.egg-info",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local quality gates used by CI.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip the unittest suite.")
    parser.add_argument("--skip-contracts", action="store_true", help="Skip public contract CLI verification.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    _run([sys.executable, "-m", "ruff", "check", "."], cwd=repo_root)
    _run([sys.executable, "-m", "mypy", "src/policy_eval_harness"], cwd=repo_root)
    _check_python_line_counts(repo_root)
    _compile_python(repo_root)

    if not args.skip_tests:
        _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."], cwd=repo_root)
    if not args.skip_contracts:
        _run([sys.executable, "scripts/verify_public_contracts.py"], cwd=repo_root)

    print("Quality gates completed successfully.")
    return 0


def _run(command: list[str], *, cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def _check_python_line_counts(repo_root: Path) -> None:
    offenders: list[tuple[int, Path]] = []
    for path in sorted(repo_root.rglob("*.py")):
        relative_path = path.relative_to(repo_root)
        if _is_excluded(relative_path):
            continue
        line_count = _line_count(path)
        if line_count > MAX_PYTHON_LINES:
            offenders.append((line_count, relative_path))
    if offenders:
        details = "\n".join(f"{line_count:4d} {path}" for line_count, path in offenders)
        raise RuntimeError(f"Python files exceed {MAX_PYTHON_LINES} lines:\n{details}")


def _compile_python(repo_root: Path) -> None:
    for relative_path in (Path("src"), Path("tests"), Path("scripts")):
        if not compileall.compile_dir(repo_root / relative_path, quiet=1):
            raise RuntimeError(f"Python bytecode compilation failed under {relative_path}.")


def _is_excluded(relative_path: Path) -> bool:
    parts = set(relative_path.parts)
    return any(excluded in parts or relative_path.as_posix().startswith(excluded + "/") for excluded in EXCLUDED_DIRS)


def _line_count(path: Path) -> int:
    return len(path.read_text(encoding="utf-8").splitlines())


if __name__ == "__main__":
    raise SystemExit(main())
