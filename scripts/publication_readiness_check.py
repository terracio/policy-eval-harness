from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = (
    Path("LICENSE"),
    Path("docs/publication_readiness.md"),
    Path("docs/releases/v0.1.0.md"),
)

FORBIDDEN_PATTERNS = (
    "PENGU",
    "Bybit",
    "Binance",
    "live_v1",
    "GOOGLE_API_KEY",
    "OPENAI_API_KEY",
    "BYBIT_",
    "BINANCE_",
    "service_account",
    ".env",
    "postgresql:///",
    "artifacts/ml",
)

TEXT_EXTENSIONS = {
    "",
    ".md",
    ".py",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".txt",
    ".csv",
    ".gitignore",
}

AUDIT_FILES = {
    Path("docs/publication_readiness.md"),
    Path("scripts/publication_readiness_check.py"),
}

MARKDOWN_FILES = (
    Path("README.md"),
    Path("docs/methodology.md"),
    Path("docs/case_study.md"),
    Path("docs/design_principles.md"),
    Path("docs/publication_readiness.md"),
    Path("docs/releases/v0.1.0.md"),
    Path("examples/core_demo/README.md"),
    Path("examples/label_compare/README.md"),
    Path("examples/ablation_2x2/README.md"),
)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]

    _check_required_files(repo_root)
    _check_versions(repo_root)
    _check_markdown_links(repo_root)
    _check_forbidden_patterns(repo_root)
    _run_contract_verifier(repo_root)

    print("Publication readiness checks completed successfully.")
    return 0


def _check_required_files(repo_root: Path) -> None:
    for relative_path in REQUIRED_FILES:
        path = repo_root / relative_path
        if not path.exists():
            raise RuntimeError(f"Missing required release artifact: {relative_path}")


def _check_versions(repo_root: Path) -> None:
    pyproject_text = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    package_text = (repo_root / "src" / "policy_eval_harness" / "__init__.py").read_text(encoding="utf-8")

    if 'version = "0.1.0"' not in pyproject_text:
        raise RuntimeError("pyproject.toml is not pinned to version 0.1.0")
    if '__version__ = "0.1.0"' not in package_text:
        raise RuntimeError("package __version__ is not pinned to 0.1.0")


def _check_markdown_links(repo_root: Path) -> None:
    link_pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    for relative_path in MARKDOWN_FILES:
        path = repo_root / relative_path
        text = path.read_text(encoding="utf-8")
        for match in link_pattern.finditer(text):
            target = match.group(1).strip()
            if not target or target.startswith(("http://", "https://", "mailto:", "#", "app://", "/")):
                continue
            resolved = (path.parent / target).resolve()
            if not resolved.exists():
                raise RuntimeError(f"Broken markdown link in {relative_path}: {target}")


def _check_forbidden_patterns(repo_root: Path) -> None:
    excluded_dirs = {".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache", "src/policy_eval_harness.egg-info"}

    for path in sorted(repo_root.rglob("*")):
        if path.is_dir():
            continue

        relative_path = path.relative_to(repo_root)
        if any(part in excluded_dirs for part in relative_path.parts):
            continue

        if relative_path in AUDIT_FILES:
            continue

        path_string = relative_path.as_posix()
        for pattern in FORBIDDEN_PATTERNS:
            if pattern in path_string:
                raise RuntimeError(f"Forbidden pattern {pattern!r} found in path {relative_path}")

        if path.suffix.lower() not in TEXT_EXTENSIONS and path.name not in {".gitignore"}:
            continue

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue

        for pattern in FORBIDDEN_PATTERNS:
            if pattern in text:
                raise RuntimeError(f"Forbidden pattern {pattern!r} found in {relative_path}")


def _run_contract_verifier(repo_root: Path) -> None:
    policy_eval = _resolve_policy_eval()
    subprocess.run([policy_eval, "--help"], cwd=repo_root, check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "verify_public_contracts.py")],
        cwd=repo_root,
        check=True,
    )


def _resolve_policy_eval() -> str:
    candidate = shutil.which("policy-eval")
    if candidate:
        return candidate

    fallback = Path(sys.executable).parent / "policy-eval"
    if fallback.exists():
        return str(fallback)

    raise RuntimeError("Could not find the installed `policy-eval` executable.")


if __name__ == "__main__":
    raise SystemExit(main())
