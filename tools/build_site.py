#!/usr/bin/env python3

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

from build_diagram_index import collect, render
from preserve_assets import verify as verify_preservation
from validate_obrain import validate_site, validate_source


def safe_output(root: Path, output: Path) -> Path:
    if output.is_symlink():
        raise ValueError(f"output may not be a symlink: {output}")
    output = output.expanduser().resolve()
    if output == root or output in root.parents or root in output.parents:
        raise ValueError(f"output must be outside the source tree: {output}")
    if output.exists():
        raise ValueError(f"output must not exist (nothing will be overwritten): {output}")
    return output


def prepare_docs(root: Path, docs: Path) -> None:
    for path in root.iterdir():
        if path.is_file() and path.suffix == ".md":
            shutil.copy2(path, docs / path.name)
        elif path.is_dir() and (path / "index.md").is_file():
            shutil.copytree(path, docs / path.name, ignore=shutil.ignore_patterns(".review", "__pycache__"))


def build(root: Path, output: Path, python: str, mode: str) -> list[str]:
    failures = validate_source(root)
    if (root / "diagram-assets.json").is_file():
        try:
            verify_preservation(root)
        except (OSError, ValueError, KeyError) as error:
            failures.append(f"asset preservation check failed: {error}")
    entries, diagram_failures = collect(root)
    failures.extend(diagram_failures)
    if not entries:
        failures.append("no diagrams found; restore the asset archive before building")
    if failures:
        return failures
    with tempfile.TemporaryDirectory(prefix="obrain-build-") as temporary:
        staging = Path(temporary)
        docs = staging / "docs"
        docs.mkdir()
        prepare_docs(root, docs)
        (docs / "diagrams").symlink_to(root / "diagrams", target_is_directory=True)
        (docs / "DIAGRAM-INDEX.md").write_text(render(entries, []), encoding="utf-8")
        built = staging / "built"
        environment = os.environ.copy()
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["XDG_CACHE_HOME"] = str(staging / "cache")
        environment["TMPDIR"] = str(staging)
        script = "\n".join([
            "from mkdocs.commands.build import build",
            "from mkdocs.config import load_config",
            "from pathspec import PathSpec",
            "import sys",
            "config = load_config(config_file=sys.argv[1], docs_dir=sys.argv[2], site_dir=sys.argv[3])",
            "config.exclude_docs = (config.exclude_docs or PathSpec([])) + PathSpec.from_lines('gitwildmatch', ['diagrams/**'])",
            "build(config)",
        ])
        result = subprocess.run([python, "-c", script, str(root / "mkdocs.yml"), str(docs), str(built)], cwd=staging, env=environment)
        if result.returncode:
            return [f"mkdocs build failed with exit code {result.returncode}"]
        target = built / "diagrams"
        if target.exists():
            return ["MkDocs unexpectedly copied diagrams; check exclude_docs before publishing"]
        if mode == "portable":
            shutil.copytree(root / "diagrams", target, ignore=shutil.ignore_patterns("*.json", ".*"))
        else:
            target.symlink_to(root / "diagrams", target_is_directory=True)
        failures = validate_site(built)
        if failures:
            return failures
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(built, output, symlinks=mode == "preview")
    return []


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("portable", "preview"), default="portable")
    parser.add_argument("--python", default=os.environ.get("OBRAIN_PYTHON", sys.executable))
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        output = safe_output(root, args.output)
        python = os.path.abspath(os.path.expanduser(args.python)) if os.sep in args.python else shutil.which(args.python)
        if not python:
            raise ValueError(f"Python executable not found: {args.python}")
        failures = build(root, output, python, args.mode)
    except (OSError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    for failure in failures:
        print(failure, file=sys.stderr)
    if failures:
        return 1
    print(f"built {args.mode} site at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
