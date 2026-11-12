#!/usr/bin/env python3

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path


TYPE_ORDER = {
    "architecture": 0,
    "dataflow": 1,
    "sequence": 2,
    "workflow": 3,
    "lifecycle": 4,
    "timeline": 5,
    "waterfall": 6,
    "tree": 7,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[str]:
    manifest = root / "diagram-assets.json"
    if not manifest.is_file():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        records = data if isinstance(data, list) else data.get("assets", data.get("files"))
        if isinstance(data, dict) and data.get("version") == 1 and "artifacts" in data:
            records = data["artifacts"] + data.get("specifications", []) + data.get("receipts", [])
        if not isinstance(records, list):
            return ["diagram-assets.json: expected version 1 artifacts or assets/files list"]
        failures = []
        for record in records:
            value = record.get("path", record.get("file"))
            expected = record.get("sha256")
            if not value or not expected:
                failures.append("diagram-assets.json: record requires path and sha256")
                continue
            artifact = (root / value).resolve()
            if Path(value).is_absolute() or not artifact.is_relative_to(root):
                failures.append(f"diagram-assets.json: path escapes repository: {value}")
            elif not artifact.is_file():
                failures.append(f"diagram-assets.json: missing artifact {value}")
            elif "bytes" in record and artifact.stat().st_size != record["bytes"]:
                failures.append(f"diagram-assets.json: size mismatch for {value}")
            elif sha256(artifact) != expected:
                failures.append(f"diagram-assets.json: hash mismatch for {value}")
        if isinstance(data, dict) and "artifacts" in data:
            expected_paths = {record["path"] for record in data["artifacts"]}
            actual_paths = {path.relative_to(root).as_posix() for path in (root / "diagrams").glob("*.html")}
            failures.extend(f"diagram-assets.json: unregistered artifact {path}" for path in sorted(actual_paths - expected_paths))
        return failures
    except (OSError, ValueError, TypeError, AttributeError) as error:
        return [f"diagram-assets.json: {error}"]


def collect(root: Path) -> tuple[list[tuple], list[str]]:
    receipts = sorted((root / "diagrams").glob("*.finalize-summary.json"))
    artifacts = {path.name: path for path in (root / "diagrams").glob("*.html")}
    entries = []
    failures = []
    if not artifacts:
        failures.append("no diagram HTML files found; restore the preservation archive")
    seen = set()
    for receipt in receipts:
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{receipt.name}: {error}")
            continue

        if not isinstance(data, dict) or not isinstance(data.get("artifact", {}), dict):
            failures.append(f"{receipt.name}: malformed receipt")
            continue
        artifact_path = Path(str(data.get("artifact", {}).get("path") or ""))
        artifact_name = artifact_path.name
        if not artifact_name:
            artifact_name = receipt.name.replace(".finalize-summary.json", ".html")
        artifact = root / "diagrams" / artifact_name
        if not artifact.is_file():
            failures.append(f"{receipt.name}: missing artifact {artifact_name}")
            continue
        failure_count = len(failures)
        if artifact_name in seen:
            failures.append(f"{receipt.name}: duplicate receipt for {artifact_name}")
        if not data.get("ok") or data.get("status") != "pass":
            failures.append(f"{receipt.name}: status={data.get('status')}, ok={data.get('ok')}")
        expected = data.get("artifact", {}).get("sha256")
        if not expected or sha256(artifact) != expected:
            failures.append(f"{receipt.name}: missing or mismatched artifact sha256")
        seen.add(artifact_name)
        name = artifact.stem
        diagram_type = data.get("type", "unknown")
        quality = data.get("quality", "unknown")
        status = "receipt-verified" if len(failures) == failure_count else "receipt-failed"
        entries.append((name, diagram_type, quality, artifact_name, status))
    for artifact_name, artifact in sorted(artifacts.items()):
        if artifact_name not in seen:
            diagram_type = artifact.stem.rsplit(".", 1)[-1] if "." in artifact.stem else "unknown"
            entries.append((artifact.stem, diagram_type, "unknown", artifact_name, "legacy-unverified"))
    failures.extend(verify_manifest(root))
    return entries, failures


def render(entries: list[tuple], failures: list[str]) -> str:
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry[1]].append(entry)

    lines = [
        "# Diagram Index",
        "",
        "Automatic index of every `diagrams/*.html`; receipts add validation status.",
        "Regenerate with `python3 tools/build_diagram_index.py`.",
        "Legacy diagrams without receipts are explicitly marked `legacy-unverified`.",
        "",
    ]

    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")

    for diagram_type in sorted(grouped, key=lambda value: (TYPE_ORDER.get(value, 99), value)):
        lines.extend([f"## {diagram_type.title()} ({len(grouped[diagram_type])})", ""])
        lines.append("| Diagram | Quality | Validation |")
        lines.append("|---|---|---|")
        for name, _diagram_type, quality, artifact_name, status in sorted(grouped[diagram_type]):
            lines.append(f"| [{name}](diagrams/{artifact_name}) | {quality} | {status} |")
        lines.append("")

    lines.extend([
        "## Totals",
        "",
        f"- Diagrams: {len(entries)}",
        f"- Legacy unverified: {sum(entry[4] == 'legacy-unverified' for entry in entries)}",
        f"- Failures: {len(failures)}",
        "",
    ])

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--output", type=Path, default=Path("DIAGRAM-INDEX.md"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    entries, failures = collect(root)
    output = root / args.output
    content = render(entries, failures)
    if args.check:
        if not output.is_file() or output.read_text(encoding="utf-8") != content:
            failures.append(f"outdated index: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
        print(f"wrote {output}: {len(entries)} diagrams, {len(failures)} failures")
    for failure in failures:
        print(failure, file=sys.stderr)
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
