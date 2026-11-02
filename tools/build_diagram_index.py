#!/usr/bin/env python3

import argparse
import json
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="DIAGRAM-INDEX.md")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    receipts = sorted((root / "diagrams").glob("*.finalize-summary.json"))

    entries = []
    failures = []
    for receipt in receipts:
        try:
            data = json.loads(receipt.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            failures.append(f"{receipt.name}: {error}")
            continue

        artifact_path = Path(data.get("artifact", {}).get("path", ""))
        artifact_name = artifact_path.name
        if not artifact_name:
            artifact_name = receipt.name.replace(".finalize-summary.json", ".html")
        artifact = root / "diagrams" / artifact_name
        if not artifact.is_file():
            failures.append(f"{receipt.name}: missing artifact {artifact_name}")
            continue
        if not data.get("ok") or data.get("status") != "pass":
            failures.append(f"{receipt.name}: status={data.get('status')}, ok={data.get('ok')}")

        name = receipt.name.replace(".finalize-summary.json", "")
        diagram_type = data.get("type", "unknown")
        quality = data.get("quality", "unknown")
        entries.append((name, diagram_type, quality, artifact_name))

    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry[1]].append(entry)

    lines = [
        "# Diagram Index",
        "",
        "Automatic index generated from `diagrams/*.finalize-summary.json`.",
        "Regenerate with `python3 tools/build_diagram_index.py`.",
        "",
    ]

    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")

    for diagram_type in sorted(grouped, key=lambda value: (TYPE_ORDER.get(value, 99), value)):
        lines.extend([f"## {diagram_type.title()} ({len(grouped[diagram_type])})", ""])
        lines.append("| Diagram | Quality |")
        lines.append("|---|---|")
        for name, _diagram_type, quality, artifact_name in grouped[diagram_type]:
            lines.append(f"| [{name}](diagrams/{artifact_name}) | {quality} |")
        lines.append("")

    lines.extend([
        "## Totals",
        "",
        f"- Diagrams: {len(entries)}",
        f"- Failures: {len(failures)}",
        "",
    ])

    output = root / args.output
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {output.relative_to(root)}: {len(entries)} diagrams, {len(failures)} failures")


if __name__ == "__main__":
    main()
