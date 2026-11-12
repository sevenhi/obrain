#!/usr/bin/env python3
"""Protect local diagram assets and verify their portable restoration contract."""

import argparse
import hashlib
import json
import shutil
import sys
import tarfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
MANIFEST = "diagram-assets.json"


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def record(path, root):
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": digest(path),
    }


def read_manifest(root):
    data = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if data.get("version") != 1 or not data.get("artifacts"):
        raise ValueError("missing or unsupported asset manifest")
    return data


def safe_path(root, relative):
    path = root / relative
    if Path(relative).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"unsafe manifest path: {relative}")
    return path


def verify(root):
    data = read_manifest(root)
    failures = []
    for section in ("artifacts", "specifications", "receipts"):
        for entry in data.get(section, []):
            path = safe_path(root, entry["path"])
            if not path.is_file():
                failures.append(f"missing {entry['path']}")
            elif path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
                failures.append(f"changed {entry['path']}")
    expected = {entry["path"] for entry in data["artifacts"]}
    actual = {path.relative_to(root).as_posix() for path in (root / "diagrams").glob("*.html")}
    failures.extend(f"unregistered {name}" for name in sorted(actual - expected))
    if failures:
        raise ValueError("\n".join(failures))
    return data


def initialize(root):
    if (root / MANIFEST).exists():
        raise ValueError("manifest already exists; refusing to overwrite the preservation baseline")
    originals = sorted((root / ".archify").glob("**/candidate.json"))
    for original in originals:
        destination = root / "diagram-specs" / original.relative_to(root / ".archify")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != digest(original):
            raise ValueError(f"conflicting preserved specification: {destination}")
        shutil.copy2(original, destination)
    data = manifest_data(root)
    if not data["artifacts"] or not data["specifications"]:
        raise ValueError("no assets or specifications to preserve")
    (root / MANIFEST).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return verify(root)


def manifest_data(root):
    return {
        "version": 1,
        "description": "Preservation baseline, not a claim of semantic correctness or deterministic regeneration.",
        "artifacts": [record(path, root) for path in sorted((root / "diagrams").glob("*.html"))],
        "specifications": [record(path, root) for path in sorted((root / "diagram-specs").glob("**/candidate.json"))],
        "receipts": [record(path, root) for path in sorted((root / "diagrams").glob("*.finalize-summary.json"))],
    }


def verify_archive(archive, entries):
    with tarfile.open(archive, "r:gz") as source:
        members = {member.name: member for member in source.getmembers()}
        for entry in entries:
            member = members.get(entry["path"])
            if member is None or not member.isfile() or member.size != entry["bytes"]:
                raise ValueError(f"backup missing asset: {entry['path']}")
            checksum = hashlib.sha256()
            with source.extractfile(member) as contents:
                for block in iter(lambda: contents.read(1024 * 1024), b""):
                    checksum.update(block)
            if checksum.hexdigest() != entry["sha256"]:
                raise ValueError(f"backup hash mismatch: {entry['path']}")


def backup(root, destination):
    data = verify(root)
    destination = destination.resolve()
    if destination.is_relative_to(root.resolve()):
        raise ValueError("backup directory must be outside the knowledge repository")
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "diagram-preservation.tar.gz"
    entries = data["artifacts"] + data["specifications"] + data["receipts"]
    if archive.exists():
        raise ValueError(f"backup already exists: {archive}")
    with tarfile.open(archive, "x:gz") as output:
        output.add(root / MANIFEST, arcname=MANIFEST)
        for entry in entries:
            output.add(safe_path(root, entry["path"]), arcname=entry["path"], recursive=False)
    verify_archive(archive, entries)
    checksum_path = archive.with_name(archive.name + ".sha256")
    checksum_path.write_text(f"{digest(archive)}  {archive.name}\n", encoding="utf-8")
    print(f"Verified backup: {archive}")


def refresh(root, archive):
    previous = read_manifest(root)
    entries = previous["artifacts"] + previous["specifications"] + previous["receipts"]
    verify_archive(archive, entries)
    manifest = root / MANIFEST
    history = root / "diagram-specs" / "manifests" / (digest(manifest) + ".json")
    history.parent.mkdir(parents=True, exist_ok=True)
    if not history.exists():
        shutil.copy2(manifest, history)
    data = manifest_data(root)
    if {entry["path"] for entry in data["artifacts"]} != {entry["path"] for entry in previous["artifacts"]}:
        raise ValueError("refresh cannot add or remove diagram URLs")
    temporary = root / (MANIFEST + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest)
    return verify(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "check", "backup", "check-backup", "refresh"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--archive", type=Path)
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    try:
        if arguments.command == "initialize":
            data = initialize(root)
        elif arguments.command == "backup":
            if arguments.destination is None:
                parser.error("backup requires --destination")
            backup(root, arguments.destination)
            return 0
        elif arguments.command == "refresh":
            if arguments.archive is None:
                parser.error("refresh requires --archive containing the entire previous baseline")
            data = refresh(root, arguments.archive)
        elif arguments.command == "check-backup":
            if arguments.archive is None:
                parser.error("check-backup requires --archive")
            data = read_manifest(root)
            entries = data["artifacts"] + data["specifications"] + data["receipts"]
            verify_archive(arguments.archive, entries)
            print("Backup members and hashes verified")
            return 0
        else:
            data = verify(root)
        print(f"Verified {len(data['artifacts'])} diagrams, {len(data['specifications'])} specifications, {len(data['receipts'])} receipts")
        return 0
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        print(f"Asset preservation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
