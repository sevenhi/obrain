#!/usr/bin/env python3

import argparse
from html.parser import HTMLParser
import os
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import markdown
import yaml


ROOT_EXEMPT = {"README.md", "DEPLOY.md", "DIAGRAM-INDEX.md", "OPTIMIZATION-SUMMARY.md"}
FENCE_RE = re.compile(r"(?ms)^[ \t]*(`{3,}|~{3,})[^\n]*\n.*?^[ \t]*\1[ \t]*$")
TLDR_RE = re.compile(r"(?mi)^(?:#{2,3}\s*(?:\d+[.、]?\s*)?(?:TL;DR|速览(?:\([^)\n]*\))?)|>\s*(?:\*\*)?\s*(?:TL;DR|速览))")
ARTICLE_TYPES = {"concept", "case", "checklist", "howto", "reference", "domain", "comparison"}


def markdown_files(root: Path) -> list[Path]:
    roots = [path for path in root.iterdir() if path.is_file() and path.suffix == ".md"]
    domains = [path for path in root.iterdir() if path.is_dir() and (path / "index.md").is_file()]
    return sorted(roots + [path for domain in domains for path in domain.rglob("*.md") if ".review" not in path.parts])


class PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids = set()
        self.links = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(values["id"])
        if tag == "a" and values.get("name"):
            self.ids.add(values["name"])
        attribute = "href" if tag in {"a", "link"} else "src" if tag in {"iframe", "script", "img", "source"} else None
        if attribute and values.get(attribute):
            self.links.append((tag, values[attribute]))


def split_frontmatter(path: Path, text: str) -> tuple[dict | None, str, list[str]]:
    match = re.match(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", text, re.S)
    if not match:
        errors = [f"{path}: unterminated YAML frontmatter"] if text.startswith("---\n") else []
        return None, text, errors
    try:
        data = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        return None, text[match.end():], [f"{path}: invalid YAML frontmatter: {error}"]
    if not isinstance(data, dict):
        return None, text[match.end():], [f"{path}: YAML frontmatter must be a mapping"]
    return data, text[match.end():], []


def is_article(root: Path, path: Path) -> bool:
    rel = path.relative_to(root)
    return len(rel.parts) > 1 and path.name not in {"index.md", "references.md"} and "notes" not in rel.parts


def is_external(target: str) -> bool:
    return bool(urlsplit(target).scheme) or target.startswith("//")


def source_links(body: str) -> list[str]:
    parser = PageParser()
    parser.feed(markdown.markdown(body, extensions=["fenced_code", "tables"]))
    return [target for tag, target in parser.links if tag in {"a", "img"}]


def validate_source(root: Path) -> list[str]:
    failures = []
    files = markdown_files(root)
    for path in files:
        rel = path.relative_to(root)
        metadata, body, errors = split_frontmatter(rel, path.read_text(encoding="utf-8"))
        failures.extend(errors)
        exempt = str(rel) in ROOT_EXEMPT or path.name == "README.md" or "notes" in rel.parts
        if metadata is None and not exempt:
            failures.append(f"{rel}: missing YAML frontmatter")
        if metadata is not None and not exempt:
            fields = ["title", "type", "tags"] + (["source", "date"] if is_article(root, path) else [])
            for field in fields:
                if metadata.get(field) in (None, "", []):
                    failures.append(f"{rel}: missing required field {field}")
            if "tags" in metadata and not isinstance(metadata["tags"], list):
                failures.append(f"{rel}: tags must be a YAML list")
            if not isinstance(metadata.get("type"), str) or metadata["type"] not in ARTICLE_TYPES:
                failures.append(f"{rel}: unsupported type {metadata.get('type')}")
        if is_article(root, path) and not TLDR_RE.search(FENCE_RE.sub("", body)):
            failures.append(f"{rel}: missing 速览 (TL;DR) section")
        for target in source_links(body):
            if is_external(target):
                continue
            raw_path = unquote(urlsplit(target).path)
            if not raw_path:
                continue
            candidate = (root / raw_path.lstrip("/") if raw_path.startswith("/") else path.parent / raw_path).resolve()
            if candidate.is_dir():
                candidate = candidate / ("index.md" if (candidate / "index.md").is_file() else "README.md")
            if not candidate.is_file():
                failures.append(f"{rel}: missing source target {target}")
    for domain in sorted(path for path in root.iterdir() if path.is_dir() and (path / "index.md").is_file()):
        indexed = set()
        for target in source_links((domain / "index.md").read_text(encoding="utf-8")):
            if not is_external(target):
                indexed.add((domain / unquote(urlsplit(target).path)).resolve())
        for path in sorted(domain.rglob("*.md")):
            if path.name in {"index.md", "references.md"} or any(part in {"notes", ".review"} for part in path.relative_to(domain).parts):
                continue
            if path.resolve() not in indexed:
                failures.append(f"{domain.name}/index.md: does not index {path.relative_to(domain)}")
    return failures


def validate_site(site: Path) -> list[str]:
    failures = []
    pages = {}

    def page_parser(path: Path) -> PageParser:
        if path not in pages:
            parser = PageParser()
            parser.feed(path.read_text(encoding="utf-8", errors="replace"))
            pages[path] = parser
        return pages[path]

    paths = [path for path in site.rglob("*.html") if "diagrams" not in path.relative_to(site).parts]
    for path in paths:
        for _, target in page_parser(path).links:
            if is_external(target):
                continue
            parsed = urlsplit(target)
            raw_path = unquote(parsed.path)
            candidate = site / raw_path.lstrip("/") if raw_path.startswith("/") else path.parent / raw_path
            if not raw_path:
                candidate = path
            elif raw_path.endswith("/") or candidate.is_dir():
                candidate = candidate / "index.html"
            candidate = Path(os.path.abspath(candidate))
            if not candidate.is_relative_to(site):
                failures.append(f"{path.relative_to(site)}: built target escapes site root: {target}")
                continue
            if not candidate.is_file():
                failures.append(f"{path.relative_to(site)}: missing built target {target}")
                continue
            if parsed.fragment and candidate.suffix == ".html":
                if unquote(parsed.fragment) not in page_parser(candidate).ids:
                    failures.append(f"{path.relative_to(site)}: missing anchor {target}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--site", type=Path)
    args = parser.parse_args()
    failures = validate_source(args.root.resolve())
    if args.site:
        failures.extend(validate_site(args.site.resolve()))
    for failure in failures:
        print(failure, file=sys.stderr)
    print(f"validation {'failed' if failures else 'passed'}: {len(failures)} issue(s)")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
