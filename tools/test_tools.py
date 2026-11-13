import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from build_diagram_index import collect, render, verify_manifest
from build_site import build, safe_output
from validate_obrain import validate_site, validate_source


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="obrain-tools-test-")
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.root = self.base / "repo"
        self.root.mkdir()

    def write(self, relative, text):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def article(self, metadata="source: [web]\ndate: 2026-11", body="## 速览\nSummary."):
        self.write("topic/index.md", "---\ntitle: Topic\ntype: domain\ntags: [x]\n---\n[entry](concepts/entry.md)\n")
        return self.write("topic/concepts/entry.md", f"---\ntitle: Entry\ntype: concept\ntags: [x]\n{metadata}\n---\n{body}\n")

    def test_source_ignores_fenced_inline_and_indented_examples(self):
        self.article(body="## TL;DR\nSummary.\n\n```md\n[bad](missing.md)\n```\n\n~~~md\n[bad](missing.md)\n~~~\n\n    [bad](missing.md)\n\n`[bad](missing.md)`\n")
        self.assertEqual(validate_source(self.root), [])

    def test_source_requires_both_source_and_date(self):
        self.article(metadata="date: 2026-11")
        self.assertTrue(any("field source" in issue for issue in validate_source(self.root)))
        self.article(metadata="source: [web]")
        self.assertTrue(any("field date" in issue for issue in validate_source(self.root)))

    def test_source_parses_invalid_yaml(self):
        self.article(metadata="source: [unterminated")
        self.assertTrue(any("invalid YAML" in issue for issue in validate_source(self.root)))

    def test_source_rejects_unknown_type(self):
        path = self.article()
        path.write_text(path.read_text().replace("type: concept", "type: implementation"))
        self.assertTrue(any("unsupported type" in issue for issue in validate_source(self.root)))

    def test_source_requires_tldr_outside_code(self):
        self.article(body="```md\n## 速览\n```")
        self.assertTrue(any("missing 速览" in issue for issue in validate_source(self.root)))

    def test_source_notes_exemption(self):
        self.article()
        self.write("topic/notes/raw.md", "No metadata and no summary.")
        self.assertEqual(validate_source(self.root), [])

    def test_source_requires_real_domain_index_link(self):
        self.article()
        self.write("topic/index.md", "---\ntitle: Topic\ntype: domain\ntags: [x]\n---\n`concepts/entry.md`\n")
        self.assertTrue(any("does not index" in issue for issue in validate_source(self.root)))

    def test_source_missing_link(self):
        self.article(body="## 速览\nSummary. [bad](missing.md)")
        self.assertTrue(any("missing source target" in issue for issue in validate_source(self.root)))

    def test_built_anchor_iframe_and_directory_index(self):
        self.write("site/diagrams/x.html", "<h1 id='diagram'>x</h1>")
        self.write("site/topic/index.html", "<h1 id='中文'>Topic</h1>")
        self.write("site/index.html", "<a href='topic/#%E4%B8%AD%E6%96%87'>x</a><iframe src='diagrams/x.html'></iframe>")
        self.assertEqual(validate_site(self.root / "site"), [])
        self.write("site/index.html", "<a href='topic/#missing'>x</a><iframe src='missing.html'></iframe>")
        failures = validate_site(self.root / "site")
        self.assertTrue(any("missing anchor" in issue for issue in failures))
        self.assertTrue(any("missing built target" in issue for issue in failures))

    def test_escaped_html_code_is_not_a_link(self):
        self.write("site/index.html", "<pre>&lt;a href='missing.md'&gt;x&lt;/a&gt;</pre>")
        self.assertEqual(validate_site(self.root / "site"), [])

    def receipt(self, status="pass", digest=None):
        artifact = self.write("diagrams/x.workflow.html", "x")
        data = {"ok": status == "pass", "status": status, "type": "workflow", "quality": "showcase", "artifact": {"path": str(artifact), "sha256": digest or hashlib.sha256(b"x").hexdigest()}}
        self.write("diagrams/x.workflow.finalize-summary.json", json.dumps(data))
        return artifact

    def test_diagram_index_marks_all_legacy(self):
        self.receipt()
        self.write("diagrams/legacy.tree.html", "legacy")
        entries, failures = collect(self.root)
        self.assertEqual(failures, [])
        self.assertEqual(len(entries), 2)
        self.assertIn("legacy-unverified", render(entries, failures))

    def test_diagram_index_rejects_status_hash_and_missing_html(self):
        self.receipt(status="fail")
        self.assertTrue(collect(self.root)[1])
        self.receipt(digest="bad")
        self.assertTrue(collect(self.root)[1])
        self.receipt().unlink()
        self.assertTrue(any("missing artifact" in issue for issue in collect(self.root)[1]))

    def test_manifest_hash_and_inventory(self):
        self.receipt()
        self.write("diagram-assets.json", json.dumps({"version": 1, "artifacts": [{"path": "diagrams/x.workflow.html", "sha256": "bad"}]}))
        self.assertTrue(any("hash mismatch" in issue for issue in verify_manifest(self.root)))
        self.write("diagrams/unregistered.html", "extra")
        self.assertTrue(any("unregistered" in issue for issue in verify_manifest(self.root)))

    def test_index_external_output_and_readonly_check(self):
        self.receipt()
        output = self.base / "external-index.md"
        command = [sys.executable, str(Path(__file__).with_name("build_diagram_index.py")), "--root", str(self.root), "--output", str(output)]
        self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
        before = output.stat().st_mtime_ns
        self.assertEqual(subprocess.run(command + ["--check"], capture_output=True).returncode, 0)
        self.assertEqual(output.stat().st_mtime_ns, before)
        output.write_text("stale", encoding="utf-8")
        self.assertNotEqual(subprocess.run(command + ["--check"], capture_output=True).returncode, 0)
        self.assertEqual(output.read_text(), "stale")

    def test_safe_output_rejects_source_existing_and_symlink(self):
        for output in (self.root, self.root / "site", self.base):
            with self.assertRaises(ValueError):
                safe_output(self.root, output)
        output = self.base / "export"
        self.assertEqual(safe_output(self.root, output), output)
        output.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            safe_output(self.root, output)

    def test_clean_checkout_build_portable_and_preview_without_staging(self):
        self.article()
        self.write("README.md", "# Home\n[Topic](topic/index.md)\n")
        self.write("mkdocs.yml", "site_name: Fixture\ndocs_dir: .site\nsite_dir: site\ntheme:\n  name: mkdocs\n")
        self.write("diagrams/x.workflow.html", "<html><body>diagram</body></html>")
        self.write("diagrams/.archify-delivery-test/stale.html", "stale render")
        self.assertFalse((self.root / ".site").exists())
        portable = self.base / "portable"
        self.assertEqual(build(self.root, portable, sys.executable, "portable"), [])
        self.assertTrue((portable / "index.html").is_file())
        self.assertFalse((portable / "diagrams").is_symlink())
        self.assertFalse((portable / "diagrams" / ".archify-delivery-test").exists())
        preview = self.base / "preview"
        self.assertEqual(build(self.root, preview, sys.executable, "preview"), [])
        self.assertTrue((preview / "diagrams").is_symlink())
        self.assertFalse((self.root / ".site").exists())
        self.assertFalse((self.root / "site").exists())
        self.assertFalse((self.root / "DIAGRAM-INDEX.md").exists())
        command = [sys.executable, str(Path(__file__).with_name("build_site.py")), "--root", str(self.root), "--output", str(self.base / "cli-portable")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
