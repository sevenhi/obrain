import importlib.util
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("preserve_assets", Path(__file__).with_name("preserve_assets.py"))
ASSETS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ASSETS)


class PreservationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "knowledge"
        (self.root / "diagrams").mkdir(parents=True)
        (self.root / "diagrams" / "example.html").write_text("<html>original</html>")
        (self.root / ".archify" / "example").mkdir(parents=True)
        (self.root / ".archify" / "example" / "candidate.json").write_text('{"title":"original"}')
        ASSETS.initialize(self.root)

    def test_initialization_preserves_source_and_refuses_overwrite(self):
        self.assertTrue((self.root / "diagram-specs" / "example" / "candidate.json").is_file())
        with self.assertRaises(ValueError):
            ASSETS.initialize(self.root)

    def test_changed_artifact_is_detected(self):
        (self.root / "diagrams" / "example.html").write_text("changed")
        with self.assertRaisesRegex(ValueError, "changed"):
            ASSETS.verify(self.root)

    def test_unregistered_artifact_is_detected(self):
        (self.root / "diagrams" / "extra.html").write_text("extra")
        with self.assertRaisesRegex(ValueError, "unregistered"):
            ASSETS.verify(self.root)

    def test_backup_and_guarded_refresh(self):
        destination = Path(self.temporary.name) / "backup"
        ASSETS.backup(self.root, destination)
        archive = destination / "diagram-preservation.tar.gz"
        (self.root / "diagrams" / "example.html").write_text("corrected")
        ASSETS.refresh(self.root, archive)
        ASSETS.verify(self.root)
        self.assertEqual(len(list((self.root / "diagram-specs" / "manifests").glob("*.json"))), 1)
        with self.assertRaises(ValueError):
            ASSETS.refresh(self.root, archive)

    def test_backup_refuses_inside_repository(self):
        with self.assertRaises(ValueError):
            ASSETS.backup(self.root, self.root / "backup")

    def test_path_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            ASSETS.safe_path(self.root, "../outside")

    def test_backup_is_not_overwritten(self):
        destination = Path(self.temporary.name) / "backup"
        ASSETS.backup(self.root, destination)
        with self.assertRaises(ValueError):
            ASSETS.backup(self.root, destination)

    def test_refresh_cannot_remove_diagram(self):
        destination = Path(self.temporary.name) / "backup"
        ASSETS.backup(self.root, destination)
        (self.root / "diagrams" / "example.html").unlink()
        with self.assertRaisesRegex(ValueError, "cannot add or remove"):
            ASSETS.refresh(self.root, destination / "diagram-preservation.tar.gz")


if __name__ == "__main__":
    unittest.main()
