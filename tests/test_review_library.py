import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from html.parser import HTMLParser
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("review_library", ROOT / "scripts/review_library.py")
library = importlib.util.module_from_spec(spec)
spec.loader.exec_module(library)


class ReadingLibrary(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.repo, self.outputs, self.archive = [base / key for key in ("repo", "outputs", "archive")]
        for folder in (self.repo, self.outputs, self.archive):
            folder.mkdir()
        (self.repo / "notes.md").write_text("# Discussion\n\nA < B.\n")
        (self.outputs / "review.html").write_text("<h1>Original review</h1>")
        self.catalogue = {"reviewed": "2026-10-23", "purpose": "Review alternatives",
            "groups": [{"id": "figure-3", "title": "Figure 3", "context": "Unfrozen",
                "entries": [
                    {"id": "notes", "title": "Notes", "storage": "repo", "path": "notes.md",
                     "status": "Discussion", "description": "Saved discussion"},
                    {"id": "review", "title": "Review", "storage": "outputs", "path": "review.html",
                     "status": "Draft", "description": "Existing review", "sha256": library.digest((self.outputs / "review.html").read_bytes())}]}]}

    def build(self):
        return library.build(self.catalogue, self.outputs, self.archive, self.repo)

    def test_local_check_does_not_require_external_storage(self):
        (self.outputs / "review.html").unlink()
        self.assertEqual(library.check(self.catalogue, self.repo, self.outputs, self.archive)["issues"], [])
        self.assertTrue(library.check(self.catalogue, self.repo, self.outputs, self.archive, external=True)["issues"])

    def test_changed_external_review_cannot_be_published(self):
        (self.outputs / "review.html").write_text("manual edit")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.build()
        self.assertFalse((self.outputs / "START-HERE.html").exists())

    def test_publication_keeps_source_bytes_and_all_generated_links_work(self):
        before = (self.outputs / "review.html").read_bytes()
        report = self.build()
        snapshot = self.outputs / report["snapshot"]
        self.assertEqual((snapshot / "reading/notes.md").read_bytes(), (self.repo / "notes.md").read_bytes())
        self.assertEqual((self.outputs / "review.html").read_bytes(), before)
        class Links(HTMLParser):
            def __init__(self):
                super().__init__()
                self.targets = []
            def handle_starttag(self, tag, attrs):
                if tag == "a" and "href" in dict(attrs):
                    self.targets.append(dict(attrs)["href"])
        for page in [self.outputs / "START-HERE.html", *snapshot.rglob("*.html")]:
            parser = Links()
            parser.feed(page.read_text())
            for target in parser.targets:
                if target.startswith("#"):
                    continue
                self.assertTrue((page.parent / unquote(target)).is_file(), (page, target))

    def test_manual_navigation_edits_are_preserved(self):
        self.build()
        page = self.outputs / "START-HERE.html"
        page.write_text("Author annotation")
        with self.assertRaisesRegex(ValueError, "unrecorded edits"):
            self.build()
        self.assertEqual(page.read_text(), "Author annotation")

    def test_manual_reading_snapshot_edits_are_preserved(self):
        result = self.build()
        notes = self.outputs / result["snapshot"] / "reading/notes.md"
        notes.write_text("Author annotation")
        with self.assertRaisesRegex(ValueError, "snapshot differs"):
            self.build()
        self.assertEqual(notes.read_text(), "Author annotation")

    def test_manual_markdown_index_edits_are_preserved(self):
        self.build()
        path = self.outputs / "REVIEW-LIBRARY.md"
        path.write_text("Author annotation")
        with self.assertRaisesRegex(ValueError, "unrecorded edits"):
            self.build()
        self.assertEqual(path.read_text(), "Author annotation")

    def test_catalogue_cannot_escape_roots_or_omit_external_identity(self):
        entry = self.catalogue["groups"][0]["entries"][1]
        entry["sha256"] = ""
        self.assertTrue(library.check(self.catalogue, self.repo, self.outputs, self.archive)["issues"])
        entry.update(storage="repo", path="../outside.md")
        self.assertTrue(library.check(self.catalogue, self.repo, self.outputs, self.archive)["issues"])


if __name__ == "__main__":
    unittest.main()
