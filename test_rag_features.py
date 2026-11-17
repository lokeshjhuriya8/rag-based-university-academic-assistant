"""Offline checks for document-management and quiz validation helpers."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import rag


class FakeCollection:
    def __init__(self, records: list[dict[str, object]]):
        self.records = records

    def get(self, where=None, include=None):
        records = self.records
        if where:
            key, value = next(iter(where.items()))
            records = [record for record in records if record["metadata"].get(key) == value]
        result = {"ids": [record["id"] for record in records]}
        if include and "metadatas" in include:
            result["metadatas"] = [record["metadata"] for record in records]
        if include and "documents" in include:
            result["documents"] = [record["document"] for record in records]
        return result

    def delete(self, ids):
        self.records[:] = [record for record in self.records if record["id"] not in ids]


class RagFeatureTests(unittest.TestCase):
    def test_safe_document_path_rejects_path_traversal(self) -> None:
        with self.assertRaises(ValueError):
            rag.safe_document_path("..\\outside.pdf")
        with self.assertRaises(ValueError):
            rag.safe_document_path("not-a-pdf.txt")

    def test_delete_document_removes_only_selected_source(self) -> None:
        records = [
            {"id": "one", "metadata": {"source": "delete-me.pdf"}, "document": "first"},
            {"id": "two", "metadata": {"source": "keep-me.pdf"}, "document": "second"},
        ]
        target = FakeCollection(records)
        # Use the project work directory because Windows sandbox temp folders
        # may prohibit atomic Path.replace(), which manifests intentionally use.
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent / "work") as temporary:
            root = Path(temporary)
            docs = root / "documents"
            docs.mkdir()
            (docs / "delete-me.pdf").write_bytes(b"to remove")
            (docs / "keep-me.pdf").write_bytes(b"to retain")
            storage = root / "storage"
            storage.mkdir()
            (storage / "index_manifest.json").write_text(json.dumps({"delete-me.pdf": "old", "keep-me.pdf": "keep"}))
            with patch.object(rag, "DOCUMENTS_DIR", docs), patch.object(rag, "ROOT", root), patch.object(rag, "_managed_collections", return_value=[target]):
                result = rag.delete_document("delete-me.pdf")
            self.assertEqual(result["removed_chunks"], 1)
            self.assertFalse((docs / "delete-me.pdf").exists())
            self.assertTrue((docs / "keep-me.pdf").exists())
            self.assertEqual([record["metadata"]["source"] for record in records], ["keep-me.pdf"])

    def test_generate_quiz_validates_source_and_page(self) -> None:
        records = [
            {"id": "one", "metadata": {"source": "attendance.pdf", "page": 2, "chunk": 0}, "document": "Students must meet the attendance rule."},
        ]
        response = json.dumps([
            {
                "question": "What must students meet?",
                "options": {"A": "Attendance rule", "B": "Parking rule", "C": "Dress rule", "D": "Library rule"},
                "correct_answer": "A",
                "explanation": "The passage states that students must meet the attendance rule.",
                "source": "attendance.pdf",
                "page": 2,
            },
        ] * 5)
        with patch.object(rag, "collection", return_value=FakeCollection(records)), patch.object(rag, "_quiz_text", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "enough valid"):
                rag.generate_quiz(["attendance.pdf"], 5, "Easy")


if __name__ == "__main__":
    unittest.main()
