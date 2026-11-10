import io
import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "fast"))
import ovcache


class PreparedCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.model = self.path / "model.onnx"
        self.model.write_bytes(b"source model")
        self.core = Mock()
        self.core.get_property.side_effect = lambda *a: str(self.path) if a == ("CACHE_DIR",) else "test CPU"
        self.core.compile_model.return_value.export_model.return_value = io.BytesIO(b"compiled graph")
        self.core.import_model.return_value = "imported graph"
        self.config = {"NUM_STREAMS": "8"}
        self.writer = ovcache.BundleWriter()
        self.writer.compile(self.core, self.model, ("preprocessing", 1), lambda m: m, self.config)
        self.bundle = self.path / ovcache.BUNDLE_NAME
        self.writer.save(self.bundle)
        ovcache._bundles.clear()
        self.addCleanup(ovcache._bundles.clear)

    def load(self, config=None):
        return ovcache.compile_model(self.core, self.model, ("preprocessing", 1), lambda m: m,
                                     self.config if config is None else config)

    def test_prepared_graph_is_imported_with_the_requested_configuration(self):
        self.assertEqual(self.load(), "imported graph")
        self.core.import_model.assert_called_once_with(b"compiled graph", "CPU", {"NUM_STREAMS": "8"})

    def test_source_change_fails_explicitly_instead_of_loading_stale_graph(self):
        self.model.write_bytes(b"changed source model")
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.load()
        self.core.import_model.assert_not_called()

    def test_configuration_change_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.load({"NUM_STREAMS": "4"})

    def test_runtime_change_fails_explicitly(self):
        self.core.get_property.side_effect = lambda *a: str(self.path) if a == ("CACHE_DIR",) else "different CPU"
        with self.assertRaisesRegex(RuntimeError, "does not match"):
            self.load()

    def test_corrupted_graph_fails_hash_validation(self):
        digest, entry = next(iter(self.writer.entries.items()))
        with zipfile.ZipFile(self.bundle, "w") as z:
            z.writestr("manifest.json", json.dumps({digest: entry}))
            z.writestr(digest, b"wrong graph")
        with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
            self.load()
        self.core.import_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
