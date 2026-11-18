import csv
import tempfile
import unittest
from pathlib import Path
from tfai.mains._shared import write_predict_samples_csv
from tfai.data.data_pipeline import DataPipeline

class BenchmarkIntegrityTests(unittest.TestCase):
    def test_identical_table_preserves_fingerprint(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "samples.csv"
            rows = [{"order": 1, "source_file": "/tmp/a.vtu", "params": {"Omega": 3.0}}]
            write_predict_samples_csv(rows, str(p))
            before = p.stat().st_mtime_ns
            write_predict_samples_csv(rows, str(p))
            self.assertEqual(before, p.stat().st_mtime_ns)
            rows[0]["params"]["Omega"] = 4.0
            write_predict_samples_csv(rows, str(p))
            self.assertIn("4.0", p.read_text())

    def test_pod_fit_excludes_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "train.csv"
            p.write_text("order,file_path\n1,/tmp/a.vtu\n3,/tmp/c.vtu\n")
            paths = [str(Path(d) / f"sample_{i}.pt") for i in [1, 2, 3]]
            self.assertEqual(DataPipeline._select_pod_fit_paths(paths, d), [paths[0], paths[2]])


class CacheRelocationTests(unittest.TestCase):
    def test_table_location_can_change_but_data_cannot(self):
        import copy
        from tfai.training.preprocess_cache import equivalent_preprocess_signatures
        a = {"general": {"train_samples": "/a.csv"}, "data": {
            "sample_table": {"path": "/a.csv", "mtime_ns": 1, "size": 10,
                             "sha256": "abc", "status": "file"},
            "sources": [{"path": "/source.vtu", "mtime_ns": 42}]}}
        b = copy.deepcopy(a)
        b["general"]["train_samples"] = "/b.csv"
        b["data"]["sample_table"].update(path="/b.csv", mtime_ns=2)
        self.assertTrue(equivalent_preprocess_signatures(a, b))
        b["data"]["sample_table"]["sha256"] = "different"
        self.assertFalse(equivalent_preprocess_signatures(a, b))
        b = copy.deepcopy(a)
        b["data"]["sources"][0]["mtime_ns"] = 99
        self.assertFalse(equivalent_preprocess_signatures(a, b))
