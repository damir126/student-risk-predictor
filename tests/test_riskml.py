import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from riskml import api
from riskml.data import CATEGORICAL, NUMERIC, TARGET, generate
from riskml.train import train

GOOD = {
    "age": 19, "faculty": "medicine", "tuition": "grant", "scholarship": "yes",
    "works_part_time": "no", "dormitory": "yes", "commute_minutes": 10, "admission_score": 175,
    "attendance_rate": 0.97, "study_hours_week": 20, "gpa_sem1": 4.8, "gpa_sem2": 4.9, "failed_courses": 0,
}
WEAK = {
    "age": 22, "faculty": "engineering", "tuition": "contract", "scholarship": "no",
    "works_part_time": "yes", "dormitory": "no", "commute_minutes": 90, "admission_score": 110,
    "attendance_rate": 0.45, "study_hours_week": 2, "gpa_sem1": 2.9, "gpa_sem2": 2.1, "failed_courses": 5,
}


class DataTests(unittest.TestCase):
    def test_generator_is_reproducible_and_valid(self):
        a, b = generate(500, seed=1), generate(500, seed=1)
        self.assertTrue(a.equals(b))
        self.assertFalse(a[NUMERIC + CATEGORICAL].isna().any().any())
        self.assertTrue(a["attendance_rate"].between(0, 1).all())
        self.assertTrue(0.05 < a[TARGET].mean() < 0.5, "class balance should be realistic")


class ModelAndApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        model_path = Path(cls.tmp.name) / "model.joblib"
        cls.summary = train(None, out_model=model_path, reports=Path(cls.tmp.name), n=2000, verbose=False)
        api.MODEL_PATH = model_path
        api.model_bundle.cache_clear()
        cls.client = TestClient(api.app)

    @classmethod
    def tearDownClass(cls):
        api.model_bundle.cache_clear()
        cls.tmp.cleanup()

    def test_model_quality(self):
        self.assertGreater(self.summary["test"]["roc_auc"], 0.8)

    def test_reports_are_written(self):
        self.assertTrue((Path(self.tmp.name) / "evaluation.png").exists())
        self.assertTrue((Path(self.tmp.name) / "metrics.json").exists())

    def test_attendance_matters(self):
        top3 = list(self.summary["importance"])[:4]
        self.assertIn("attendance_rate", top3)

    def test_health(self):
        self.assertEqual(self.client.get("/health").json()["status"], "ok")

    def test_predictions_make_sense(self):
        good = self.client.post("/predict", json=GOOD).json()
        weak = self.client.post("/predict", json=WEAK).json()
        self.assertLess(good["dropout_probability"], weak["dropout_probability"])
        self.assertFalse(good["at_risk"])
        self.assertTrue(weak["at_risk"])
        self.assertEqual(weak["risk_level"], "high")
        self.assertTrue(any("Attendance" in r for r in weak["recommendations"]))

    def test_batch(self):
        r = self.client.post("/predict/batch", json={"students": [GOOD, WEAK]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.json()), 2)

    def test_validation(self):
        bad = {**GOOD, "attendance_rate": 1.7}
        self.assertEqual(self.client.post("/predict", json=bad).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={**GOOD, "faculty": "magic"}).status_code, 422)
        self.assertEqual(self.client.post("/predict/batch", json={"students": []}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
