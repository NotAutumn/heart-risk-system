"""End-to-end smoke tests for the heart disease explainable prediction project."""

from __future__ import annotations

import io
import unittest
from pathlib import Path

import pandas as pd

from app import create_app


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "sample_upload.csv"
UPLOAD_DIR = PROJECT_DIR / "web" / "uploads"


class ProjectFlowTestCase(unittest.TestCase):
    """Verify the main upload -> predict -> report workflow."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()
        self.generated_files: list[Path] = []

    def tearDown(self) -> None:
        for path in self.generated_files:
            if path.exists():
                path.unlink()

    def test_metadata_endpoint(self) -> None:
        response = self.client.get("/api/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        self.assertEqual(len(payload["data"]["required_columns"]), 13)

    def test_end_to_end_prediction_flow(self) -> None:
        with RAW_DATA_PATH.open("rb") as handle:
            response = self.client.post(
                "/api/upload",
                data={"file": (io.BytesIO(handle.read()), "heart.csv")},
                content_type="multipart/form-data",
            )

        self.assertEqual(response.status_code, 200)
        upload_payload = response.get_json()
        self.assertTrue(upload_payload["success"])
        file_id = upload_payload["data"]["file_id"]
        self.generated_files.append(UPLOAD_DIR / file_id)

        predict_response = self.client.post("/api/predict", json={"file_id": file_id})
        self.assertEqual(predict_response.status_code, 200)
        predict_payload = predict_response.get_json()
        self.assertTrue(predict_payload["success"])
        self.assertIn("global_importance", predict_payload["data"])
        self.assertIn("local_explanations", predict_payload["data"])
        self.assertIn("interaction_summary", predict_payload["data"])

        report_url = predict_payload["data"]["report_url"]
        report_name = report_url.rsplit("/", 1)[-1]
        self.generated_files.append(UPLOAD_DIR / report_name)
        report_response = self.client.get(report_url)
        self.assertEqual(report_response.status_code, 200)
        report_response.close()

    def test_large_upload_is_rejected(self) -> None:
        raw_df = pd.read_csv(RAW_DATA_PATH)
        large_df = pd.concat([raw_df] * 2, ignore_index=True)
        if len(large_df) <= 500:
            large_df = pd.concat([large_df] * 2, ignore_index=True)
        csv_bytes = large_df.to_csv(index=False).encode("utf-8")

        response = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(csv_bytes), "too_large.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.get_json()
        self.assertIn("最多支持", payload["message"])


if __name__ == "__main__":
    unittest.main()

