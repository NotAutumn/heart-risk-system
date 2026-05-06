"""Broader API, page, and security tests for the heart disease web application."""

from __future__ import annotations

import io
import json
import unittest
from pathlib import Path

import pandas as pd

from app import create_app


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_DIR / "sample_upload.csv"
UPLOAD_DIR = PROJECT_DIR / "web" / "uploads"

try:
    import openpyxl  # noqa: F401

    HAS_OPENPYXL = True
except Exception:  # pragma: no cover - optional test dependency
    HAS_OPENPYXL = False


class WebAppTestCase(unittest.TestCase):
    """Validate pages, APIs, validation rules, and security headers."""

    def setUp(self) -> None:
        self.app = create_app()
        self.client = self.app.test_client()
        self.generated_files: list[Path] = []
        self.raw_df = pd.read_csv(RAW_DATA_PATH)

    def tearDown(self) -> None:
        for path in self.generated_files:
            if path.exists():
                path.unlink()

    def upload_bytes(self, file_bytes: bytes, filename: str):
        response = self.client.post(
            "/api/upload",
            data={"file": (io.BytesIO(file_bytes), filename)},
            content_type="multipart/form-data",
        )
        payload = response.get_json()
        if response.status_code == 200 and payload and payload.get("success"):
            self.generated_files.append(UPLOAD_DIR / payload["data"]["file_id"])
        return response, payload

    def upload_csv_df(self, df: pd.DataFrame, filename: str = "sample.csv"):
        return self.upload_bytes(df.to_csv(index=False).encode("utf-8"), filename)

    def test_index_page_renders(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        self.assertIn("上传文件", text)
        self.assertIn("cdn.plot.ly", text)

    def test_health_endpoint(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_metadata_endpoint_has_expected_shape(self) -> None:
        response = self.client.get("/api/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        data = payload["data"]
        self.assertEqual(len(data["required_columns"]), 13)
        self.assertIn(".csv", data["allowed_extensions"])
        self.assertEqual(data["max_upload_rows"], 500)

    def test_security_headers_present(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertEqual(response.headers["X-Frame-Options"], "SAMEORIGIN")
        self.assertIn("https://cdn.plot.ly", response.headers["Content-Security-Policy"])

    def test_upload_csv_preview(self) -> None:
        response, payload = self.upload_csv_df(self.raw_df)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["rows"], len(self.raw_df))
        self.assertTrue(payload["data"]["preview"])

    @unittest.skipUnless(HAS_OPENPYXL, "openpyxl is required for XLSX upload test")
    def test_upload_xlsx_preview(self) -> None:
        buffer = io.BytesIO()
        self.raw_df.head(5).to_excel(buffer, index=False)
        response, payload = self.upload_bytes(buffer.getvalue(), "sample.xlsx")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(payload["success"])
        self.assertEqual(payload["data"]["rows"], 5)

    def test_invalid_extension_rejected(self) -> None:
        response, payload = self.upload_bytes(b"hello", "bad.txt")
        self.assertEqual(response.status_code, 400)
        self.assertIn("仅支持上传", payload["message"])

    def test_missing_columns_rejected(self) -> None:
        reduced = self.raw_df.drop(columns=["chol"])
        response, payload = self.upload_csv_df(reduced, "missing_col.csv")
        self.assertEqual(response.status_code, 400)
        self.assertIn("缺少必要字段", payload["message"])

    def test_empty_upload_rejected(self) -> None:
        response, payload = self.upload_bytes(b"", "empty.csv")
        self.assertEqual(response.status_code, 400)
        self.assertFalse(payload["success"])

    def test_large_upload_is_rejected(self) -> None:
        large_df = pd.concat([self.raw_df] * 2, ignore_index=True)
        if len(large_df) <= 500:
            large_df = pd.concat([large_df] * 2, ignore_index=True)
        response, payload = self.upload_csv_df(large_df, "too_large.csv")
        self.assertEqual(response.status_code, 400)
        self.assertIn("最多支持", payload["message"])

    def test_safe_filename_is_used_for_upload(self) -> None:
        response, payload = self.upload_csv_df(self.raw_df.head(3), "../../../<script>.csv")
        self.assertEqual(response.status_code, 200)
        file_id = payload["data"]["file_id"]
        self.assertNotIn("..", file_id)
        self.assertNotIn("<", file_id)
        self.assertTrue((UPLOAD_DIR / file_id).exists())

    def test_predict_missing_file_returns_404(self) -> None:
        response = self.client.post("/api/predict", json={"file_id": "missing.csv"})
        self.assertEqual(response.status_code, 404)
        self.assertIn("找不到上传文件", response.get_json()["message"])

    def test_predict_success_contains_expected_keys(self) -> None:
        upload_response, upload_payload = self.upload_csv_df(self.raw_df)
        self.assertEqual(upload_response.status_code, 200)
        file_id = upload_payload["data"]["file_id"]

        response = self.client.post("/api/predict", json={"file_id": file_id})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["success"])
        data = payload["data"]
        self.assertIn("summary", data)
        self.assertIn("predictions", data)
        self.assertIn("global_importance", data)
        self.assertIn("local_explanations", data)
        self.assertIn("interaction_summary", data)
        self.assertIn("report_url", data)

        report_name = data["report_url"].rsplit("/", 1)[-1]
        self.generated_files.append(UPLOAD_DIR / report_name)

    def test_report_download_missing_returns_404(self) -> None:
        response = self.client.get("/api/report/not_found.md")
        self.assertEqual(response.status_code, 404)
        self.assertIn("报告不存在", response.get_json()["message"])

    def test_report_download_sanitizes_filename(self) -> None:
        response = self.client.get("/api/report/../../app.py")
        self.assertEqual(response.status_code, 404)

    def test_prediction_report_can_be_downloaded(self) -> None:
        upload_response, upload_payload = self.upload_csv_df(self.raw_df.head(10), "download.csv")
        self.assertEqual(upload_response.status_code, 200)
        response = self.client.post("/api/predict", json={"file_id": upload_payload["data"]["file_id"]})
        self.assertEqual(response.status_code, 200)

        payload = response.get_json()["data"]
        report_name = payload["report_url"].rsplit("/", 1)[-1]
        self.generated_files.append(UPLOAD_DIR / report_name)
        report_response = self.client.get(payload["report_url"])
        self.assertEqual(report_response.status_code, 200)
        self.assertIn("attachment", report_response.headers.get("Content-Disposition", ""))
        report_response.close()

    def test_metadata_and_health_are_json_serializable(self) -> None:
        meta_response = self.client.get("/api/metadata")
        health_response = self.client.get("/api/health")
        json.dumps(meta_response.get_json(), ensure_ascii=False)
        json.dumps(health_response.get_json(), ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()

