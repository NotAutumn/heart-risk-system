"""HTTP routes for the Flask web application."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_file

from web_service import (
    compute_prediction_payload,
    get_project_metadata,
    preview_file,
    save_uploaded_file,
)

api_bp = Blueprint("api", __name__)
page_bp = Blueprint("page", __name__)


@page_bp.route("/")
def index():
    """Render the main browser page."""
    return render_template("index.html")


@api_bp.route("/health", methods=["GET"])
def health_check():
    """Simple endpoint used by Docker and deployment checks."""
    return jsonify({"status": "ok"})


@api_bp.route("/metadata", methods=["GET"])
def metadata():
    """Return reusable project metadata for the frontend."""
    return jsonify({"success": True, "data": get_project_metadata()})


@api_bp.route("/upload", methods=["POST"])
def upload_file():
    """Upload and validate a CSV/Excel file."""
    if "file" not in request.files:
        return jsonify({"success": False, "message": "未接收到上传文件。"}), 400

    file_storage = request.files["file"]
    try:
        save_path = save_uploaded_file(file_storage)
        payload = preview_file(save_path)
        return jsonify({"success": True, "message": "文件上传成功。", "data": payload})
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@api_bp.route("/predict", methods=["POST"])
def predict():
    """Run model inference and return SHAP-ready JSON payload."""
    data = request.get_json(silent=True) or {}
    file_id = data.get("file_id", "")
    try:
        payload = compute_prediction_payload(file_id)
        return jsonify({"success": True, "message": "预测完成。", "data": payload})
    except FileNotFoundError as exc:
        return jsonify({"success": False, "message": str(exc)}), 404
    except Exception as exc:
        return jsonify({"success": False, "message": str(exc)}), 400


@api_bp.route("/report/<path:filename>", methods=["GET"])
def download_report(filename: str):
    """Download the generated Markdown report."""
    file_path = Path("web/uploads") / Path(filename).name
    if not file_path.exists():
        return jsonify({"success": False, "message": "报告不存在。"}), 404
    return send_file(file_path, as_attachment=True)

