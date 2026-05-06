"""Flask entry point for the heart disease web application."""

from __future__ import annotations

import os

from flask import Flask
from flask_cors import CORS
from markupsafe import escape
from waitress import serve

from routes import api_bp, page_bp



def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="web/templates", static_folder="web/static")
    app.config["JSON_AS_ASCII"] = False
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024
    CORS(app)

    # 统一注册前台页面和 RESTful API 蓝图。
    app.register_blueprint(page_bp)
    app.register_blueprint(api_bp, url_prefix="/api")

    @app.after_request
    def set_security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "SAMEORIGIN"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self' https://cdn.plot.ly; "
            "script-src 'self' https://cdn.plot.ly; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:;"
        )
        return response

    @app.errorhandler(413)
    def file_too_large(_error):
        return {"success": False, "message": "上传文件超过 2MB 限制。"}, 413

    @app.template_filter("safe_text")
    def safe_text(value: str) -> str:
        return str(escape(value))

    return app


app = create_app()

if __name__ == "__main__":
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "5000"))
    threads = int(os.getenv("APP_THREADS", "16"))
    serve(app, host=host, port=port, threads=threads)

