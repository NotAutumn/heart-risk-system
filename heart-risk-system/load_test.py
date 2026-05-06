"""Simple concurrent load test for the Flask web application.

Run `python app.py` first, then execute this script to validate
upload and prediction stability under concurrent requests.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from uuid import uuid4


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_FILE_PATH = PROJECT_DIR / "sample_upload.csv"
REPORT_PATH = PROJECT_DIR / "outputs" / "reports" / "web_runtime_report.md"


def request_json(url: str) -> dict:
    """Send a GET request and parse JSON."""
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    """Send a JSON POST request and parse JSON."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def post_multipart(url: str, file_path: Path) -> dict:
    """Upload a file using multipart/form-data built from the standard library."""
    boundary = f"----UploadBoundary{uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    file_bytes = file_path.read_bytes()

    body = [
        f"--{boundary}".encode("utf-8"),
        (
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"'
        ).encode("utf-8"),
        f"Content-Type: {mime_type}".encode("utf-8"),
        b"",
        file_bytes,
        f"--{boundary}--".encode("utf-8"),
        b"",
    ]
    payload = b"\r\n".join(body)
    request = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def percentile(sorted_values: list[float], ratio: float) -> float:
    """Return an approximate percentile from sorted latency values."""
    if not sorted_values:
        return 0.0
    index = min(len(sorted_values) - 1, max(0, int(round(ratio * (len(sorted_values) - 1)))))
    return sorted_values[index]


def run_load_test(base_url: str, file_path: Path, concurrency: int, requests_count: int) -> dict:
    """Upload once, then send concurrent prediction requests."""
    health = request_json(f"{base_url}/api/health")
    if health.get("status") != "ok":
        raise RuntimeError("服务健康检查失败。")

    upload_response = post_multipart(f"{base_url}/api/upload", file_path)
    if not upload_response.get("success"):
        raise RuntimeError(f"上传失败: {upload_response.get('message')}")
    file_id = upload_response["data"]["file_id"]

    latencies: list[float] = []
    successes = 0
    failures: list[str] = []
    started_at = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = []
        for _ in range(requests_count):
            futures.append(executor.submit(run_single_prediction, base_url, file_id))

        for future in as_completed(futures):
            try:
                latency = future.result()
                latencies.append(latency)
                successes += 1
            except Exception as exc:  # pragma: no cover - load test error collection
                failures.append(str(exc))

    total_seconds = time.perf_counter() - started_at
    sorted_latencies = sorted(latencies)
    report = {
        "base_url": base_url,
        "test_file": str(file_path),
        "concurrency": concurrency,
        "requests_count": requests_count,
        "successes": successes,
        "failures": len(failures),
        "total_seconds": round(total_seconds, 4),
        "avg_latency_seconds": round(statistics.mean(latencies), 4) if latencies else None,
        "p95_latency_seconds": round(percentile(sorted_latencies, 0.95), 4) if latencies else None,
        "max_latency_seconds": round(max(latencies), 4) if latencies else None,
        "error_samples": failures[:5],
    }
    write_runtime_report(report)
    return report


def run_single_prediction(base_url: str, file_id: str) -> float:
    """Execute one prediction request and return latency."""
    started_at = time.perf_counter()
    response = post_json(f"{base_url}/api/predict", {"file_id": file_id})
    if not response.get("success"):
        raise RuntimeError(response.get("message", "预测失败"))
    return time.perf_counter() - started_at


def write_runtime_report(report: dict) -> None:
    """Write the load-test result as Markdown."""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Web 应用 并发压测报告",
        "",
        f"- 测试地址: `{report['base_url']}`",
        f"- 测试文件: `{report['test_file']}`",
        f"- 并发线程数: {report['concurrency']}",
        f"- 总请求数: {report['requests_count']}",
        f"- 成功请求数: {report['successes']}",
        f"- 失败请求数: {report['failures']}",
        f"- 总耗时: {report['total_seconds']} 秒",
        f"- 平均响应时间: {report['avg_latency_seconds']} 秒",
        f"- P95 响应时间: {report['p95_latency_seconds']} 秒",
        f"- 最大响应时间: {report['max_latency_seconds']} 秒",
    ]
    if report["error_samples"]:
        lines.append("- 错误样例:")
        lines.extend([f"  - {item}" for item in report["error_samples"]])
    else:
        lines.append("- 错误样例: 无")
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description="Heart disease load test")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="running web service base url")
    parser.add_argument("--file", default=str(DEFAULT_FILE_PATH), help="input CSV/XLSX file used for upload")
    parser.add_argument("--concurrency", type=int, default=50, help="concurrent prediction workers")
    parser.add_argument("--requests", type=int, default=50, help="total prediction requests")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    file_path = Path(os.path.expanduser(args.file)).resolve()
    try:
        result = run_load_test(args.base_url.rstrip("/"), file_path, args.concurrency, args.requests)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print(f"压测报告已输出到: {REPORT_PATH}")
    except urllib.error.URLError as exc:
        raise SystemExit(f"无法连接到服务，请先启动 app.py: {exc}") from exc


