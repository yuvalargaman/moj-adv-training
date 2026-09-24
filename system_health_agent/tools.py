import os
import time
from typing import Any, Dict
import psutil
import requests
from langchain_core.tools import tool


@tool
def get_system_metrics() -> Dict[str, Any]:
    """Gathers real-time host performance metrics: CPU usage percentage,
    RAM memory consumption and availability, and root disk space details.
    Unrestricted / Auto-approved execution.
    """
    cpu_percent = psutil.cpu_percent(interval=0.5)

    mem = psutil.virtual_memory()
    memory_info = {
        "total_gb": round(mem.total / (1024**3), 2),
        "available_gb": round(mem.available / (1024**3), 2),
        "used_gb": round(mem.used / (1024**3), 2),
        "percent_used": mem.percent,
    }

    disk = psutil.disk_usage("/")
    disk_info = {
        "total_gb": round(disk.total / (1024**3), 2),
        "free_gb": round(disk.free / (1024**3), 2),
        "percent_used": disk.percent,
    }

    return {
        "status": "success",
        "cpu_percent": cpu_percent,
        "memory": memory_info,
        "disk": disk_info,
    }


@tool
def check_endpoint_health(url: str, timeout_seconds: float = 5.0) -> Dict[str, Any]:
    """Sends an HTTP GET request to evaluate endpoint health.
    Returns status code, latency in milliseconds, and key response headers.
    Unrestricted / Auto-approved execution.
    """
    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    try:
        start_time = time.perf_counter()
        response = requests.get(url, timeout=timeout_seconds)
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        return {
            "status": "success",
            "target_url": url,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "headers": dict(list(response.headers.items())[:5]),
        }
    except requests.RequestException as exc:
        return {
            "status": "error",
            "target_url": url,
            "error": str(exc),
        }


@tool
def inspect_directory_metadata(dir_path: str) -> Dict[str, Any]:
    """Inspects a local directory path to gather structural metadata:
    total size, file count, subdirectory count, and a breakdown of file extensions.
    Strict HITL Approval Required before running.
    """
    normalized_path = os.path.abspath(dir_path)

    if not os.path.exists(normalized_path):
        return {"status": "error", "message": f"Directory not found: {dir_path}"}

    if not os.path.isdir(normalized_path):
        return {"status": "error", "message": f"Path is not a directory: {dir_path}"}

    total_size_bytes = 0
    file_count = 0
    dir_count = 0
    extension_counts: Dict[str, int] = {}

    for root, dirs, files in os.walk(normalized_path):
        dir_count += len(dirs)
        for f in files:
            file_count += 1
            file_path = os.path.join(root, f)
            try:
                total_size_bytes += os.path.getsize(file_path)
            except OSError:
                pass  # Skip inaccessible files

            _, ext = os.path.splitext(f)
            ext_key = ext.lower() if ext else "[no_extension]"
            extension_counts[ext_key] = extension_counts.get(ext_key, 0) + 1

    return {
        "status": "success",
        "path": normalized_path,
        "file_count": file_count,
        "subdir_count": dir_count,
        "total_size_mb": round(total_size_bytes / (1024 * 1024), 2),
        "extension_breakdown": extension_counts,
    }
