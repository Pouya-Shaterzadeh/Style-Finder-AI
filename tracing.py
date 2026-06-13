"""
LangSmith tracing and structured metrics logging for Style Finder AI v2.

Set these environment variables in HF Space secrets:
  LANGCHAIN_TRACING_V2=true
  LANGCHAIN_API_KEY=your_langsmith_api_key
  LANGCHAIN_PROJECT=style-finder-ai-v2
"""

import os
import json
import time
import hashlib
import logging
from datetime import datetime, timezone
from contextlib import contextmanager
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


def _is_langsmith_enabled():
    """Check at call time, not import time."""
    return os.environ.get("LANGCHAIN_TRACING_V2", "false").lower() == "true"


def _get_client():
    """Lazy-import LangSmith client to avoid startup cost when disabled."""
    if not _is_langsmith_enabled():
        return None
    try:
        from langsmith import Client
        return Client()
    except ImportError:
        logger.warning("langsmith not installed — tracing disabled")
        return None
    except Exception as e:
        logger.warning(f"LangSmith client init failed: {e}")
        return None


_client = None


def get_client():
    global _client
    if _client is None:
        _client = _get_client()
    return _client


def _image_hash(image_bytes: bytes) -> str:
    """Generate short hash for image deduplication in traces."""
    return hashlib.md5(image_bytes).hexdigest()[:8]


@contextmanager
def trace_fashion_analysis(image_bytes: bytes, prompt_version: str):
    """
    Context manager that traces the full fashion analysis pipeline.

    Usage:
        with trace_fashion_analysis(image_bytes, "1.0.0") as trace:
            trace.log("vlm_analysis", latency_ms=1200, items_detected=3)
            trace.log("trendyol_search", latency_ms=800, results_found=12)
            trace.log("clip_scoring", latency_ms=500, top_score=0.89)
    """
    run_data = {
        "image_hash": _image_hash(image_bytes),
        "image_size_bytes": len(image_bytes),
        "prompt_version": prompt_version,
        "start_time": datetime.now(timezone.utc).isoformat(),
        "steps": [],
        "total_latency_ms": 0,
    }

    start = time.monotonic()
    trace = _TraceRun(run_data)

    try:
        yield trace
    except Exception as e:
        run_data["error"] = str(e)
        raise
    finally:
        run_data["total_latency_ms"] = int((time.monotonic() - start) * 1000)
        run_data["end_time"] = datetime.now(timezone.utc).isoformat()
        _log_run(run_data)


class _TraceRun:
    """Internal helper to collect step data during a traced run."""

    def __init__(self, run_data: dict):
        self._data = run_data

    def log(self, step_name: str, **metrics):
        """Log a pipeline step with arbitrary metrics."""
        step = {"name": step_name, **metrics}
        self._data["steps"].append(step)
        self._data["total_latency_ms"] += metrics.get("latency_ms", 0)

    @property
    def steps(self):
        return self._data["steps"]


def _log_run(run_data: dict):
    """Send run data to LangSmith and local structured log."""
    # Structured log (always)
    logger.info(
        json.dumps(
            {
                "event": "fashion_analysis",
                "image_hash": run_data.get("image_hash"),
                "image_size_bytes": run_data.get("image_size_bytes"),
                "prompt_version": run_data.get("prompt_version"),
                "total_latency_ms": run_data.get("total_latency_ms"),
                "step_count": len(run_data.get("steps", [])),
                "error": run_data.get("error"),
                "steps": run_data.get("steps"),
            }
        )
    )

    # LangSmith trace
    client = get_client()
    if client is None:
        logger.info("LangSmith client not available — skipping remote trace")
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "style-finder-ai-v2")
        now = datetime.now(timezone.utc)

        client.create_run(
            id=uuid4(),
            name="fashion_analysis",
            run_type="chain",
            inputs={
                "image_hash": run_data.get("image_hash"),
                "image_size_bytes": run_data.get("image_size_bytes"),
                "prompt_version": run_data.get("prompt_version"),
            },
            outputs={
                "total_latency_ms": run_data.get("total_latency_ms"),
                "steps": run_data.get("steps", []),
                "error": run_data.get("error"),
            },
            start_time=now,
            end_time=now,
            project_name=project,
        )
        logger.info("LangSmith trace sent successfully")
    except Exception as e:
        logger.warning(f"LangSmith trace failed: {e}")


def log_llm_call(
    prompt_name: str,
    prompt_version: str,
    model: str,
    image_bytes: bytes,
    response_text: str,
    latency_ms: int,
    items_detected: int = 0,
    overall_style: str = "",
    temperature: float = 0.1,
    max_tokens: int = 1024,
):
    """
    Log a single LLM (Groq) call for monitoring and evaluation.
    """
    record = {
        "event": "llm_call",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "model": model,
        "image_hash": _image_hash(image_bytes),
        "response_preview": response_text[:300] + "..." if len(response_text) > 300 else response_text,
        "latency_ms": latency_ms,
        "items_detected": items_detected,
        "overall_style": overall_style,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    logger.info(json.dumps(record))

    client = get_client()
    if client is None:
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "style-finder-ai-v2")
        now = datetime.now(timezone.utc)

        client.create_run(
            id=uuid4(),
            name=f"llm_{prompt_name}",
            run_type="llm",
            inputs={
                "image_hash": _image_hash(image_bytes),
                "prompt_version": prompt_version,
            },
            outputs={
                "response_preview": response_text[:1000],
                "latency_ms": latency_ms,
                "items_detected": items_detected,
                "overall_style": overall_style,
            },
            start_time=now,
            end_time=now,
            extra={
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            project_name=project,
        )
    except Exception as e:
        logger.warning(f"LangSmith log_llm_call failed: {e}")


def log_search_call(
    query: str,
    results_count: int,
    latency_ms: int,
):
    """Log a Trendyol search call."""
    record = {
        "event": "search_call",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "results_count": results_count,
        "latency_ms": latency_ms,
    }
    logger.info(json.dumps(record))

    client = get_client()
    if client is None:
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "style-finder-ai-v2")
        now = datetime.now(timezone.utc)

        client.create_run(
            id=uuid4(),
            name="trendyol_search",
            run_type="tool",
            inputs={"query": query},
            outputs={"results_count": results_count, "latency_ms": latency_ms},
            start_time=now,
            end_time=now,
            project_name=project,
        )
    except Exception as e:
        logger.warning(f"LangSmith log_search_call failed: {e}")


def log_clip_scoring(
    image_hash: str,
    products_scored: int,
    top_score: float,
    latency_ms: int,
):
    """Log fashion-CLIP similarity scoring."""
    record = {
        "event": "clip_scoring",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_hash": image_hash,
        "products_scored": products_scored,
        "top_score": top_score,
        "latency_ms": latency_ms,
    }
    logger.info(json.dumps(record))

    client = get_client()
    if client is None:
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "style-finder-ai-v2")
        now = datetime.now(timezone.utc)

        client.create_run(
            id=uuid4(),
            name="clip_scoring",
            run_type="tool",
            inputs={"image_hash": image_hash, "products_scored": products_scored},
            outputs={"top_score": top_score, "latency_ms": latency_ms},
            start_time=now,
            end_time=now,
            project_name=project,
        )
    except Exception as e:
        logger.warning(f"LangSmith log_clip_scoring failed: {e}")


def log_evaluation_result(
    prompt_name: str,
    prompt_version: str,
    test_case_id: str,
    image_hash: str,
    response: str,
    scores: dict,
):
    """Log an evaluation result for prompt quality tracking."""
    record = {
        "event": "evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_name": prompt_name,
        "prompt_version": prompt_version,
        "test_case_id": test_case_id,
        "image_hash": image_hash,
        "response_preview": response[:300],
        "scores": scores,
    }
    logger.info(json.dumps(record))

    client = get_client()
    if client is None:
        return

    try:
        project = os.environ.get("LANGCHAIN_PROJECT", "style-finder-ai-v2")
        now = datetime.now(timezone.utc)

        client.create_run(
            id=uuid4(),
            name=f"eval_{prompt_name}",
            run_type="chain",
            inputs={
                "test_case_id": test_case_id,
                "image_hash": image_hash,
            },
            outputs={"scores": scores, "response": response[:1000]},
            start_time=now,
            end_time=now,
            extra={"prompt_version": prompt_version},
            project_name=project,
        )
    except Exception as e:
        logger.warning(f"LangSmith log_evaluation_result failed: {e}")
