"""Tests for tracing module."""
import pytest
from unittest.mock import patch, MagicMock
from tracing import (
    _TraceRun,
    _image_hash,
    trace_fashion_analysis,
    log_llm_call,
    log_search_call,
    log_clip_scoring,
    log_evaluation_result,
)


class TestImageHash:
    def test_hash_deterministic(self):
        data = b"test image data"
        assert _image_hash(data) == _image_hash(data)

    def test_hash_different_for_different_data(self):
        assert _image_hash(b"data1") != _image_hash(b"data2")

    def test_hash_length(self):
        h = _image_hash(b"test")
        assert len(h) == 8


class TestTraceRun:
    def test_log_step(self):
        run_data = {"steps": [], "total_latency_ms": 0}
        trace = _TraceRun(run_data)
        trace.log("vlm_analysis", latency_ms=1000, items_detected=3)
        assert len(run_data["steps"]) == 1
        assert run_data["steps"][0]["name"] == "vlm_analysis"
        assert run_data["steps"][0]["latency_ms"] == 1000
        assert run_data["steps"][0]["items_detected"] == 3

    def test_multiple_steps(self):
        run_data = {"steps": [], "total_latency_ms": 0}
        trace = _TraceRun(run_data)
        trace.log("step1", latency_ms=100)
        trace.log("step2", latency_ms=200)
        assert len(run_data["steps"]) == 2
        assert run_data["total_latency_ms"] == 300

    def test_steps_property(self):
        run_data = {"steps": [{"name": "test"}], "total_latency_ms": 0}
        trace = _TraceRun(run_data)
        assert trace.steps == [{"name": "test"}]


class TestTraceFashionAnalysis:
    def test_context_manager_populates_data(self):
        with trace_fashion_analysis(b"test image", "1.0.0") as trace:
            trace.log("test_step", latency_ms=500)
        # Should not raise

    def test_context_manager_with_error(self):
        with pytest.raises(ValueError):
            with trace_fashion_analysis(b"test image", "1.0.0") as trace:
                raise ValueError("test error")


class TestLogLlmCall:
    @patch("tracing.get_client", return_value=None)
    def test_logs_without_langsmith(self, mock_client):
        log_llm_call(
            prompt_name="fashion_analysis",
            prompt_version="1.0.0",
            model="test-model",
            image_bytes=b"test image",
            response_text="test response",
            latency_ms=100,
        )

    @patch("tracing.get_client", return_value=None)
    def test_logs_with_items_detected(self, mock_client):
        log_llm_call(
            prompt_name="fashion_analysis",
            prompt_version="1.0.0",
            model="test-model",
            image_bytes=b"test image",
            response_text="test response",
            latency_ms=100,
            items_detected=3,
            overall_style="casual",
        )


class TestLogSearchCall:
    @patch("tracing.get_client", return_value=None)
    def test_logs_search(self, mock_client):
        log_search_call(
            query="Kadın Lacivert Jean",
            results_count=12,
            latency_ms=500,
        )


class TestLogClipScoring:
    @patch("tracing.get_client", return_value=None)
    def test_logs_scoring(self, mock_client):
        log_clip_scoring(
            image_hash="abc12345",
            products_scored=10,
            top_score=0.89,
            latency_ms=300,
        )


class TestLogEvaluationResult:
    @patch("tracing.get_client", return_value=None)
    def test_logs_evaluation(self, mock_client):
        log_evaluation_result(
            prompt_name="fashion_analysis",
            prompt_version="1.0.0",
            test_case_id="fashion_001",
            image_hash="test123",
            response="test response",
            scores={"overall_score": 4.5},
        )
