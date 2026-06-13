"""Tests for prompts module."""
import pytest
from prompts import (
    FASHION_ANALYSIS,
    JUDGE_PROMPT,
    VERSION,
    get_prompt,
)


class TestPromptStructure:
    def test_fashion_analysis_has_required_fields(self):
        assert "version" in FASHION_ANALYSIS
        assert "name" in FASHION_ANALYSIS
        assert "model" in FASHION_ANALYSIS
        assert "provider" in FASHION_ANALYSIS
        assert "temperature" in FASHION_ANALYSIS
        assert "max_tokens" in FASHION_ANALYSIS
        assert "prompt" in FASHION_ANALYSIS

    def test_judge_prompt_has_required_fields(self):
        assert "version" in JUDGE_PROMPT
        assert "name" in JUDGE_PROMPT
        assert "system" in JUDGE_PROMPT


class TestPromptVersioning:
    def test_all_prompts_same_version(self):
        assert FASHION_ANALYSIS["version"] == VERSION
        assert JUDGE_PROMPT["version"] == VERSION

    def test_version_is_string(self):
        assert isinstance(VERSION, str)
        assert len(VERSION) > 0


class TestPromptContent:
    def test_fashion_analysis_prompt_contains_json_structure(self):
        prompt = FASHION_ANALYSIS["prompt"]
        assert '"gender"' in prompt
        assert '"items"' in prompt
        assert '"overall_style"' in prompt
        assert '"occasion"' in prompt
        assert '"stylist_notes"' in prompt

    def test_fashion_analysis_prompt_contains_rules(self):
        prompt = FASHION_ANALYSIS["prompt"]
        assert "Rules:" in prompt
        assert "CLEARLY VISIBLE" in prompt
        assert "Maximum 5 items" in prompt

    def test_fashion_analysis_prompt_has_style_categories(self):
        prompt = FASHION_ANALYSIS["prompt"]
        assert "casual" in prompt
        assert "formal" in prompt
        assert "streetwear" in prompt

    def test_fashion_analysis_prompt_has_material_options(self):
        prompt = FASHION_ANALYSIS["prompt"]
        assert "denim" in prompt
        assert "cotton" in prompt
        assert "leather" in prompt


class TestGetPrompt:
    def test_get_fashion_analysis(self):
        prompt = get_prompt("fashion_analysis")
        assert prompt == FASHION_ANALYSIS

    def test_get_judge_prompt(self):
        prompt = get_prompt("fashion_analysis_judge")
        assert prompt == JUDGE_PROMPT

    def test_get_unknown_prompt_raises(self):
        with pytest.raises(ValueError, match="Unknown prompt"):
            get_prompt("nonexistent")

    def test_get_prompt_wrong_version_raises(self):
        with pytest.raises(ValueError, match="not found"):
            get_prompt("fashion_analysis", version="99.99.99")


class TestPromptModelConfig:
    def test_fashion_analysis_uses_groq(self):
        assert FASHION_ANALYSIS["provider"] == "groq"

    def test_fashion_analysis_model_name(self):
        assert "llama" in FASHION_ANALYSIS["model"].lower()

    def test_fashion_analysis_temperature(self):
        assert 0 <= FASHION_ANALYSIS["temperature"] <= 1

    def test_fashion_analysis_max_tokens(self):
        assert FASHION_ANALYSIS["max_tokens"] > 0
