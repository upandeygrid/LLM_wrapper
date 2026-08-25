"""
Fallback chain and template fallback tests.
"""

import pytest

from llm_shield.exceptions import FallbackExhaustedError, TemplateError
from llm_shield.fallback import FallbackChain, TemplateFallback


class TestFallbackChain:
    def test_iterate_models(self):
        chain = FallbackChain(["model-a", "model-b", "model-c"])

        assert chain.has_next
        assert chain.current_model == "model-a"
        chain.advance()

        assert chain.has_next
        assert chain.current_model == "model-b"
        chain.advance()

        assert chain.has_next
        assert chain.current_model == "model-c"
        chain.advance()

        assert not chain.has_next

    def test_exhausted_raises(self):
        chain = FallbackChain(["model-a"])
        chain.advance()

        with pytest.raises(FallbackExhaustedError):
            _ = chain.current_model

    def test_models_tried(self):
        chain = FallbackChain(["a", "b", "c"])
        chain.advance()
        chain.advance()

        assert chain.models_tried == ["a", "b"]

    def test_empty_chain(self):
        chain = FallbackChain([])
        assert not chain.has_next


class TestTemplateFallback:
    def test_dict_template(self):
        fb = TemplateFallback()
        result = fb.get_response(template={"name": "Default", "age": 0})
        assert '"name"' in result
        assert '"Default"' in result

    def test_string_template(self):
        fb = TemplateFallback()
        result = fb.get_response(template="fallback text")
        assert result == "fallback text"

    def test_no_template_raises(self):
        fb = TemplateFallback()
        with pytest.raises(TemplateError):
            fb.get_response(template=None)

    def test_callable_template(self):
        fb = TemplateFallback()
        result = fb.get_response(
            template=lambda prompt: {"prompt_echo": prompt},
            prompt="hello",
        )
        assert "hello" in result
