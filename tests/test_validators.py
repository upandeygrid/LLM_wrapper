"""
Validator unit tests — tests each validator type independently.
"""

import pytest

from llm_shield.validators import (
    CompositeValidator,
    JsonSchemaValidator,
    LengthValidator,
    RegexValidator,
    build_validators,
)


class TestJsonSchemaValidator:
    def test_valid_json(self, user_schema):
        v = JsonSchemaValidator(user_schema)
        errors = v.validate('{"name": "Alice", "age": 28, "email": "a@b.com"}')
        assert errors == []

    def test_invalid_json(self, user_schema):
        v = JsonSchemaValidator(user_schema)
        errors = v.validate("{invalid json")
        assert len(errors) == 1
        assert "Invalid JSON" in errors[0].message

    def test_missing_required_field(self, user_schema):
        v = JsonSchemaValidator(user_schema)
        errors = v.validate('{"name": "Alice", "age": 28}')
        assert len(errors) >= 1
        assert any("email" in e.message for e in errors)

    def test_wrong_type(self, user_schema):
        v = JsonSchemaValidator(user_schema)
        errors = v.validate('{"name": "Alice", "age": "twenty", "email": "a@b.com"}')
        assert len(errors) >= 1

    def test_strips_markdown_fences(self, user_schema):
        v = JsonSchemaValidator(user_schema)
        response = '```json\n{"name": "Alice", "age": 28, "email": "a@b.com"}\n```'
        errors = v.validate(response)
        assert errors == []

    def test_negative_age(self, user_schema):
        v = JsonSchemaValidator(user_schema)
        errors = v.validate('{"name": "Alice", "age": -5, "email": "a@b.com"}')
        assert len(errors) >= 1


class TestRegexValidator:
    def test_matches(self):
        v = RegexValidator(r"\d{3}-\d{4}", "phone pattern")
        assert v.validate("Call 555-1234") == []

    def test_no_match(self):
        v = RegexValidator(r"\d{3}-\d{4}", "phone pattern")
        errors = v.validate("No phone here")
        assert len(errors) == 1
        assert "phone pattern" in errors[0].message


class TestLengthValidator:
    def test_valid_length(self):
        v = LengthValidator(min_length=5, max_length=100)
        assert v.validate("Hello World") == []

    def test_too_short(self):
        v = LengthValidator(min_length=10)
        errors = v.validate("Hi")
        assert len(errors) == 1
        assert "too short" in errors[0].message.lower()

    def test_too_long(self):
        v = LengthValidator(max_length=5)
        errors = v.validate("Way too long")
        assert len(errors) == 1
        assert "too long" in errors[0].message.lower()


class TestCompositeValidator:
    def test_all_pass(self, user_schema):
        v = CompositeValidator([
            JsonSchemaValidator(user_schema),
            LengthValidator(min_length=5),
        ])
        errors = v.validate('{"name": "Alice", "age": 28, "email": "a@b.com"}')
        assert errors == []

    def test_collects_all_errors(self):
        v = CompositeValidator([
            LengthValidator(min_length=100),
            RegexValidator(r"xyz", "must contain xyz"),
        ])
        errors = v.validate("short")
        assert len(errors) == 2


class TestBuildValidators:
    def test_schema_only(self, user_schema):
        v = build_validators(response_schema=user_schema)
        assert v is not None
        assert v.name == "json_schema"

    def test_no_validators(self):
        v = build_validators()
        assert v is None

    def test_mixed(self, user_schema):
        v = build_validators(
            response_schema=user_schema,
            validator_configs=[{"type": "length", "min_length": 10}],
        )
        assert v is not None
        assert v.name == "composite"
