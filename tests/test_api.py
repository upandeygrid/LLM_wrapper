"""
FastAPI endpoint tests — validates API contracts using TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from llm_shield.config import ShieldConfig
from llm_shield.engine import Shield
from llm_shield.escalation import InMemoryEscalation
from llm_shield.exceptions import LLMCallError
from llm_shield.providers import MockProvider
from server.app import create_app
from server import dependencies as deps


@pytest.fixture
def app_with_mock():
    """Create a test app with mock provider."""
    # Reset singletons
    deps.reset_singletons()

    # Override dependencies
    config = ShieldConfig(
        default_model="groq/test",
        fallback_models=["groq/fallback"],
        max_retries=1,
        max_repairs=1,
        timeout_seconds=5.0,
        escalation_mode="in_memory",
    )
    escalation = InMemoryEscalation()
    provider = MockProvider(responses=[
        '{"name": "Alice", "age": 28, "email": "alice@example.com"}'
    ])
    shield = Shield(config=config, provider=provider, escalation_handler=escalation)

    app = create_app()

    # Override dependency injection
    app.dependency_overrides[deps.get_shield] = lambda: shield
    app.dependency_overrides[deps.get_config] = lambda: config
    app.dependency_overrides[deps.get_escalation_handler] = lambda: escalation

    yield app, escalation, provider

    # Cleanup
    app.dependency_overrides.clear()
    deps.reset_singletons()


@pytest.fixture
def client(app_with_mock):
    app, _, _ = app_with_mock
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "config" in data

    def test_health_shows_config(self, client):
        response = client.get("/health")
        config = response.json()["config"]
        assert "default_model" in config
        assert "max_retries" in config


class TestExecuteEndpoint:
    def test_happy_path(self, client):
        response = client.post("/execute", json={
            "prompt": "Generate a user profile",
            "response_schema": {
                "type": "object",
                "required": ["name", "age", "email"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "email": {"type": "string"},
                },
            },
        })

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "SUCCEEDED"
        assert data["result"]["name"] == "Alice"
        assert "execution_trace" in data

    def test_missing_prompt_returns_422(self, client):
        response = client.post("/execute", json={})
        assert response.status_code == 422

    def test_empty_prompt_returns_422(self, client):
        response = client.post("/execute", json={"prompt": ""})
        assert response.status_code == 422

    def test_response_includes_trace(self, client):
        response = client.post("/execute", json={
            "prompt": "Say hello",
        })

        data = response.json()
        assert data["status"] == "SUCCEEDED"
        trace = data["execution_trace"]
        assert "states_visited" in trace
        assert "transitions" in trace
        assert trace["total_llm_calls"] >= 1


class TestExecuteWithFailures:
    def test_escalation_on_failure(self, app_with_mock):
        app, escalation, _ = app_with_mock

        # Create a new shield with failing provider
        config = ShieldConfig(
            default_model="groq/test",
            fallback_models=[],
            max_retries=0,
            max_repairs=0,
            timeout_seconds=5.0,
            escalation_mode="in_memory",
        )
        failing_provider = MockProvider(responses=[
            LLMCallError("fail", model="groq/test"),
        ] * 10)
        shield = Shield(
            config=config,
            provider=failing_provider,
            escalation_handler=escalation,
        )
        app.dependency_overrides[deps.get_shield] = lambda: shield

        client = TestClient(app)
        response = client.post("/execute", json={
            "prompt": "Generate something",
            "response_schema": {"type": "object"},
        })

        data = response.json()
        assert data["status"] == "FAILED"
        assert data["escalation"] is not None
        assert escalation.pending_count >= 1


class TestEscalationEndpoints:
    def test_list_empty(self, client):
        response = client.get("/escalations")
        assert response.status_code == 200
        assert response.json() == []

    def test_escalation_lifecycle(self, app_with_mock):
        """Create an escalation via failed execute, then list, view, and resolve it."""
        app, escalation, _ = app_with_mock

        # Force failure
        config = ShieldConfig(
            default_model="groq/test",
            fallback_models=[],
            max_retries=0,
            max_repairs=0,
            escalation_mode="in_memory",
        )
        failing_provider = MockProvider(responses=[
            LLMCallError("fail", model="groq/test"),
        ] * 10)
        shield = Shield(
            config=config,
            provider=failing_provider,
            escalation_handler=escalation,
        )
        app.dependency_overrides[deps.get_shield] = lambda: shield
        client = TestClient(app)

        # 1. Trigger escalation
        exec_response = client.post("/execute", json={
            "prompt": "test", "response_schema": {"type": "object"},
        })
        assert exec_response.json()["status"] == "FAILED"

        # 2. List escalations
        list_response = client.get("/escalations")
        assert list_response.status_code == 200
        escalations = list_response.json()
        assert len(escalations) >= 1
        esc_id = escalations[0]["id"]

        # 3. View escalation detail
        detail_response = client.get(f"/escalations/{esc_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["original_prompt"] == "test"
        assert detail["status"] == "pending"

        # 4. Resolve escalation
        resolve_response = client.post(
            f"/escalations/{esc_id}/resolve",
            json={"notes": "Handled manually"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "resolved"

        # 5. Stats
        stats_response = client.get("/escalations/stats")
        assert stats_response.status_code == 200
        stats = stats_response.json()
        assert stats["resolved"] >= 1

    def test_get_nonexistent_escalation(self, client):
        response = client.get("/escalations/nonexistent")
        assert response.status_code == 404

    def test_resolve_nonexistent(self, client):
        response = client.post(
            "/escalations/nonexistent/resolve",
            json={"notes": "test"},
        )
        assert response.status_code == 404
