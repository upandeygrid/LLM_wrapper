# LLM Shield

**Production-ready library that wraps any LLM call in a reliable control loop:**

```
Validate → Retry → Repair → Fallback → Human Escalation
```

Every execution reaches a clear, traceable terminal state. No infinite loops. No silent failures.

---

## Quick Start

### 1. Install

```bash
# Clone the repo
cd LLM_wrapper

# Install with all dependencies (library + server + dev tools)
make install-dev

# Copy and configure environment
cp .env.example .env
# Edit .env with your GROQ_API_KEY
```

### 2. Run the Server

```bash
make run
# Server starts at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
# OpenAPI spec at http://localhost:8000/openapi.json
```

### 3. Use with Postman

**Option A — Import OpenAPI spec (recommended):**
1. Open Postman → Import → Link
2. Enter: `http://localhost:8000/openapi.json`
3. All endpoints auto-generated with schemas

**Option B — Import pre-built collection:**
1. Open Postman → Import → File
2. Select `postman/llm_shield.postman_collection.json`
3. Set variable `base_url` = `http://localhost:8000`

### 4. Run Tests

```bash
# Run all tests
make test

# Run with coverage
make test-cov
```

---

## API Endpoints

| Method | Endpoint | Description |
|:-------|:---------|:------------|
| `GET` | `/health` | Server health and config |
| `POST` | `/execute` | Run LLM call through Shield |
| `GET` | `/escalations` | List all escalations |
| `GET` | `/escalations/stats` | Escalation statistics |
| `GET` | `/escalations/{id}` | Get escalation details |
| `POST` | `/escalations/{id}/resolve` | Resolve an escalation |

### Execute Request

```bash
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Generate a JSON user profile with name, age, and email",
    "response_schema": {
      "type": "object",
      "required": ["name", "age", "email"],
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "email": {"type": "string"}
      }
    },
    "config": {
      "max_retries": 3,
      "max_repairs": 2,
      "fallback_models": ["groq/llama-3.1-8b-instant"],
      "template_response": {"name": "Unknown", "age": 0, "email": "n/a"}
    }
  }'
```

---

## Library Usage (Without Server)

```python
import asyncio
from llm_shield import Shield, ShieldRequest, ShieldConfig

async def main():
    shield = Shield(config=ShieldConfig(default_model="groq/llama-3.1-70b-versatile"))

    response = await shield.execute(ShieldRequest(
        prompt="Generate a JSON user profile",
        response_schema={
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        },
    ))

    print(f"Status: {response.status}")
    print(f"Result: {response.result}")
    print(f"LLM calls made: {response.execution_trace.total_llm_calls}")

asyncio.run(main())
```

---

## State Machine

```
INIT → CALLING_LLM → VALIDATING → SUCCEEDED ✅
                   ↘ RETRYING → REPAIRING → FALLING_BACK → TEMPLATE_FALLBACK → ESCALATING → FAILED ❌
```

| Config | Default | Description |
|:-------|:--------|:------------|
| `max_retries` | 3 | Retries before repair |
| `max_repairs` | 2 | Repair attempts before fallback |
| `timeout_seconds` | 30 | Per-call timeout |
| `fallback_models` | `["groq/llama-3.1-8b-instant", "groq/gemma2-9b-it"]` | Fallback model chain |
| `escalation_mode` | `"in_memory"` | `"in_memory"`, `"webhook"`, or `"log"` |

---

## Configuration

Set via environment variables (prefixed with `SHIELD_`) or per-request overrides:

```env
GROQ_API_KEY=gsk_your_key_here
SHIELD_DEFAULT_MODEL=groq/llama-3.1-70b-versatile
SHIELD_FALLBACK_MODELS=["groq/llama-3.1-8b-instant","groq/gemma2-9b-it"]
SHIELD_MAX_RETRIES=3
SHIELD_MAX_REPAIRS=2
SHIELD_TIMEOUT_SECONDS=30
SHIELD_ESCALATION_MODE=in_memory
```

---

## Architecture

```
llm_shield/          # The library (pip-installable)
├── engine.py        # Core state machine orchestrator
├── states.py        # State & transition definitions
├── models.py        # Pydantic data models
├── validators.py    # Pluggable validation (JSON schema, regex, length)
├── providers.py     # LLM abstraction (Groq via litellm)
├── repair.py        # Repair strategies (regex + LLM self-repair)
├── fallback.py      # Fallback chain + template fallback
├── escalation.py    # Human escalation handlers
├── config.py        # Configuration management
├── trace.py         # Execution tracing
└── exceptions.py    # Exception hierarchy

server/              # FastAPI API layer
├── app.py           # Application factory
├── dependencies.py  # Dependency injection
└── routes/          # API endpoints

tests/               # Comprehensive test suite
postman/             # Postman collection
```

## License

MIT
