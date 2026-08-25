import asyncio
from llm_shield import Shield, ShieldRequest

async def test_groq():
    # Initializes Shield (loads GROQ_API_KEY from .env)
    shield = Shield()

    print("Sending request to Groq via Shield control loop...")
    response = await shield.execute(ShieldRequest(
        prompt="Generate a JSON profile for a user named Alice who is 30 years old.",
        response_schema={
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            }
        }
    ))

    print(f"\nStatus: {response.status}")
    print(f"Result: {response.result}")
    print(f"LLM Calls Made: {response.execution_trace.total_llm_calls}")
    print(f"Total Duration: {response.execution_trace.total_duration_ms} ms")

asyncio.run(test_groq())
