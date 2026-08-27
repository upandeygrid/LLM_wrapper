import asyncio
import json
import httpx

async def run():
    payload = {
        "prompt": "Extract the user's birth date from this text: \"I was born on January 15th, 1995.\"\n\nCRITICAL INSTRUCTION: You must extract the date EXACTLY as it is written in the text above. Do not change the formatting.",
        "response_schema": {
            "type": "object",
            "required": ["birth_date"],
            "properties": {
                "birth_date": { 
                    "type": "string", 
                    "pattern": "^[0-9]{4}-[0-9]{2}-[0-9]{2}$"
                }
            }
        },
        "system_prompt": "You are a helpful data extraction API. Return ONLY a valid JSON object matching the user's request. Do NOT wrap the JSON in markdown blocks. Output raw JSON only."
    }

    async with httpx.AsyncClient() as client:
        res = await client.post("http://localhost:8000/execute", json=payload, timeout=120)
        data = res.json()
        print(f"Status: {data.get('status')}")
        
        trace = data.get("execution_trace", {})
        print("\n--- LLM Calls ---")
        for c in trace.get("llm_calls", []):
            print(f"Phase: {c.get('phase')}, Attempt: {c.get('attempt')}")
            print(f"Response: {c.get('response')}")
        
        print("\n--- Validation Errors ---")
        for e in trace.get("validation_errors", [])[-5:]:
            print(e)
            
asyncio.run(run())
