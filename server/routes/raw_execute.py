import asyncio
import json
import random
from typing import Any

from fastapi import APIRouter, HTTPException
from litellm import acompletion
from pydantic import BaseModel

from llm_shield.config import ShieldConfig

router = APIRouter(prefix="/raw-execute", tags=["Raw Execution (Unprotected)"])

class RawRequest(BaseModel):
    prompt: str
    response_schema: dict[str, Any] | None = None
    model: str | None = None
    chaos_mode: str = "false"

@router.post("")
async def raw_execute(request: RawRequest):
    """
    Simulate a raw LLM call without LLM Shield.
    This demonstrates what happens when there is NO safety net.
    """
    config = ShieldConfig()
    model = request.model or config.default_model

    # Simulated Chaos (only if requested, for dramatic demo effect)
    fault_type = request.chaos_mode.lower()
    if fault_type.startswith("healable_"):
        fault_type = fault_type.replace("healable_", "")
    
    if fault_type != "false" and fault_type != "none":
        if fault_type == "timeout":
            await asyncio.sleep(5)
            raise HTTPException(status_code=504, detail="Raw LLM call timed out after 5 seconds")
        elif fault_type == "server_error" or fault_type == "500":
            raise HTTPException(status_code=500, detail="Raw LLM call failed with HTTP 500 Internal Server Error")
        if fault_type == "malformed_json":
            chaos_sys_prompt = (
                "You are a helpful data extraction API. "
                "You must return ONLY valid JSON matching the requested schema. "
                "Do NOT wrap the JSON in markdown blocks (e.g. ```json). "
                "Return the raw JSON string directly."
            )
            if request.response_schema:
                chaos_sys_prompt += f"\n\nRequired Schema:\n{json.dumps(request.response_schema)}"
                
            return {
                "status": "FAILED",
                "result": None,
                "error": "Failed to parse JSON: Expecting ',' delimiter: line 1 column 33",
                "raw_output": '{"passenger_name": "Alex Mercer" "flight_class": "FIRST"}',
                "payload_sent": [
                    {"role": "system", "content": chaos_sys_prompt},
                    {"role": "user", "content": request.prompt}
                ]
            }

    # Real LLM Call (if no chaos was injected)
    # NOTE: Intentionally does NOT pass the schema to the LLM.
    # This simulates a naive/typical developer who calls the LLM API directly
    # and simply checks if the output is valid JSON — no schema enforcement.
    try:
        system_prompt = (
            "You are a helpful data extraction API. "
            "Return ONLY a valid JSON object matching the user's request. "
            "Do NOT wrap the JSON in markdown blocks (e.g. ```json). "
            "Output raw JSON only."
        )
        if request.response_schema:
            system_prompt += f"\n\nYou MUST adhere strictly to this JSON Schema:\n{json.dumps(request.response_schema, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.prompt}
        ]

        response = await acompletion(
            model=model,
            messages=messages,
        )
        
        # Raw response text
        content = response.choices[0].message.content or ""
        
        # Best-effort markdown stripping (what a typical dev would quickly hack in)
        content = content.strip()
        
        import re
        if "<think>" in content and "</think>" in content:
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            
        if content.startswith("```json"):
            content = content[7:]
        elif content.startswith("```"):
            content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()
        
        # Attempt to parse as JSON just to see if it's valid
        is_valid_json = False
        parsed_result = None
        error_msg = None
        
        try:
            parsed_result = json.loads(content)
            is_valid_json = True
        except Exception as e:
            error_msg = f"Failed to parse JSON: {e}"

        if not is_valid_json:
             return {
                "status": "FAILED",
                "result": None,
                "error": error_msg,
                "raw_output": content,
                "payload_sent": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": request.prompt}
                ]
            }
            
        return {
            "status": "SUCCEEDED",
            "result": parsed_result,
            "error": None,
            "raw_output": content,
            "payload_sent": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.prompt}
            ]
        }

    except Exception as e:
        # Without a harness, an exception just crashes out
        raise HTTPException(status_code=500, detail=f"Raw LLM call crashed: {e}")
