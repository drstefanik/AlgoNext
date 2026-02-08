import json
import os
from typing import Any, Dict, List

from openai import OpenAI


AI_REPORT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "headline": {"type": "string"},
        "summary_bullets": {
            "type": "array",
            "minItems": 3,
            "maxItems": 5,
            "items": {"type": "string"},
        },
        "strengths": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "key_moments": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "clip_url": {"type": "string"},
                    "startSec": {"type": "number"},
                    "endSec": {"type": "number"},
                },
                "required": ["title", "why", "clip_url", "startSec", "endSec"],
            },
        },
        "training_plan_14d": {"type": "array", "items": {"type": "string"}},
        "limitations": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
    "required": [
        "headline",
        "summary_bullets",
        "strengths",
        "risks",
        "key_moments",
        "training_plan_14d",
        "limitations",
        "confidence",
    ],
}


SYSTEM_PROMPT = (
    "Sei un AI scout calcistico. Genera un report stile eyeball.club con tono "
    "professionale e conciso in italiano. Usa solo i dati forniti (nessun dato palla/"
    "eventi; non inventare etichette tecniche o tattiche non osservabili). Se mancano dati, "
    "esplicita i limiti. I key moments devono riferirsi solo ai clip_url forniti."
)


def _extract_output_json(response: Any) -> Dict[str, Any]:
    if hasattr(response, "output_text") and response.output_text:
        return json.loads(response.output_text)
    outputs = getattr(response, "output", []) or []
    for output in outputs:
        for content in getattr(output, "content", []) or []:
            content_type = getattr(content, "type", None)
            if content_type == "output_text":
                text = getattr(content, "text", None)
                if text:
                    return json.loads(text)
            if hasattr(content, "json") and content.json:
                return content.json
    raise ValueError("No JSON output found in OpenAI response.")


def generate_ai_report(payload: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any] | None]:
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    model = (os.environ.get("OPENAI_MODEL") or "gpt-5.2").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip() or None

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(payload, ensure_ascii=False),
            },
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "ai_scout_report",
                "schema": AI_REPORT_SCHEMA,
                "strict": True,
            },
        },
    )
    usage = getattr(response, "usage", None)
    return _extract_output_json(response), usage
