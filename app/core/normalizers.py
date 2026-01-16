import json
from typing import Any, Optional


def normalize_failure_reason(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (dict, list, tuple)):
        if not value:
            return None
        return json.dumps(value, ensure_ascii=False)
    return str(value)
