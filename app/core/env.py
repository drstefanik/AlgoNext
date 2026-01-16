import os
from pathlib import Path

from dotenv import load_dotenv


def is_production_env() -> bool:
    return (
        os.getenv("APP_ENV", "").lower() == "production"
        or os.getenv("ENV", "").lower() == "production"
    )


def load_env() -> None:
    env_path = Path(os.getenv("ENV_FILE", ".env"))
    load_dotenv(dotenv_path=env_path, override=False)
    if is_production_env() and not (os.getenv("MINIO_PUBLIC_ENDPOINT") or "").strip():
        raise RuntimeError(
            "MINIO_PUBLIC_ENDPOINT must be set when running in production."
        )
