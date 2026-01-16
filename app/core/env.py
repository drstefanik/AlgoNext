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

    def _apply_env_alias(primary: str, aliases: list[str]) -> None:
        value = (os.getenv(primary) or "").strip()
        if not value:
            for alias in aliases:
                alias_value = (os.getenv(alias) or "").strip()
                if alias_value:
                    value = alias_value
                    os.environ[primary] = alias_value
                    break
        if value:
            for alias in aliases:
                if not (os.getenv(alias) or "").strip():
                    os.environ[alias] = value

    _apply_env_alias("S3_ENDPOINT_URL", ["MINIO_INTERNAL_ENDPOINT"])
    _apply_env_alias("S3_PUBLIC_ENDPOINT_URL", ["MINIO_PUBLIC_ENDPOINT"])
    _apply_env_alias("S3_ACCESS_KEY", ["MINIO_ACCESS_KEY"])
    _apply_env_alias("S3_SECRET_KEY", ["MINIO_SECRET_KEY"])
    _apply_env_alias("S3_BUCKET", ["MINIO_BUCKET"])
    _apply_env_alias("S3_REGION", ["MINIO_REGION"])

    if is_production_env() and not (os.getenv("S3_PUBLIC_ENDPOINT_URL") or "").strip():
        raise RuntimeError(
            "S3_PUBLIC_ENDPOINT_URL must be set when running in production."
        )
