import os
from pathlib import Path

from dotenv import load_dotenv


def load_env() -> None:
    env_path = Path(os.getenv("ENV_FILE", ".env"))
    load_dotenv(dotenv_path=env_path, override=False)
