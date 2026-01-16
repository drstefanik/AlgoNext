import logging
import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.engine.url import make_url
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.env import load_env
from app.core.db import Base, SessionLocal, engine, DATABASE_URL
from app.api import router as api_router

load_env()

APP_ENV = os.getenv("APP_ENV", "development").lower()
DOCS_URL = None if APP_ENV == "production" else "/docs"
REDOC_URL = None if APP_ENV == "production" else "/redoc"
OPENAPI_URL = None if APP_ENV == "production" else "/openapi.json"
logger = logging.getLogger(__name__)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        response.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=()")
        response.headers.setdefault("Cross-Origin-Resource-Policy", "same-origin")
        return response


app = FastAPI(
    title="FNX Video AI Backend",
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL,
    openapi_url=OPENAPI_URL,
)

app.add_middleware(SecurityHeadersMiddleware)

cors_allowed_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ALLOW_ORIGINS", "").split(",")
    if origin.strip()
]
cors_allow_origin_regex = os.getenv(
    "CORS_ALLOW_ORIGIN_REGEX",
    r"https://.*\.vercel\.app",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins,
    allow_origin_regex=cors_allow_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if APP_ENV == "production":
    allowed_hosts_env = os.getenv("ALLOWED_HOSTS", "")
    allowed_hosts = [host.strip() for host in allowed_hosts_env.split(",") if host.strip()]
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

def init_db():
    for _ in range(30):  # ~30s
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError:
            time.sleep(1)
    Base.metadata.create_all(bind=engine)

init_db()

def mask_database_url(url: str) -> str:
    try:
        return make_url(url).render_as_string(hide_password=True)
    except Exception:
        return "<invalid DATABASE_URL>"

@app.on_event("startup")
def fail_fast_db_check():
    logger.info("DATABASE_URL: %s", mask_database_url(DATABASE_URL))
    logger.info(
        "S3 config: S3_ENDPOINT_URL=%s S3_PUBLIC_ENDPOINT_URL=%s S3_BUCKET=%s",
        os.environ.get("S3_ENDPOINT_URL"),
        os.environ.get("S3_PUBLIC_ENDPOINT_URL"),
        os.environ.get("S3_BUCKET"),
    )
    session = SessionLocal()
    try:
        session.execute(text("SELECT 1"))
    except Exception:
        logger.exception("Database connectivity check failed during startup.")
        raise
    finally:
        session.close()

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

app.include_router(api_router)
