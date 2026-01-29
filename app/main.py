import logging
import os
import time
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
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


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-Id") or str(uuid4())
        request.state.request_id = request_id
        start_time = time.monotonic()
        response = await call_next(request)
        response.headers["X-Request-Id"] = request_id
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            "API_REQUEST",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(duration_ms, 2),
                "request_id": request_id,
            },
        )
        return response


app = FastAPI(
    title="FNX Video AI Backend",
    docs_url=DOCS_URL,
    redoc_url=REDOC_URL,
    openapi_url=OPENAPI_URL,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RequestContextMiddleware)

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


def _error_payload(code: str, message: str, details: dict | None = None) -> dict:
    payload = {"code": code, "message": message}
    if details:
        payload["details"] = details
    return payload


def _meta_payload(request: Request) -> dict:
    request_id = getattr(request.state, "request_id", None)
    return {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    payload_keys: list[str] = []
    if request.method == "POST" and request.url.path == "/jobs":
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            payload_keys = list(body.keys())
        request_id = getattr(request.state, "request_id", None)
        logger.info(
            "JOBS_PAYLOAD_KEYS",
            extra={"payload_keys": payload_keys, "request_id": request_id},
        )
    return JSONResponse(
        status_code=422,
        content={
            "ok": False,
            "error": _error_payload(
                "VALIDATION_ERROR",
                "Request validation failed",
                {"errors": exc.errors(), "payload_keys": payload_keys},
            ),
            "meta": _meta_payload(request),
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception during request.")
    return JSONResponse(
        status_code=500,
        content={
            "ok": False,
            "error": _error_payload("INTERNAL_ERROR", "Unexpected server error"),
            "meta": _meta_payload(request),
        },
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    detail = exc.detail
    if isinstance(detail, dict):
        code = detail.get("code") or "HTTP_ERROR"
        message = detail.get("message") or "Request failed"
        details = detail.get("details")
    else:
        code = "HTTP_ERROR"
        message = str(detail)
        details = None
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "ok": False,
            "error": _error_payload(code, message, details),
            "meta": _meta_payload(request),
        },
    )

def init_db():
    for _ in range(30):  # ~30s
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError:
            time.sleep(1)
    try:
        Base.metadata.create_all(bind=engine)
    except OperationalError:
        logger.warning("Database unavailable during init_db; continuing without DB.")

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
    finally:
        session.close()

@app.get("/health", include_in_schema=False)
def health():
    return {
        "ok": True,
        "service": "algonext-api",
        "ts": datetime.now(timezone.utc).isoformat(),
    }

app.include_router(api_router)
