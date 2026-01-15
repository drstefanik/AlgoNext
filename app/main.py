import os
import time

from fastapi import FastAPI
from sqlalchemy.exc import OperationalError
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.core.db import Base, engine
from app.api import router as api_router

APP_ENV = os.getenv("APP_ENV", "development").lower()
DOCS_URL = None if APP_ENV == "production" else "/docs"
REDOC_URL = None if APP_ENV == "production" else "/redoc"
OPENAPI_URL = None if APP_ENV == "production" else "/openapi.json"


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

@app.get("/health", include_in_schema=False)
def health():
    return {"status": "ok"}

app.include_router(api_router)
