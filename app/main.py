import time
from fastapi import FastAPI
from sqlalchemy.exc import OperationalError

from app.core.db import Base, engine
from app.api import router as api_router

app = FastAPI(title="FNX Video AI Backend")

def init_db():
    for _ in range(30):  # ~30s
        try:
            Base.metadata.create_all(bind=engine)
            return
        except OperationalError:
            time.sleep(1)
    Base.metadata.create_all(bind=engine)

init_db()

@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(api_router)
