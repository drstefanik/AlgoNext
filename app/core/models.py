from datetime import datetime

from sqlalchemy import String, DateTime, Text, JSON, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.db import Base


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[str] = mapped_column(String, primary_key=True)

    status: Mapped[str] = mapped_column(
        String,
        default="WAITING_FOR_SELECTION",
        nullable=False,
    )

    category: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)

    video_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_bucket: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_key: Mapped[str | None] = mapped_column(Text, nullable=True)
    video_meta: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    target: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    anchor: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)

    player_ref: Mapped[dict | None] = mapped_column(JSONB, nullable=True)

    progress: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    result: Mapped[dict] = mapped_column(JSON, default=dict, nullable=False)
    preview_frames: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )
    warnings: Mapped[list] = mapped_column(
        JSONB,
        default=list,
        nullable=False,
        server_default=text("'[]'::jsonb"),
    )

    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    failure_reason: Mapped[str | None] = mapped_column(Text, nullable=True)

    # timestamps robusti (lato DB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
