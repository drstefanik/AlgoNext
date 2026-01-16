"""Add preview_frames to analysis_jobs.

Revision ID: 20260116_add_preview_frames
Revises: 20250923_add_failure_reason
Create Date: 2026-01-16 18:28:39.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20260116_add_preview_frames"
down_revision = "20250923_add_failure_reason"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "analysis_jobs",
        sa.Column(
            "preview_frames",
            sa.JSON(),
            nullable=False,
            server_default=sa.text("'[]'"),
        ),
    )
    op.alter_column("analysis_jobs", "preview_frames", server_default=None)


def downgrade() -> None:
    op.drop_column("analysis_jobs", "preview_frames")
