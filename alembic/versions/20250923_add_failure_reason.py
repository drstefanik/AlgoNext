"""Add failure_reason to analysis_jobs.

Revision ID: 20250923_add_failure_reason
Revises: 20250923_add_player_ref
Create Date: 2025-09-23 00:10:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "20250923_add_failure_reason"
down_revision = "20250923_add_player_ref"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {col["name"] for col in insp.get_columns("analysis_jobs")}
    if "failure_reason" not in cols:
        op.add_column("analysis_jobs", sa.Column("failure_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {col["name"] for col in insp.get_columns("analysis_jobs")}
    if "failure_reason" in cols:
        op.drop_column("analysis_jobs", "failure_reason")
