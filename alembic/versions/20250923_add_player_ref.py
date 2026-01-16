"""Add player_ref to analysis_jobs.

Revision ID: 20250923_add_player_ref
Revises: 
Create Date: 2025-09-23 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy import inspect


# revision identifiers, used by Alembic.
revision = "20250923_add_player_ref"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {col["name"] for col in insp.get_columns("analysis_jobs")}
    if "player_ref" not in cols:
        op.add_column("analysis_jobs", sa.Column("player_ref", sa.Text(), nullable=True))


def downgrade() -> None:
    bind = op.get_bind()
    insp = inspect(bind)
    cols = {col["name"] for col in insp.get_columns("analysis_jobs")}
    if "player_ref" in cols:
        op.drop_column("analysis_jobs", "player_ref")
