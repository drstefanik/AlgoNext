"""Add player_ref to analysis_jobs.

Revision ID: 20250923_add_player_ref
Revises: 
Create Date: 2025-09-23 00:00:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20250923_add_player_ref"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("player_ref", sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "player_ref")
