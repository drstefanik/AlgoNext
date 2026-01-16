"""Add failure_reason to analysis_jobs.

Revision ID: 20250923_add_failure_reason
Revises: 20250923_add_player_ref
Create Date: 2025-09-23 00:10:00.000000
"""

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "20250923_add_failure_reason"
down_revision = "20250923_add_player_ref"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("analysis_jobs", sa.Column("failure_reason", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("analysis_jobs", "failure_reason")
