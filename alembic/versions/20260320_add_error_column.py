"""Add error column to analysis_jobs.

Revision ID: 20260320_add_error_column
Revises: 20260318_ensure_analysis_job_columns
Create Date: 2026-03-20 10:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260320_add_error_column"
down_revision = "20260318_ensure_analysis_job_columns"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS error TEXT;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          DROP COLUMN IF EXISTS error;
        """
    )
