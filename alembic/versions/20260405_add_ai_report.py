"""Add ai_report column to analysis_jobs.

Revision ID: 20260405_add_ai_report
Revises: 20260401_add_analysis_job_warnings
Create Date: 2026-04-05 09:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260405_add_ai_report"
down_revision = "20260401_add_analysis_job_warnings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS ai_report JSONB;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          DROP COLUMN IF EXISTS ai_report;
        """
    )
