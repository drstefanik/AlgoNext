"""Add warnings column to analysis_jobs.

Revision ID: 20260401_add_analysis_job_warnings
Revises: 20260322_add_video_bucket_key_nullable_url
Create Date: 2026-04-01 09:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260401_add_analysis_job_warnings"
down_revision = "20260322_add_video_bucket_key_nullable_url"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS warnings JSONB NOT NULL DEFAULT '[]'::jsonb;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          DROP COLUMN IF EXISTS warnings;
        """
    )
