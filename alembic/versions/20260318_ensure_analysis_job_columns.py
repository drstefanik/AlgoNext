"""Ensure analysis_jobs columns are present and aligned.

Revision ID: 20260318_ensure_analysis_job_columns
Revises: 20260116_widen_alembic_version
Create Date: 2026-03-18 12:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260318_ensure_analysis_job_columns"
down_revision = "20260116_widen_alembic_version"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS player_ref TEXT;

        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS failure_reason TEXT;

        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS preview_frames JSONB NOT NULL DEFAULT '[]'::jsonb;
        """
    )
    op.execute(
        """
        UPDATE analysis_jobs
        SET preview_frames = '[]'::jsonb
        WHERE preview_frames IS NULL;
        """
    )
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ALTER COLUMN preview_frames TYPE JSONB USING preview_frames::jsonb,
          ALTER COLUMN preview_frames SET DEFAULT '[]'::jsonb,
          ALTER COLUMN preview_frames SET NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ALTER COLUMN preview_frames TYPE JSON USING preview_frames::json,
          ALTER COLUMN preview_frames DROP DEFAULT;
        """
    )
