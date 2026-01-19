"""Add video bucket/key and allow null video_url.

Revision ID: 20260322_add_video_bucket_key_nullable_url
Revises: 20260321_merge_heads
Create Date: 2026-03-22 09:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260322_add_video_bucket_key_nullable_url"
down_revision = "20260321_merge_heads"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS video_bucket TEXT;

        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS video_key TEXT;

        ALTER TABLE analysis_jobs
          ALTER COLUMN video_url DROP NOT NULL;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        UPDATE analysis_jobs
        SET video_url = ''
        WHERE video_url IS NULL;

        ALTER TABLE analysis_jobs
          ALTER COLUMN video_url SET NOT NULL;

        ALTER TABLE analysis_jobs
          DROP COLUMN IF EXISTS video_key;

        ALTER TABLE analysis_jobs
          DROP COLUMN IF EXISTS video_bucket;
        """
    )
