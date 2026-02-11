"""Add async report columns to analysis_jobs.

Revision ID: 20260410_add_async_report_columns
Revises: 20260405_add_ai_report
Create Date: 2026-04-10
"""

from alembic import op


revision = "20260410_add_async_report_columns"
down_revision = "20260405_add_ai_report"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          ADD COLUMN IF NOT EXISTS report JSONB,
          ADD COLUMN IF NOT EXISTS report_status VARCHAR NOT NULL DEFAULT 'PENDING',
          ADD COLUMN IF NOT EXISTS report_error TEXT;
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE analysis_jobs
          DROP COLUMN IF EXISTS report_error,
          DROP COLUMN IF EXISTS report_status,
          DROP COLUMN IF EXISTS report;
        """
    )
