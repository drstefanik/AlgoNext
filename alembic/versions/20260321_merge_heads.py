"""Merge heads after widening alembic_version.

Revision ID: 20260321_merge_heads
Revises: 20260116_widen_alembic_version, 20260320_add_error_column
Create Date: 2026-03-21 10:00:00.000000
"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "20260321_merge_heads"
down_revision = (
    "20260116_widen_alembic_version",
    "20260320_add_error_column",
)
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
