from alembic import op

revision = "20260116_widen_alembic_version"
down_revision = "20260116_add_preview_frames"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(255)"
    )


def downgrade() -> None:
    # Non safe fare downgrade a 32: potresti troncare valori reali.
    pass
