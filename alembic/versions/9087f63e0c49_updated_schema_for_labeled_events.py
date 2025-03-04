"""Updated schema for labeled events

Revision ID: 9087f63e0c49
Revises: f056e5a2084c
Create Date: 2025-03-04 08:20:52.623304

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '9087f63e0c49'
down_revision: Union[str, None] = 'f056e5a2084c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('logs', sa.Column('event', sa.String(), nullable=True))
    op.drop_index('ix_logs_event_type', table_name='logs')
    op.create_index(op.f('ix_logs_event'), 'logs', ['event'], unique=False)
    op.drop_column('logs', 'file_accesses')
    op.drop_column('logs', 'connection_attempts')
    op.drop_column('logs', 'anomaly_detected')
    op.drop_column('logs', 'failed_logins')
    op.drop_column('logs', 'threat_level')
    op.drop_column('logs', 'event_type')
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('logs', sa.Column('event_type', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('logs', sa.Column('threat_level', sa.VARCHAR(), autoincrement=False, nullable=True))
    op.add_column('logs', sa.Column('failed_logins', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('logs', sa.Column('anomaly_detected', sa.BOOLEAN(), autoincrement=False, nullable=True))
    op.add_column('logs', sa.Column('connection_attempts', sa.INTEGER(), autoincrement=False, nullable=True))
    op.add_column('logs', sa.Column('file_accesses', sa.INTEGER(), autoincrement=False, nullable=True))
    op.drop_index(op.f('ix_logs_event'), table_name='logs')
    op.create_index('ix_logs_event_type', 'logs', ['event_type'], unique=False)
    op.drop_column('logs', 'event')
    # ### end Alembic commands ###
