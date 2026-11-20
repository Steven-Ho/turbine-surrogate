"""Timestamped, immediately flushed training progress without device synchronization."""
import builtins
from datetime import datetime
import os
from zoneinfo import ZoneInfo


def print(*values, **kwargs):
    text = kwargs.pop('sep', ' ').join(str(value) for value in values)
    kwargs.setdefault('flush', True)
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
    builtins.print('\n'.join(f'[{stamp}] {line}' for line in text.split('\n')), **kwargs)


def log_every():
    value = int(os.environ.get('FANN_LOG_EVERY', '10'))
    if value < 1:
        raise ValueError('FANN_LOG_EVERY must be positive')
    return value


def should_log(batch, total, interval):
    return batch == 1 or batch % interval == 0 or batch == total
