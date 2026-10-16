"""
Shared logging setup for WPO training/test scripts.
Redirects stdout/stderr to both the terminal and a file in log/.
"""

import os
import sys


class _Tee:
    """Duplicates writes to multiple streams with immediate flush on file streams."""

    def __init__(self, *streams, file_streams=None):
        self.streams = streams
        self.file_streams = file_streams or []

    def write(self, data):
        for s in self.streams:
            s.write(data)
        # Immediately flush file streams to prevent data loss on crash
        for f in self.file_streams:
            f.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(log_name: str):
    os.makedirs('log', exist_ok=True)
    log_path = os.path.join('log', log_name)
    log_file = open(log_path, 'a', buffering=1, encoding='utf-8')
    sys.stdout = _Tee(sys.__stdout__, log_file, file_streams=[log_file])
    sys.stderr = _Tee(sys.__stderr__, log_file, file_streams=[log_file])
    print(f'[Log] output also written to {log_path}')
    return log_path