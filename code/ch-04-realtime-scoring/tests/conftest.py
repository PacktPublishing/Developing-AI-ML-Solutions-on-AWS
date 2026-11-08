"""Test setup for the gateway.

Set ENDPOINTS before importing the gateway, which reads it at module load.
"""

import os

os.environ.setdefault("ENDPOINTS", "scorecard=http://a,challenger=http://b")
