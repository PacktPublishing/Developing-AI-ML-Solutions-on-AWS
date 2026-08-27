"""Test setup for the gateway.

The gateway reads ENDPOINTS at import time, so it has to be set before app is
imported. Doing that here keeps the test module's imports at the top of the file
where they belong.
"""

import os

os.environ.setdefault("ENDPOINTS", "scorecard=http://a,challenger=http://b")
