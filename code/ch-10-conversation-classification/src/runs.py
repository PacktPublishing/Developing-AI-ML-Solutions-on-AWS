# /// script
# dependencies = []
# ///
"""One identifier per run, shared by both routers so their outputs can be told apart."""

import time
import uuid


def new_run_id() -> str:
    """Return an identifier unique to one run, legal in a Bedrock job name."""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
