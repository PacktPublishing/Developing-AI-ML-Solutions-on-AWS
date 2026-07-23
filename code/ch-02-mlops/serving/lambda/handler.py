"""Lambda handler that scores applications with the challenger model.

The model is loaded once at cold start (module import) and reused across warm
invocations — the same ChallengerModel the BYOC container serves, so a request
gets the same score whether it hits the SageMaker endpoint or this function.

Accepts an API Gateway proxy event (JSON body) or a direct invoke (the records
themselves): a record, a list of records, or {"instances": [...]}.
"""

import json

from challenger_model import ChallengerModel

# Cold-start work: load the model from the image (baked at /opt/ml/model).
_model = ChallengerModel.load("/opt/ml/model")


def lambda_handler(event, context):
    """Score a batch of applications and return probability of default per row."""
    body = event.get("body", event) if isinstance(event, dict) else event
    if isinstance(body, str):
        body = json.loads(body)
    records = (
        body["instances"] if isinstance(body, dict) and "instances" in body else body
    )
    if isinstance(records, dict):
        records = [records]
    pd_default = _model.predict_proba(records)
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"pd": pd_default}),
    }
