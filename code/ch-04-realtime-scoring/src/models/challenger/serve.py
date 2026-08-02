"""SageMaker inference container for the scorecard model.

Serves the contract SageMaker requires of a bring-your-own container — GET
/ping for health and POST /invocations for scoring — with FastAPI/uvicorn.
https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html
"""

import json

from fastapi import FastAPI, Request, Response, status
from model import probability_of_default

app = FastAPI(title="challenger-model")


@app.get("/ping")
def ping() -> Response:
    """Health check: 200 once the model is ready to serve."""
    return Response(status_code=status.HTTP_200_OK)


@app.post("/invocations")
async def invocations(request: Request) -> Response:
    """Score one application's features and return {"pd": ...}."""
    features = json.loads(await request.body())
    pd = probability_of_default(features)
    return Response(
        content=json.dumps({"pd": round(pd, 6)}),
        media_type="application/json",
        status_code=status.HTTP_200_OK,
    )
