# /// script
# requires-python = ">=3.12"
# dependencies = ["starlette", "uvicorn", "croniter"]
# ///
"""A local EventBridge and SNS: rules, schedules, topics, one service.

EventBridge (aws-json on X-Amz-Target) serves PutRule, PutTargets,
ListRules, ListTargetsByRule, DeleteRule, RemoveTargets. SNS (query
protocol, XML) serves CreateTopic, Subscribe, Publish, ListTopics,
DeleteTopic. A scheduler thread fires rules on their rate() or cron()
expression; SNS-topic targets deliver through the SNS plane, every other
target is printed with its event — locally the reader sees when the
pipeline WOULD start, and starts it (on AWS, EventBridge holds the
StartPipelineExecution role).

Subscriptions with protocol http/https are POSTed the message; any other
protocol is printed. Everything is in-memory.

Usage:
  uv run events-local/app.py          # serves on :8009
"""

import json
import os
import threading
import time
import urllib.parse
import urllib.request
import uuid
from datetime import UTC, datetime

import uvicorn
from croniter import croniter
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

REGION = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
PORT = int(os.environ.get("EVENTS_PORT", "8009"))
ACCOUNT = "000000000000"

rules: dict[str, dict] = {}  # name -> {"schedule", "targets": [...]}
topics: dict[str, dict] = {}  # arn -> {"name", "subscriptions": [...]}


def rule_arn(name: str) -> str:
    """Build a rule ARN."""
    return f"arn:aws:events:{REGION}:{ACCOUNT}:rule/{name}"


def topic_arn(name: str) -> str:
    """Build a topic ARN."""
    return f"arn:aws:sns:{REGION}:{ACCOUNT}:{name}"


def next_fire(expression: str, after: float) -> float | None:
    """Return the next epoch a rate() or cron() expression fires after `after`."""
    if expression.startswith("rate("):
        value, unit = expression[5:-1].split()
        seconds = (
            int(value)
            * {
                "minute": 60,
                "minutes": 60,
                "hour": 3600,
                "hours": 3600,
                "day": 86400,
                "days": 86400,
            }[unit]
        )
        return after + seconds
    if expression.startswith("cron("):
        # AWS cron has six fields (minute hour day month weekday year);
        # croniter takes five, so drop the year and map ? to *
        fields = expression[5:-1].replace("?", "*").split()
        return croniter(
            " ".join(fields[:5]), datetime.fromtimestamp(after, UTC)
        ).get_next(float)
    return None


def deliver_sns(arn: str, subject: str, message: str) -> None:
    """Deliver one message to a topic's subscriptions."""
    for sub in topics.get(arn, {}).get("subscriptions", []):
        if sub["Protocol"] in ("http", "https"):
            body = json.dumps(
                {"Type": "Notification", "Subject": subject, "Message": message}
            ).encode()
            request = urllib.request.Request(
                sub["Endpoint"], data=body, headers={"Content-Type": "application/json"}
            )
            try:
                urllib.request.urlopen(request, timeout=5)
            except OSError as e:
                print(f"[sns] delivery to {sub['Endpoint']} failed: {e}", flush=True)
        else:
            print(
                f"[sns] {arn} -> {sub['Protocol']}:{sub['Endpoint']}: "
                f"{subject or ''} {message}",
                flush=True,
            )


def fire(name: str, rule: dict) -> None:
    """Fire one rule: SNS targets deliver, everything else is printed."""
    event = {
        "source": "events-local",
        "detail-type": "Scheduled Event",
        "time": datetime.now(UTC).isoformat(),
        "resources": [rule_arn(name)],
    }
    for target in rule["targets"]:
        arn = target["Arn"]
        if arn.startswith("arn:aws:sns:"):
            deliver_sns(arn, f"rule {name}", json.dumps(event))
        else:
            print(
                f"[events] rule {name} fired -> {arn} {json.dumps(event)}", flush=True
            )


def scheduler() -> None:
    """Fire every enabled scheduled rule at its time."""
    fire_at: dict[str, float] = {}
    while True:
        now = time.time()
        for name, rule in list(rules.items()):
            expression = rule.get("schedule")
            if not expression or rule.get("state") == "DISABLED":
                continue
            when = fire_at.get(name)
            if when is None:
                when = next_fire(expression, now)
            if when is None:
                continue
            if now >= when:
                fire(name, rule)
                when = next_fire(expression, now) or now + 60.0
            fire_at[name] = when
        time.sleep(1)


async def events_api(request: Request, operation: str, body: dict) -> JSONResponse:
    """Serve the EventBridge operations."""
    name = body.get("Name") or body.get("Rule", "")
    if operation == "PutRule":
        rules.setdefault(name, {"targets": []}).update(
            schedule=body.get("ScheduleExpression"),
            pattern=body.get("EventPattern"),
            state=body.get("State", "ENABLED"),
        )
        return JSONResponse({"RuleArn": rule_arn(name)})
    if operation == "PutTargets":
        rules[name]["targets"] = body.get("Targets", [])
        return JSONResponse({"FailedEntryCount": 0, "FailedEntries": []})
    if operation == "ListRules":
        return JSONResponse(
            {
                "Rules": [
                    {
                        "Name": n,
                        "Arn": rule_arn(n),
                        "ScheduleExpression": r.get("schedule"),
                        "State": r.get("state", "ENABLED"),
                    }
                    for n, r in sorted(rules.items())
                ]
            }
        )
    if operation == "ListTargetsByRule":
        return JSONResponse({"Targets": rules.get(name, {}).get("targets", [])})
    if operation == "RemoveTargets":
        ids = set(body.get("Ids", []))
        rules[name]["targets"] = [
            t for t in rules[name]["targets"] if t["Id"] not in ids
        ]
        return JSONResponse({"FailedEntryCount": 0, "FailedEntries": []})
    if operation == "DeleteRule":
        rules.pop(name, None)
        return JSONResponse({})
    return JSONResponse(
        {"__type": "UnknownOperationException", "message": operation}, status_code=400
    )


def sns_xml(action: str, inner: str) -> Response:
    """Wrap one SNS query-protocol response."""
    return Response(
        f'<{action}Response xmlns="http://sns.amazonaws.com/doc/2010-03-31/">'
        f"<{action}Result>{inner}</{action}Result>"
        f"<ResponseMetadata><RequestId>{uuid.uuid4()}</RequestId></ResponseMetadata>"
        f"</{action}Response>",
        media_type="text/xml",
    )


def sns_api(form: dict) -> Response:
    """Serve the SNS operations."""
    action = form.get("Action", "")
    if action == "CreateTopic":
        arn = topic_arn(form["Name"])
        topics.setdefault(arn, {"name": form["Name"], "subscriptions": []})
        return sns_xml(action, f"<TopicArn>{arn}</TopicArn>")
    if action == "Subscribe":
        arn = form["TopicArn"]
        sub_arn = f"{arn}:{uuid.uuid4()}"
        topics[arn]["subscriptions"].append(
            {
                "Protocol": form["Protocol"],
                "Endpoint": form.get("Endpoint", ""),
                "SubscriptionArn": sub_arn,
            }
        )
        return sns_xml(action, f"<SubscriptionArn>{sub_arn}</SubscriptionArn>")
    if action == "Publish":
        deliver_sns(form["TopicArn"], form.get("Subject", ""), form.get("Message", ""))
        return sns_xml(action, f"<MessageId>{uuid.uuid4()}</MessageId>")
    if action == "ListTopics":
        members = "".join(
            f"<member><TopicArn>{arn}</TopicArn></member>" for arn in sorted(topics)
        )
        return sns_xml(action, f"<Topics>{members}</Topics>")
    if action == "DeleteTopic":
        topics.pop(form.get("TopicArn", ""), None)
        return sns_xml(action, "")
    return Response(
        '<ErrorResponse xmlns="http://sns.amazonaws.com/doc/2010-03-31/">'
        f"<Error><Type>Sender</Type><Code>InvalidAction</Code>"
        f"<Message>{action}</Message></Error></ErrorResponse>",
        status_code=400,
        media_type="text/xml",
    )


async def dispatch(request: Request) -> Response:
    """Route by protocol: X-Amz-Target for EventBridge, Action form for SNS."""
    target = request.headers.get("x-amz-target", "")
    if target.startswith("AWSEvents."):
        body = json.loads(await request.body() or b"{}")
        return await events_api(request, target.rpartition(".")[2], body)
    form = dict(urllib.parse.parse_qsl((await request.body()).decode()))
    return sns_api(form)


app = Starlette(routes=[Route("/", dispatch, methods=["POST"])])

if __name__ == "__main__":
    threading.Thread(target=scheduler, daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
