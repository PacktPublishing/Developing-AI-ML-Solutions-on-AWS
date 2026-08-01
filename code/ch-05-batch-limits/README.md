# Batch scoring and credit-limit management

Decisions at portfolio scale: once a month a batch run re-scores the card
book, shortlists who is eligible for a limit change, and applies the
decisions. The steps hand artifacts to each other through S3 —
`eligible.csv` → `scores.csv` → `decisions.csv` — which is the shape a
SageMaker Pipeline formalizes.

## Run it

```
make up      # S3 and the portfolio warehouse
make seed    # synthesize the portfolio, load the warehouse
make run     # shortlist -> score -> decide -> apply
make query   # average limit after the run
make down    # stop and clean
```

## Local vs AWS

| In this directory | On AWS |
| --- | --- |
| the four step scripts, run in order | the same four steps as a SageMaker Pipeline |
| `make pipeline`: local executor, real S3 | `PIPELINE_MODE=aws`: SageMaker jobs |
| `steps/decide.py` | the limit-manager Lambda |
| S3Proxy | Amazon S3 |
| redshift-local | Amazon Redshift Serverless |
| the events-local shim | EventBridge and SNS |
| `make schedule`: rule + topic via the API | the same resources in aws/template.yaml |

The pipeline (`pipeline/`) uses ch-02's local pipeline session: the same
DAG runs on the local executor and as SageMaker jobs, with only the session
changing. `make pipeline` needs real credentials, a real S3 bucket
(`BATCH_BUCKET`), and a SageMaker role — local compute, real control plane.

The schedule runs locally too: `make events` serves an EventBridge- and
SNS-shaped service (rules fire on their cron or rate expression; SNS
targets deliver, other targets are printed with their event), and
`make schedule` creates the rule and the alerts topic with the same
configuration the template declares. A failed local pipeline run publishes
to the alerts topic when SNS_ENDPOINT is set.
