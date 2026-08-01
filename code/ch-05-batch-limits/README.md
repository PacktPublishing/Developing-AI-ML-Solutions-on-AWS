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
| the four step scripts, run in order | a SageMaker Pipeline (steps in the chapter's image) |
| `steps/decide.py` | the limit-manager Lambda |
| S3Proxy | Amazon S3 |
| redshift-local | Amazon Redshift Serverless |
| `make run` by hand | an EventBridge schedule |

Still to come: the SageMaker Pipeline wrapper (ch-02's local pipeline
session), the EventBridge schedule, CloudWatch + SNS on the job, and the
`aws/` templates, with Redshift Serverless as in ch-01.
