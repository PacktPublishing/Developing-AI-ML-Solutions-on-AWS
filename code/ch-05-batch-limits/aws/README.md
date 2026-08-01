# The batch limit run on AWS

The stack carries the pipeline's bucket, the step image built in the cloud,
the role the pipeline runs as, the EventBridge schedule that starts it, and
an SNS topic that hears about failures. The pipeline definition is the same
`pipeline/pipeline.py` that ran locally, registered with
`PIPELINE_MODE=aws`. The warehouse is Redshift Serverless — deploy the
chapter-01 stack (`code/ch-01-data-engineering/aws/redshift-serverless.yaml`)
and pass its endpoint as `WAREHOUSE_DSN`.

## Before the first deploy

```
aws iam create-group --group-name book-ch05
aws iam put-group-policy --group-name book-ch05 \
  --policy-name CreditBookCh5Deploy --policy-document file://iam/deploy.json
aws iam add-user-to-group --group-name book-ch05 --user-name <you>
```

## Run it, in order

```
make deploy      # bucket, ECR, CodeBuild, roles, schedule, alerts
make image       # build the step image in the cloud
make pipeline WAREHOUSE_DSN=postgresql://...   # register the pipeline
make run-now     # one execution now; the schedule owns the rest
make teardown    # remove the pipeline and the stack
```

Deploy with an email to hear about failures:

```
sam deploy --parameter-overrides AlertEmail=you@example.com
```
