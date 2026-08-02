# IAM for chapter 1

**Deploy identity.** This chapter deploys with the `ch01-user` role, not admin —
its permissions are in `deploy.json`. Bootstrap it once and assume it as described
in [`code/README.md`](../../../README.md) (`make iam` in aws/).

**Service roles.** The Glue job needs its own role (Glue assumes it at run time),
templated here with `<ACCOUNT_ID>`/`<REGION>`:

| File | Applied as |
|---|---|
| ch01-glue-trust.json | trust policy of role `ch01-glue` (+ managed AWSGlueServiceRole) |
| ch01-glue-s3.json | inline policy `s3-bureau-raw` on role `ch01-glue` |
| sagemaker-feature-store.json | reference policy for the SageMaker Feature Store calls; these actions are already in `deploy.json`, so `ch01-user` can make them directly |

```
aws iam create-role --role-name ch01-glue --assume-role-policy-document file://ch01-glue-trust.json
aws iam attach-role-policy --role-name ch01-glue --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
aws iam put-role-policy --role-name ch01-glue --policy-name s3-bureau-raw --policy-document file://ch01-glue-s3.json
```
