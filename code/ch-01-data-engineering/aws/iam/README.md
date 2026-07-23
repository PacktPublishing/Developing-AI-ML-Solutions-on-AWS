# IAM applied for the chapter's cloud verification

Templated with <ACCOUNT_ID>/<REGION>; applied to account via the commands below.

| File | Applied as |
|---|---|
| glue-book-ch01-trust.json | trust policy of role `glue-book-ch01` (+ managed AWSGlueServiceRole) |
| glue-book-ch01-s3.json | inline policy `s3-bureau-raw` on role `glue-book-ch01` |
| sagemaker-feature-store.json | managed policy `sagemaker-feature-store-ch01`, attached to group `book-ch01` (user `admin` is a member; user policy quota was full) |

```
aws iam create-role --role-name glue-book-ch01 --assume-role-policy-document file://glue-book-ch01-trust.json
aws iam attach-role-policy --role-name glue-book-ch01 --policy-arn arn:aws:iam::aws:policy/service-role/AWSGlueServiceRole
aws iam put-role-policy --role-name glue-book-ch01 --policy-name s3-bureau-raw --policy-document file://glue-book-ch01-s3.json
aws iam create-policy --policy-name sagemaker-feature-store-ch01 --policy-document file://sagemaker-feature-store.json
aws iam create-group --group-name book-ch01
aws iam attach-group-policy --group-name book-ch01 --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/sagemaker-feature-store-ch01
aws iam add-user-to-group --group-name book-ch01 --user-name <YOUR_USER>
```
