# Warehouse catalog

The tables the assistant may query. Regenerate from information_schema when
the warehouse changes; do not let this file drift from reality.

## analytics.applicant_credit_profile

One row per applicant: the latest credit profile from the gold mart.

| column           | type    | notes                          |
|------------------|---------|--------------------------------|
| applicant_id     | bigint  | join key                       |
| score            | int     | bureau score, 300-850          |
| state            | varchar | two-letter US state            |
| tradeline_count  | int     |                                |
| total_balance    | bigint  | across open tradelines         |
| delinquent_count | int     | delinquent tradelines          |

## analytics.applicants

Contact details. email, phone, and full_name are PII and arrive masked.

| column       | type    | notes            |
|--------------|---------|------------------|
| applicant_id | bigint  | join key         |
| full_name    | varchar | masked: redact   |
| email        | varchar | masked: hash     |
| phone        | varchar | masked: last 4   |
