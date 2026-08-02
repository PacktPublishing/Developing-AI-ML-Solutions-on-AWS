---
name: redshift-data-api
description: Run read-only SQL against the warehouse through the Redshift Data API helper, with masking, timeouts, and row caps applied. Use for every data question.
---

# Redshift Data API

Run queries through `helper.py`, never through a direct database connection.
The default way is the CLI, one command, no Python of your own:

```bash
uv run skills/redshift-data-api/helper.py \
  "SELECT state, AVG(score) AS avg_score
   FROM analytics.applicant_credit_profile
   GROUP BY state ORDER BY avg_score DESC LIMIT 10"
```

Each result row prints as one JSON object; read the values and present them
as a markdown table. Only import `execute_redshift_query` in Python when you
genuinely need computation over the rows, and remember rows are dicts keyed
by column name, never tuples.
- The helper enforces a query timeout and a result row cap, and masks PII
  columns listed in `config/pii_columns.yaml` before results reach you.
- The schema catalog lives in `knowledge/catalog.md`; check it before
  guessing table or column names.
