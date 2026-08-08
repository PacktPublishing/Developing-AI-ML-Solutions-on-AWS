-- one row per bureau report, typed and renamed
select
    report_id,
    applicant_id,
    cast(generated_at as timestamp) as generated_at,
    score__value as score,
    score__model as score_model,
    address__state as state
from {{ source('bureau_raw', 'reports') }}
