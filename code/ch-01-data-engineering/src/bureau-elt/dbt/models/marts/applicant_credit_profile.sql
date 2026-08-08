-- the gold table: one row per applicant, the shape a model team consumes
with latest_report as (
    select
        applicant_id,
        report_id,
        score,
        state,
        row_number() over (
            partition by applicant_id order by generated_at desc
        ) as rn
    from {{ ref('stg_reports') }}
),

-- aggregate per report, so the totals belong to a single bureau snapshot;
-- grouping by applicant would sum every day's report for that applicant
tradeline_summary as (
    select
        report_id,
        count(*) as tradeline_count,
        sum(balance) as total_balance,
        sum(case when delinquent then 1 else 0 end) as delinquent_count
    from {{ ref('stg_tradelines') }}
    group by report_id
)

select
    r.applicant_id,
    r.score,
    r.state,
    coalesce(t.tradeline_count, 0) as tradeline_count,
    coalesce(t.total_balance, 0) as total_balance,
    coalesce(t.delinquent_count, 0) as delinquent_count
from latest_report r
left join tradeline_summary t on r.report_id = t.report_id
where r.rn = 1
