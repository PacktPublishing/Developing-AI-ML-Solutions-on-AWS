-- tradelines joined back to their parent report
select
    r.report_id,
    r.applicant_id,
    t.product,
    t.balance,
    t.months_on_book,
    t.delinquent
from {{ source('bureau_raw', 'reports__tradelines') }} t
join {{ source('bureau_raw', 'reports') }} r
    on t._dlt_parent_id = r._dlt_id
