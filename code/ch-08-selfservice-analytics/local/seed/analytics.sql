-- The credit mart the assistant queries: the same shape the data engineering
-- chapter's gold mart produces, seeded small so questions have real answers.
CREATE SCHEMA IF NOT EXISTS analytics;

CREATE TABLE analytics.applicant_credit_profile (
    applicant_id     BIGINT,
    score            INT,
    state            VARCHAR(2),
    tradeline_count  INT,
    total_balance    BIGINT,
    delinquent_count INT
);

INSERT INTO analytics.applicant_credit_profile VALUES
    (10036, 850, 'CA', 4,  63563, 0),
    (10073, 850, 'NY', 7, 172317, 0),
    (10042, 712, 'CA', 5,  88210, 1),
    (10058, 640, 'TX', 3,  40110, 2),
    (10091, 588, 'NY', 6, 129400, 3),
    (10014, 705, 'FL', 2,  20050, 0),
    (10027, 771, 'TX', 4,  56800, 0),
    (10065, 623, 'FL', 5,  97320, 2);

CREATE TABLE analytics.applicants (
    applicant_id BIGINT,
    full_name    VARCHAR(120),
    email        VARCHAR(120),
    phone        VARCHAR(32)
);

INSERT INTO analytics.applicants VALUES
    (10036, 'Ana Demo',  'ana@example.com',  '+15550100036'),
    (10073, 'Ben Demo',  'ben@example.com',  '+15550100073'),
    (10042, 'Cara Demo', 'cara@example.com', '+15550100042');
