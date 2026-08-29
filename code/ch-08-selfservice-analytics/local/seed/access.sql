-- The group config/access.yaml grants to. Local only: on AWS the accounts are
-- created by config/provision.sh (PASSWORD DISABLE, IAM credentials only), which
-- also puts them in this group. Here bi_analyst is the Compose superuser and
-- already exists, so it joins at seed time.
CREATE GROUP analysts;
ALTER GROUP analysts ADD USER bi_analyst;
