# Grafana with the fraud board baked in: the same provisioning and dashboard JSON as
# the local stack, so the Fargate service comes up already wired to the warehouse.
# The datasource reads WAREHOUSE_* from the environment; the task sets them from the
# Redshift Serverless workgroup.
FROM grafana/grafana:11.2.0

COPY provisioning /etc/grafana/provisioning
COPY dashboards /var/lib/grafana/dashboards
