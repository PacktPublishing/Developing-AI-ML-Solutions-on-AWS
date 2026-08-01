#!/usr/bin/env bash
# Create one assumable IAM role per chapter -- ch01-user .. ch05-user -- each
# carrying only that chapter's aws/iam/deploy.json. You assume the chapter's
# role to deploy it, instead of running everything as admin (which piles every
# chapter's permissions onto one identity and hits the IAM per-user quota).
#
# Run ONCE, with a privileged identity that can create roles:
#
#   code/setup-users.sh create arn:aws:iam::<ACCOUNT_ID>:user/<your-base-user>
#   code/setup-users.sh delete
#
# The ARN you pass is the principal allowed to sts:AssumeRole each chapter role.
# Then wire up a per-chapter profile (see code/README.md) and deploy with it.
set -euo pipefail

ACTION="${1:-}"
HERE="$(cd "$(dirname "$0")" && pwd)" # code/

# chapter dir : role name
CHAPTERS=(
  "ch-01-data-engineering:ch01-user"
  "ch-02-mlops:ch02-user"
  "ch-03-generative-ai:ch03-user"
  "ch-04-realtime-scoring:ch04-user"
  "ch-05-batch-limits:ch05-user"
  "ch-06-streaming-fraud:ch06-user"
  "ch-07-serverless-vision-rag-kyc:ch07-user"
  "ch-08-self-service-analytics:ch08-user"
)

create() {
  local principal="${2:-}"
  [ -n "$principal" ] || {
    echo "usage: $0 create <trusted-principal-arn>" >&2
    exit 2
  }
  local trust
  trust=$(printf '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"AWS":"%s"},"Action":"sts:AssumeRole"}]}' "$principal")
  for entry in "${CHAPTERS[@]}"; do
    local dir="${entry%%:*}" role="${entry##*:}"
    local policy="$HERE/$dir/aws/iam/deploy.json"
    [ -f "$policy" ] || {
      echo "skip $role: no $policy" >&2
      continue
    }
    echo "== $role =="
    aws iam create-role --role-name "$role" \
      --assume-role-policy-document "$trust" \
      --description "Deploy/operate $dir (least-privilege)" >/dev/null 2>&1 ||
      aws iam update-assume-role-policy --role-name "$role" --policy-document "$trust"
    aws iam put-role-policy --role-name "$role" \
      --policy-name deploy --policy-document "file://$policy"
    echo "   role arn: $(aws iam get-role --role-name "$role" --query 'Role.Arn' --output text)"
  done
}

delete() {
  for entry in "${CHAPTERS[@]}"; do
    local role="${entry##*:}"
    echo "== deleting $role =="
    aws iam delete-role-policy --role-name "$role" --policy-name deploy 2>/dev/null || true
    aws iam delete-role --role-name "$role" 2>/dev/null || true
  done
}

case "$ACTION" in
create) create "$@" ;;
delete) delete ;;
*)
  echo "usage: $0 {create <trusted-principal-arn>|delete}" >&2
  exit 2
  ;;
esac
