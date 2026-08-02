# Cloud deployment

`template.yaml` deploys the assistant: Redshift Serverless, the ECS Fargate
service (desired count 0 at rest), the scoped task role, ECR, and logs. The
image is a Docker build pushed to the stack's repository; the service scales
to 1 for a session and back to 0 after.

The permissions the deploying principal needs are collected in
`iam/deploy.json`: the stack (CloudFormation, ECS, ECR, CodeBuild, Redshift
Serverless, the roles it creates), plus the seed-and-provision round through
the Data API. Scope resources to your account before production use.

## The identity ladder

Authentication comes in layers, and the template ships the first; the chapter
builds the second; the third is the fleet-scale posture.

1. **Network (shipped)**: the security group admits one address
   (`AllowedCidr`). Good enough for a demo; it authenticates a network, not a
   person, and the `?arg=` username the terminal forwards is client-side
   input, not identity.
2. **User (the chapter's auth section)**: an Application Load Balancer in
   front of the service with an `authenticate-oidc` action, wired to any
   OIDC provider (Amazon Cognito or the corporate IdP). The browser signs in
   through SSO before a byte reaches ttyd, and the ALB forwards the identity
   as a signed `x-amzn-oidc-data` JWT. The launcher then derives
   `REDSHIFT_USER_NAME` from the token's claim instead of trusting the URL:
   the analyst's warehouse identity follows their corporate identity, and the
   useractivitylog attributes every statement to a person.
3. **Fleet (prose, not deployed)**: when many analysts run Claude Code, the
   Claude Apps Gateway pattern replaces direct task-role Bedrock access with
   a self-hosted OIDC control plane: `claude /login` opens corporate SSO, the
   gateway issues short-lived tokens, refreshes them silently, and central
   policy covers cost, rate, and revocation. Removing a user from the IdP
   ends their access within the token lifetime. Same OIDC provider as layer
   2, one identity story end to end. Shops that run this workload on EKS
   instead of Fargate typically deliver it GitOps-style, an Argo CD
   application synced from the repository, and the same OIDC provider signs
   analysts into Argo CD too.

## Monitoring

Three audit surfaces, one per layer of the stack: the helper's JSON audit
lines (stdout, CloudWatch via the awslogs driver), the namespace's
useractivitylog export (every statement, as which user, at the engine), and
Bedrock model invocation logging (account-level; remember the chapter 3
warning that those logs carry pre-guardrail content).
