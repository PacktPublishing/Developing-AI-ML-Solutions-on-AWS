# ch-09 on AWS: underwriting knowledge base

The cloud round provisions one managed store — an Amazon OpenSearch Service
domain — and uses **OpenSearch Dashboards** as the search UI. The same
`opensearch_store` code indexes and queries the domain, and only the endpoint
and the auth change.

## Run it

```
make iam                              # create ch09-user from iam/deploy.json (privileged identity)
# add a [profile ch09] that assumes ch09-user, then run the rest as AWS_PROFILE=ch09
make deploy                           # secret + domain (fine-grained access control; ~15-20 min)
make gen                              # synthetic corpus (skip if data/generated/memos exists)
make seed                             # embed via Bedrock, index into the domain
make url                              # print the Dashboards URL and master user
make ask  Q="How is DTI assessed for a grocery business?"
make teardown                         # delete the domain and the master secret
```

Sign in to Dashboards as `kbadmin` (password in the `ch09-os-master` secret) and
search the `memo_chunks` index; `make ask` / `make cases` run the same grounded
retrieval from the command line against the domain.

## Cost posture

A single `t3.small.search` node on gp3 storage, deleted at teardown. The domain
bills per hour while it exists, so this is a deploy-seed-search-teardown round,
not a resource left running. Bedrock bills per token for the embeddings.

## Notes

- Fine-grained access control requires node-to-node and at-rest encryption and
  enforced HTTPS; the template sets all three.
- The endpoint is public but its access policy opens only to your IP, and every
  request still authenticates against the master user.
- Embeddings use Amazon Titan Text Embeddings v2; enable that model in Bedrock
  model access first if it is not already on.
