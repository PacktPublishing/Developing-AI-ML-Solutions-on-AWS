# Chapter 7 — Serverless Vision RAG for KYC (design spec)

Internal build spec, not chapter prose. Strip or rewrite before the chapter's
final promotion. All code/library claims here are **untested** and must be
author-verified before shipping (fair-use policy).

## The story
Face verification for onboarding, three serverless pieces over one facenet
embedder and a pgvector store:

- **compare (1:1):** given two photos, return the cosine similarity. The core KYC
  check — is this the same person? A genuine applicant photographed twice (or
  tilting/moving their face to try to fake the check) still scores high, because
  MTCNN re-aligns the pose and the embedding is robust to it. Optional Captum
  Integrated Gradients explanation shows which regions drive the match.
- **enrol:** an S3 upload under `enrolled/{subject}/...` triggers a Lambda that
  embeds the face into one `faces` table (pgvector).
- **match (1:N):** embed a probe and KNN-search the enrolled corpus — the embedding
  DB *is* the retrieval corpus (the "vision RAG" framing).

Numbering caveat: the preface docx has this as **Chapter 7**; an older draft had
it as Chapter 8. Confirm against the latest outline.

## Locked decisions

### 1. Embedder — replace the private-weights hybrid
- OLD (reference impl): InsightFace ONNX + ArcFace R50, weights from a **private S3
  bucket** baked at build, `arcface_torch` on `sys.path`. Not reproducible;
  `insightface` weights are non-commercial.
- NEW: **`facenet-pytorch` — `InceptionResnetV1(pretrained='vggface2')`** + its
  MTCNN detector. MIT code, weights auto-download (no private bucket), 512-dim,
  L2-normalized (cosine = dot), CPU-viable in a container Lambda. VERIFIED loading
  + inference on Python 3.12 CPU.
- ⚠️ AUTHOR-VERIFY: VGGFace2-derived weights acceptable to redistribute (Packt).

### 2. Explainer — Captum Integrated Gradients
`src/face_explainer.py`: embed two faces, attribute the cosine similarity back to
each image's pixels (bidirectional IG), and render a 2-panel `torchvision
make_grid` with a purple saliency overlay per face. Used by `local/compare.py` and
the compare Lambda's opt-in `{"explain": true}` path — just "how similar, and why".

### 3. Data — real KYC faces are confidential
`data/fetch_faces.py` pulls a small sample of **SFHQ-T2I** synthetic faces (Flux/
SDXL/DALL-E 3, no real people) from Kaggle for the enrol gallery and the compare
demos. The same-person "face-move" example is just a rotation of one photo (the
fraud attempt), which the embedding still matches.
- ⚠️ AUTHOR-VERIFY: SFHQ-T2I license before bundling any images (fetch to a
  gitignored folder; commit none).

### 4. HNSW index-first (streaming ingestion)
Create the HNSW index **with the `faces` table, while empty**, so every async enrol
write maintains the graph incrementally rather than paying for a rebuild over a
full table later. (Behaviour of the enrol handler's bootstrap.)
- ⚠️ Benchmark: incremental HNSW insert cost vs. one bulk build; for a one-time
  backfill of millions, drop + rebuild once, then switch to index-first.

### 5. Vector store + infra
- **small RDS Postgres + pgvector** (same as ch-03 — Aurora Serverless v2 has no
  clean CFN "express" path). Local: a Postgres+pgvector container.
- Three container Lambdas (enrol / match / compare), one shared image, behind
  **API Gateway**. Image built by **CodeBuild → ECR** (native amd64, no laptop
  cross-build); the app stack references the pushed `ImageUri`.
- DEPLOY-TESTED 2026-08-01 end to end on 823613469927 (CodeBuild → ECR → Lambda →
  RDS pgvector), then torn down. Caught + fixed: baked torch weights needed
  `chmod a+rX` for the non-root Lambda user; concurrent `CREATE EXTENSION` race on
  cold-start bootstrap (async retry recovers — worth serialising for the book).

## Open author-verification flags
- [ ] VGGFace2 weight provenance acceptable (Packt).
- [ ] SFHQ-T2I license confirms redistribution.
- [ ] Benchmark incremental-HNSW insert vs. bulk build.

## Code layout (`code/ch-07-serverless-vision-rag-kyc/`)
```
src/            face_embedder.py (facenet), face_explainer.py (captum)
data/           fetch_faces.py (SFHQ sample)
local/          compare.py (two photos + captum grid), smoke_embed.py
aws/vision-rag/ build.yaml (ECR + CodeBuild), template.yaml (RDS + 3 Lambdas + API GW),
                src/ (Dockerfile + buildspec + handlers)
diagrams/       yaml + png (fig_serverless_kyc, fig_local)
```
