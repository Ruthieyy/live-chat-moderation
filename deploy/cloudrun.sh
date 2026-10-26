#!/usr/bin/env bash
# Deploy the sync moderation API to Cloud Run.
# The async worker is a second service that needs Redis (Memorystore); this
# script ships the latency-critical path, which does not require Redis.
set -euo pipefail

: "${PROJECT:?set PROJECT to the GCP project id}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-live-chat-moderation}"
REPO="${REPO:-moderation}"

SHORT_SHA="$(git rev-parse --short HEAD)"
IMAGE="${REGION}-docker.pkg.dev/${PROJECT}/${REPO}/${SERVICE}:${SHORT_SHA}"

if ! gcloud artifacts repositories describe "$REPO" --location="$REGION" >/dev/null 2>&1; then
  gcloud artifacts repositories create "$REPO" \
    --repository-format=docker \
    --location="$REGION" \
    --description="Live chat moderation images"
fi

gcloud auth configure-docker "${REGION}-docker.pkg.dev" --quiet
gcloud builds submit --tag "$IMAGE" .

# min-instances=1 and --no-cpu-throttling keep a warm process with CPU between
# requests. Cold starts and CPU throttling are what blow a 100ms p99 budget.
# concurrency is capped at 16 so one vCPU is not buried under the threadpool.
gcloud run deploy "$SERVICE" \
  --image "$IMAGE" \
  --region "$REGION" \
  --platform managed \
  --port 8080 \
  --cpu 1 \
  --memory 512Mi \
  --min-instances 1 \
  --max-instances 5 \
  --concurrency 16 \
  --no-cpu-throttling \
  --set-env-vars "MODERATION_MODEL_DIR=/app/models,MODERATION_REDIS_ENABLED=false,MODERATION_LOG_LEVEL=info,MODERATION_ORT_INTRA_OP_THREADS=1" \
  --allow-unauthenticated

gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
