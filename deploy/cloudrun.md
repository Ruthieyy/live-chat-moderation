# Cloud Run

The container listens on `$PORT` (Cloud Run sets this to 8080) and reads the ONNX model from `MODERATION_MODEL_DIR`. The sync API does not need Redis. The async worker is a separate process and expects Memorystore, or any Redis 6.2+, at `MODERATION_REDIS_URL`.

## What the deploy script sets

`deploy/cloudrun.sh` builds the image with Cloud Build, pushes it to Artifact Registry, and deploys one service:

- 1 vCPU, 512MiB, concurrency 16
- `--min-instances 1` so the process stays warm
- `--no-cpu-throttling` so the instance keeps CPU between requests
- `MODERATION_REDIS_ENABLED=false` because this service is the sync path
- `MODERATION_ORT_INTRA_OP_THREADS=1` because a single short sequence is slower with extra intra-op threads

```bash
PROJECT=your-project REGION=us-central1 ./deploy/cloudrun.sh
```

Unauthenticated invoke is on so a demo client can call it. A production gateway should put IAP or the platform IAM check in front and turn that off.

## Async path

Run the same image with `python -m moderation.worker`, point it at Memorystore, and set `MODERATION_REDIS_ENABLED=true` plus `MODERATION_REDIS_URL` on both the API service and the worker. Keep the worker's consumer name unique per instance (`MODERATION_WORKER_CONSUMER` overrides the default hostname-pid name). Redis Streams consumer groups then share the backlog.

## Cold start

The model is baked into the image (no download on boot) and the process warms the ONNX session before it reports ready. Min instances exist because a scaled-to-zero instance still pays a Python import plus session setup that will not make a 100ms p99 while it is starting. CPU throttling has the same effect on the first request after an idle period.

If `gcloud run deploy` rejects `--no-cpu-throttling`, the installed CLI is old: upgrade it, or set CPU allocation to "always allocated" in the console. That flag is the whole point of the latency configuration.
