# LoRA MVP API

## POST /train
Request body:
`json
{
  "dataset_manifest": "/app/data/processed/manifest.json"
}
`

Response:
`json
{
  "job_id": "<uuid>",
  "status": "queued"
}
`

Submit a manifest, then poll /status/{job_id} until the job finishes.

## GET /status/{job_id}
Returns job metadata including timestamps and error message if one occurred.

## POST /evaluate
Request body:
`json
{
  "model_path": "/app/outputs/lora",
  "real_samples_glob": "data/processed/val/**/*.png"
}
`

Response body contains id and clip metrics.

## GET /healthz
Simple health check endpoint suitable for readiness probes.
