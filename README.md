# sd-train-architecture · LoRA MVP

## Overview

This project is a lightweight training stack for adapting Stable Diffusion with LoRA. It ships with tools for curating datasets, fine‑tuning, scoring results, and serving jobs over an API, so you can go from raw images to deployable adapters in one repo.

## Features

- **Data pipeline** that validates images, auto-generates captions, and emits a manifest split into train/val
- **Accelerate-driven LoRA trainer** for Stable Diffusion, with gradient accumulation, checkpoint rotation, and persisted loss curves
- **Evaluation tooling** for FID/CLIP plus A/B comparison against a baseline model
- **Production-ready FastAPI** surface for submitting training runs and tracking their status
- **Dockerfile & Compose** setup with GPU support for easy packaging and deployment
- **Example inference script** showing how to fuse the trained LoRA back into a base model
- **Pytest suite** plus lint/format tooling configured via `pyproject.toml`

## Repository Layout

```
├── docs/                 # Architecture notes and REST API contract
├── examples/             # Inference helper and sample renders
├── scripts/              # Utility scripts (benchmarking, etc.)
├── src/lora_mvp/         # Library code: data, training, evaluation, API
├── tests/                # Pytest coverage for key workflows
├── docker/               # Container build + compose definitions
└── pyproject.toml        # Dependencies, dev extras, and tooling config
```

## Installation

### Local Environment

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .[dev]            # Editable install with dev extras
```

**Requirements:**
- Python ≥ 3.10
- GPU with CUDA support (or adjust configs for CPU-only experimentation)

### Docker

```bash
docker compose -f docker/compose.yaml up --build
```

This builds the API image, mounts local `data/`, `outputs/`, `state/`, and exposes `http://localhost:8000`.

## Workflow

### 1. Prepare Data

Place raw images under a class-labelled directory structure, e.g. `data/raw/<label>/*.png`. Then:

```bash
python -m lora_mvp.cli prepare-data --config config.yaml
```

If `--config` is omitted, defaults are derived from the current working directory. The command:

- Runs resolution/blur checks (OpenCV Laplacian variance)
- Writes captions using the configured template
- Creates a train/val split and emits `data/processed/manifest.json`

Use `--no-copy` to hard-link instead of copying when the filesystem supports it.

### 2. Train a LoRA Adapter

```bash
python -m lora_mvp.cli train \
  --manifest data/processed/manifest.json \
  --config config.yaml
```

**Outputs** (default `outputs/lora/`):
- `lora_adapter/` with LoRA weights (`*.bin/*.safetensors`)
- Rotating `checkpoint-*/` directories
- `loss_history.csv` and `loss_curve.png` for quick diagnostics
- Full pipeline save via `StableDiffusionPipeline.save_pretrained`

### 3. Evaluate

```bash
python -m lora_mvp.cli evaluate \
  --model-path outputs/lora \
  --real-glob "data/processed/val/**/*.png" \
  --config config.yaml
```

Generates samples using prompts from `EvaluationConfig`, then reports FID and CLIP scores. `Evaluator.ab_test` is also available if you want to compare two runs programmatically.

### 4. Inference Example

`examples/inference.py` illustrates how to load a base SD model, fuse the LoRA adapter (with fused-weight logging), and render prompts. Update the paths and `lora_scale` as needed.

### 5. Benchmark (Optional)

```bash
python scripts/benchmark.py --manifest data/processed/manifest.json --steps 50
```

Runs a short training loop to measure throughput.

## API Service

Serve the training/evaluation pipeline via FastAPI:

```bash
uvicorn lora_mvp.api.main:app --host 0.0.0.0 --port 8000
```

### Key Endpoints

- `POST /train` — submit a manifest; returns `job_id`
- `GET /status/{job_id}` — polling endpoint with timestamps and error messages
- `POST /evaluate` — compute FID/CLIP for a saved pipeline
- `GET /healthz` — readiness check

Docs live under `docs/API.md`; job metadata persists to `state/training_jobs.json` by default.

## Configuration

All knobs are defined in `lora_mvp.config`:

- **DataConfig**: directories, split ratio, image size, augmentation list, blur threshold
- **TrainingConfig**: base model, batch size, LoRA rank/alpha/dropout, learning rate schedule, checkpoint cadence
- **EvaluationConfig**: prompts, sample counts, CLIP/FID settings
- **ApiConfig**: worker pool size, polling interval, state DB path

Override them via a YAML file passed to CLI/API commands. Example snippet:

```yaml
data:
  raw_data_dir: data/raw
  processed_dir: data/processed

training:
  pretrained_model: runwayml/stable-diffusion-v1-5
  output_dir: outputs/lora
  max_train_steps: 1000
  learning_rate: 1e-4

evaluation:
  prompts:
    - "A photo of a cat"
    - "A futuristic cityscape at sunset"

api:
  max_workers: 2
  db_path: state/training_jobs.json
```

## License

[Add your license information here]

## Contributing

[Add contributing guidelines here]
