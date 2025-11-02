`mermaid
flowchart TD
    A[Raw Images] -->|Quality Check| B[Processed Dataset]
    B --> C[Caption Generation]
    C --> D[Manifest + Splits]
    D --> E[LoRA Trainer]
    E -->|Checkpoints| F[LoRA Artifacts]
    F --> G[Evaluator]
    G -->|Metrics| H[API Layer]
    H -->|REST| I[Client]
`

- prepare-data handles image validation, optional hard linking, caption files, and manifest output.
- 	rain launches the LoRA fine-tuning loop with Accelerate-driven checkpointing.
- evaluate produces FID / CLIP scores and supports A/B comparison.
- FastAPI wraps the pipeline for remote orchestration and monitoring.
