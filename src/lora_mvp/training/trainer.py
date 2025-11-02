from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from accelerate import Accelerator
from diffusers import DDPMScheduler, StableDiffusionPipeline
from peft import LoraConfig
from torch.nn import Module
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    plt = None

from ..config import TrainingConfig
from ..logging import get_logger
from ..utils.image_io import load_image
from .utils import create_scheduler, rotate_checkpoints, should_save


log = get_logger(__name__)


class SimpleImageDataset(Dataset):
    def __init__(self, manifest: list[dict], transform: transforms.Compose):
        self.manifest = manifest
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int):
        entry = self.manifest[index]
        image = load_image(Path(entry["processed"]))
        tensor = self.transform(image)
        caption_path = Path(entry["processed"]).with_suffix(".txt")
        caption = caption_path.read_text(encoding="utf-8")
        return {"pixel_values": tensor, "captions": caption}


@dataclass
class LoRATrainingPipeline:
    config: TrainingConfig

    def _enable_lora(self, unet) -> list[nn.Parameter]:
        unet.requires_grad_(False)
        adapter_name = "default"

        if adapter_name not in getattr(unet, "peft_config", {}):
            lora_config = LoraConfig(
                r=self.config.lora_rank,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                init_lora_weights="gaussian",
            )
            unet.add_adapter(lora_config, adapter_name=adapter_name)

        trainable_params = []
        for param in unet.parameters():
            if param.requires_grad:
                # keep LoRA params in float32 for stable training with gradient scaling
                if param.data.dtype != torch.float32:
                    param.data = param.data.to(dtype=torch.float32)
                trainable_params.append(param)
        return trainable_params

    def run(
        self,
        train_manifest: list[dict],
        val_manifest: list[dict],  # noqa: ARG002 - reserved for future use
        transform: transforms.Compose,
    ) -> None:
        accelerator = Accelerator(mixed_precision=self.config.mixed_precision)
        accelerator.print("Starting LoRA fine-tuning")

        train_dataset = SimpleImageDataset(train_manifest, transform)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )

        noise_scheduler = DDPMScheduler.from_pretrained(
            self.config.pretrained_model,
            subfolder="scheduler",
        )
        torch_dtype = (
            torch.float16
            if self.config.mixed_precision == "fp16"
            else torch.bfloat16
            if self.config.mixed_precision == "bf16"
            else torch.float32
        )
        device = accelerator.device
        pipeline = StableDiffusionPipeline.from_pretrained(
            self.config.pretrained_model,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        pipeline = pipeline.to(device)
        pipeline.unet.requires_grad_(False)
        pipeline.text_encoder.requires_grad_(False)

        lora_params = self._enable_lora(pipeline.unet)

        optimizer = AdamW(lora_params, lr=self.config.learning_rate)

        total_batch_size = self.config.batch_size * accelerator.num_processes * self.config.gradient_accumulation
        accelerator.print(f"Total batch size: {total_batch_size}")

        max_train_steps = self.config.max_train_steps
        lr_scheduler = create_scheduler(self.config, optimizer, max_train_steps)

        (
            pipeline.unet,
            optimizer,
            train_dataloader,
            lr_scheduler,
        ) = accelerator.prepare(
            pipeline.unet,
            optimizer,
            train_dataloader,
            lr_scheduler,
        )

        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        global_step = 0
        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
        )
        pipeline.unet.train()
        loss_history: list[tuple[int, float]] = []

        epoch = 0
        while global_step < max_train_steps:
            for batch in train_dataloader:
                with accelerator.accumulate(pipeline.unet):
                    pixel_values = batch["pixel_values"].to(device=device, dtype=pipeline.vae.dtype)
                    latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
                    latents = latents * pipeline.vae.config.scaling_factor
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(
                        0,
                        noise_scheduler.config.num_train_timesteps,
                        (latents.shape[0],),
                        device=device,
                        dtype=torch.long,
                    )
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    text_inputs = pipeline.tokenizer(
                        batch["captions"],
                        padding="max_length",
                        truncation=True,
                        max_length=pipeline.tokenizer.model_max_length,
                        return_tensors="pt",
                    )
                    encoder_hidden_states = pipeline.text_encoder(
                        text_inputs.input_ids.to(device)
                    ).last_hidden_state

                    model_pred = pipeline.unet(
                        sample=noisy_latents,
                        timestep=timesteps,
                        encoder_hidden_states=encoder_hidden_states,
                    ).sample
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(lora_params, 1.0)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    loss_value = loss.item()
                    accelerator.log({"train_loss": loss_value, "step": global_step})
                    if accelerator.is_local_main_process:
                        progress_bar.set_postfix(loss=f"{loss_value:.4f}")
                    if accelerator.is_main_process:
                        accelerator.print(f"Step {global_step}: loss={loss_value:.4f}")
                        loss_history.append((global_step, loss_value))
                    if should_save(global_step, self.config):
                        self._save_checkpoint(accelerator, pipeline.unet, global_step)

                if global_step >= max_train_steps:
                    break

            epoch += 1
            if global_step >= max_train_steps:
                break
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if loss_history:
                loss_file = self.config.output_dir / "loss_history.csv"
                loss_file.parent.mkdir(parents=True, exist_ok=True)
                with loss_file.open("w", encoding="utf-8") as f:
                    f.write("step,loss\n")
                    for step, value in loss_history:
                        f.write(f"{step},{value}\n")
                log.info("Saved loss history to %s", loss_file)
                if plt is not None:
                    steps, losses = zip(*loss_history)
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.plot(steps, losses, label="Training Loss")
                    ax.set_xlabel("Step")
                    ax.set_ylabel("Loss")
                    ax.set_title("LoRA Training Loss")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="upper right")
                    fig.tight_layout()
                    loss_plot = self.config.output_dir / "loss_curve.png"
                    fig.savefig(loss_plot, dpi=150)
                    plt.close(fig)
                    log.info("Saved loss curve plot to %s", loss_plot)
                else:
                    log.warning("matplotlib not available; skipping loss curve plotting.")

            unwrapped_unet = accelerator.unwrap_model(pipeline.unet)

            # 保存独立的 LoRA 适配器，便于后续加载到底模上使用
            lora_dir = self.config.output_dir / "lora_adapter"
            lora_dir.mkdir(parents=True, exist_ok=True)
            unwrapped_unet.save_attn_procs(lora_dir)
            log.info("Saved LoRA adapter weights to %s", lora_dir)

            pipeline.unet = unwrapped_unet.to("cpu")
            pipeline.vae.to("cpu")
            if hasattr(pipeline, "text_encoder"):
                pipeline.text_encoder.to("cpu")
            pipeline.save_pretrained(self.config.output_dir)
            log.info("Saved final LoRA pipeline to %s", self.config.output_dir)

    def _save_checkpoint(self, accelerator: Accelerator, model: Module, step: int) -> None:
        checkpoint_dir = self.config.output_dir / f"checkpoint-{step}"
        accelerator.save_state(checkpoint_dir)
        rotate_checkpoints(self.config.output_dir, self.config.max_checkpoints)
        accelerator.print(f"Saved checkpoint to {checkpoint_dir}")
