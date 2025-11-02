from pathlib import Path
import torch
from diffusers import StableDiffusionPipeline
from safetensors.torch import load_file

# 路径配置
base_model_dir = "/home/tione/notebook/lq/sd/zuoye/model"
lora_adapter_dir = "/home/tione/notebook/lq/sd/zuoye/outputs/lora6/lora_adapter"
lora_weight_file = Path(lora_adapter_dir) / "pytorch_lora_weights.safetensors"
output_dir = Path("/home/tione/notebook/lq/sd/zuoye/renders_lora11888")
output_dir.mkdir(parents=True, exist_ok=True)

# 设备与精度
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32
lora_scale = 0.7  # 按需调整 LoRA 强度

# 加载底模
pipe = StableDiffusionPipeline.from_pretrained(
    base_model_dir,
    torch_dtype=dtype,
    low_cpu_mem_usage=False,
).to(device)

# 读取 LoRA 文件并找出将被修改的权重
lora_state = load_file(lora_weight_file)
target_weight_names = sorted(
    {
        key.rsplit(".lora_", 1)[0] + ".weight"
        for key in lora_state.keys()
        if ".lora_" in key
    }
)

baseline: dict[str, torch.Tensor] = {}
with torch.no_grad():
    unet_state = pipe.unet.state_dict()

    # 保存融合前的权重，用于对比
    for name in target_weight_names:
        if name not in unet_state:
            print(f"[warning] {name} 未在 UNet 中找到，可能与底模版本不匹配。")
            continue
        weight_before = unet_state[name].detach().float().cpu().clone()
        baseline[name] = weight_before
        print(f"[融合前] {name}")
        print(f"    sample: {weight_before.view(-1)[:5].numpy()}")

    if not baseline:
        raise RuntimeError("LoRA 中的目标层未在 UNet 里找到，无法合并。")

    # 把 LoRA 增量合并到权重里
    for name in baseline:
        prefix = name[:-len(".weight")]
        base_param = unet_state[name]
        weight_device = base_param.device
        orig_dtype = base_param.dtype

        base_fp32 = base_param.detach().to(device=weight_device, dtype=torch.float32)

        down = lora_state[prefix + ".lora_A.weight"].to(device=weight_device, dtype=torch.float32)
        up = lora_state[prefix + ".lora_B.weight"].to(device=weight_device, dtype=torch.float32)

        rank = down.shape[0]
        alpha_tensor = lora_state.get(prefix + ".alpha")
        alpha = alpha_tensor.to(torch.float32).item() if alpha_tensor is not None else float(rank)
        scale = (alpha / rank) * lora_scale

        delta = torch.matmul(up, down)
        delta = delta.view_as(base_fp32) * scale

        base_fp32 += delta
        unet_state[name] = base_fp32.to(dtype=orig_dtype)

    pipe.unet.load_state_dict(unet_state)

# 打印融合后的对比
with torch.no_grad():
    merged_state = pipe.unet.state_dict()
    for name, before in baseline.items():
        after = merged_state[name].detach().float().cpu()
        diff = after - before
        print(f"\n{name}")
        print(f"    before: {before.view(-1)[:5].numpy()}")
        print(f"    after : {after.view(-1)[:5].numpy()}")
        print(f"    diff  : {diff.view(-1)[:5].numpy()}")
        print(f"    max abs diff: {diff.abs().max().item():.6f}")
        print(f"    mean abs diff: {diff.abs().mean().item():.6f}")

# 推理配置
pipe.set_progress_bar_config(disable=True)
pipe.enable_attention_slicing()

prompt = "a detailed portrait of an ancient-style warrior, elegant lighting, 4k"
prompts = [prompt] * 4
generator = torch.Generator(device=device).manual_seed(42)

images = pipe(
    prompt=prompts,
    num_inference_steps=40,
    guidance_scale=7.5,
    generator=generator,
).images

for idx, img in enumerate(images):
    img.save(output_dir / f"lora_{idx:02d}.png")

print(f"\nSaved {len(images)} images to {output_dir}")
