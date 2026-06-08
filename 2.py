import os
import gc
import torch

# ⚠️ CPU stabilizatsiya (MUHIM)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from diffusers import StableDiffusionPipeline

# ==========================================
# CONFIG
# ==========================================

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
LORA_PATH = r"D:\dog\lora_model"
OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# DEVICE (CPU ONLY)
# ==========================================

device = "cpu"

print("Device:", device)

# ==========================================
# MODEL LOAD (LOW RAM SAFE)
# ==========================================

print("Model yuklanmoqda...")

pipe = StableDiffusionPipeline.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float32,
    safety_checker=None,
    low_cpu_mem_usage=True,
    use_safetensors=False   # 🔥 CPU’da stabilroq
)

pipe = pipe.to(device)

# ==========================================
# MEMORY OPTIMIZATION
# ==========================================

pipe.enable_attention_slicing()
pipe.enable_vae_slicing()

# 🔥 CPU uchun eng muhim optimization
try:
    pipe.enable_sequential_cpu_offload()
except:
    print("CPU offload ishlamadi (normal)")

print("LoRA yuklanmoqda...")

# ==========================================
# LoRA LOAD
# ==========================================

pipe.load_lora_weights(LORA_PATH)

try:
    pipe.fuse_lora()
    print("LoRA fused")
except:
    print("LoRA fuse skip")

print("Model tayyor")

# ==========================================
# PROMPT
# ==========================================

prompt = """
A large light brown golden retriever running at full speed across a lush green grass field,
long flowing fur moving in the wind, joyful expression, dynamic action shot,
natural outdoor environment, vibrant green lawn, realistic dog anatomy,
highly detailed fur texture, professional wildlife photography,
shallow depth of field, cinematic lighting, ultra realistic,
sharp focus, 8K HDR, masterpiece, photorealistic, award-winning photography
"""

negative_prompt = """
blurry, low quality, low resolution, ugly, deformed, bad anatomy,
extra legs, extra limbs, missing legs, duplicated dog, distorted body,
cartoon, anime, painting, CGI, oversaturated colors, watermark,
text, logo, cropped, out of frame, motion artifacts, noise,
poor lighting, unrealistic fur, mutated, bad proportions
"""

# ==========================================
# GENERATION
# ==========================================

generator = torch.Generator(device="cpu").manual_seed(42)

print("Rasm yaratilmoqda...")

with torch.inference_mode():
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=384,
        width=384,
        num_inference_steps=30,   # 🔥 tezroq CPU uchun
        guidance_scale=6.0,
        generator=generator
    ).images[0]

# ==========================================
# SAVE
# ==========================================

save_path = os.path.join(OUTPUT_DIR, "result_lora.png")
image.save(save_path)

print("Saqlandi:", save_path)

# ==========================================
# CLEAN MEMORY
# ==========================================

del pipe
gc.collect()

print("Tugatildi")