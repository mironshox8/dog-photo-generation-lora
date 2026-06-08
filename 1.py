from diffusers import StableDiffusionPipeline
import torch

base_model = "runwayml/stable-diffusion-v1-5"

pipe = StableDiffusionPipeline.from_pretrained(
    base_model,
    torch_dtype=torch.float16,
    safety_checker=None
)

# PEFT LoRA yuklash
pipe.load_lora_weights("D:\dog\lora_model")

pipe = pipe.to("cuda")

prompt = """
dog123, a black rottweiler dog playing tug of war with a woman wearing a red shirt, both pulling opposite ends of a rope toy, outdoor park, grassy field, surrounded by trees, happy interaction, action shot, dynamic movement, natural daylight, full body dog, realistic scene, detailed fur, sharp focus, candid photography, ultra detailed, professional photography, high quality, 8k"""

image = pipe(
    prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

image.save("result.png")
print("Rasm yaratildi")