from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm
import os
import gc

from prompts import prompt_list

# SDXL code: https://github.com/huggingface/diffusers/pull/3859

model_dir = os.getenv("SDXL_MODEL_DIR")

if model_dir:
    # Use local model
    model_key_base = os.path.join(model_dir, "stable-diffusion-xl-base-1.0")
    model_key_refiner = os.path.join(model_dir, "stable-diffusion-xl-refiner-1.0")
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-1.0"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-1.0"

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")

pipe.enable_model_cpu_offload()
# pipe.to("cuda")

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

print("Loading model", model_key_refiner)
pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe_refiner.enable_model_cpu_offload()
# pipe_refiner.to("cuda")

# pipe_refiner.unet = torch.compile(pipe_refiner.unet, mode="reduce-overhead", fullgraph=True)

BASE_PATH = "imgs_sdxl"

os.makedirs(BASE_PATH, exist_ok=True)

def prompt_to_path(prompt, seed, i):
    return f"{BASE_PATH}/{seed}_{prompt.replace(' ', '_')}_{i}.jpg"

def generate(prompt, negative, scale, samples=4, steps=50, refiner_strength=0.3, seed=0):
    generator = torch.Generator("cuda").manual_seed(seed)
    prompt_list, negative_list = [prompt] * samples, [negative] * samples
    
    images = pipe(prompt=prompt_list, negative_prompt=negative_list, guidance_scale=scale, num_inference_steps=steps, generator=generator).images

    gc.collect()
    torch.cuda.empty_cache()

    images = pipe_refiner(prompt=prompt_list, negative_prompt=negative_list, image=images, num_inference_steps=steps, strength=refiner_strength).images

    gc.collect()
    torch.cuda.empty_cache()

    for i, image in enumerate(images):
        image.save(prompt_to_path(prompt, seed, i), format="JPEG")

neg_prompt = "low-quality, artifacts, blurry, watermark"

if __name__ == "__main__":
    for prompt_id, prompt in enumerate(tqdm(prompt_list)):
        generate(prompt, neg_prompt, scale=9.0, samples=4, steps=50, refiner_strength=0.3, seed=prompt_id)
