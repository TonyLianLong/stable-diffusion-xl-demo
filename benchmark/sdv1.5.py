from diffusers import DiffusionPipeline
import torch
from tqdm import tqdm
import os
import gc

from prompts import prompt_list

model_key_base = "runwayml/stable-diffusion-v1-5"

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_safetensors=True, variant="fp16")


pipe.safety_checker = lambda images, clip_input: (images, None)

# pipe.enable_model_cpu_offload()
pipe.to("cuda")

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

BASE_PATH = "imgs_sdv1.5"

os.makedirs(BASE_PATH, exist_ok=True)

def prompt_to_path(prompt, seed, i):
    return f"{BASE_PATH}/{seed}_{prompt.replace(' ', '_')}_{i}.jpg"

def generate(prompt, negative, scale, samples=4, steps=50, seed=0):
    generator = torch.Generator("cuda").manual_seed(seed)
    prompt_list, negative_list = [prompt] * samples, [negative] * samples
    
    images = pipe(prompt=prompt_list, negative_prompt=negative_list, guidance_scale=scale, num_inference_steps=steps, generator=generator).images

    gc.collect()
    torch.cuda.empty_cache()

    for i, image in enumerate(images):
        image.save(prompt_to_path(prompt, seed, i), format="JPEG")

neg_prompt = "low-quality, artifacts, blurry, smooth texture, bad quality, distortions, unrealistic, distorted image, bad proportions, watermark"

if __name__ == "__main__":
    for prompt_id, prompt in enumerate(tqdm(prompt_list)):
        generate(prompt, neg_prompt, scale=9.0, samples=4, steps=50, seed=prompt_id)
