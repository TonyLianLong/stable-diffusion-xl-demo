import gradio as gr

from diffusers import DiffusionPipeline
import torch

import base64
from io import BytesIO
import os
import gc

from share_btn import community_icon_html, loading_icon_html, share_js

# SDXL code: https://github.com/huggingface/diffusers/pull/3859

model_dir = os.getenv("SDXL_MODEL_DIR")
access_token = os.getenv("ACCESS_TOKEN")

if model_dir:
    # Use local model
    model_key_base = os.path.join(model_dir, "stable-diffusion-xl-base-0.9")
    model_key_refiner = os.path.join(model_dir, "stable-diffusion-xl-refiner-0.9")
else:
    model_key_base = "stabilityai/stable-diffusion-xl-base-0.9"
    model_key_refiner = "stabilityai/stable-diffusion-xl-refiner-0.9"

enable_refiner = True
# Output images before the refiner and after the refiner
output_images_before_refiner = False

print("Loading model", model_key_base)
pipe = DiffusionPipeline.from_pretrained(model_key_base, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", use_auth_token=access_token)

pipe.enable_model_cpu_offload()
# pipe.to("cuda")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

if enable_refiner:
    print("Loading model", model_key_refiner)
    pipe_refiner = DiffusionPipeline.from_pretrained(model_key_refiner, torch_dtype=torch.float16, use_safetensors=True, variant="fp16", use_auth_token=access_token)
    pipe_refiner.enable_model_cpu_offload()
    # pipe_refiner.to("cuda")

    # if using torch < 2.0
    # pipe_refiner.enable_xformers_memory_efficient_attention()

    # pipe_refiner.unet = torch.compile(pipe_refiner.unet, mode="reduce-overhead", fullgraph=True)

# NOTE: we do not have word list filtering in this gradio demo

is_gpu_busy = False
def infer(prompt, negative, scale, num_generation=4):
    prompt, negative = [prompt] * num_generation, [negative] * num_generation
    images = pipe(prompt=prompt, negative_prompt=negative, guidance_scale=scale).images

    images_b64_list = []

    if not enable_refiner and output_images_before_refiner:
        for image in images:
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            image_b64 = (f"data:image/jpeg;base64,{img_str}")
            images_b64_list.append(image_b64)

    gc.collect()
    torch.cuda.empty_cache()

    if enable_refiner:
        images = pipe_refiner(prompt=prompt, negative_prompt=negative, image=images).images

    for image in images:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        image_b64 = (f"data:image/jpeg;base64,{img_str}")
        images_b64_list.append(image_b64)
    
    return images_b64_list
    
    
css = """
        .gradio-container {
            font-family: 'IBM Plex Sans', sans-serif;
        }
        .gr-button {
            color: white;
            border-color: black;
            background: black;
        }
        input[type='range'] {
            accent-color: black;
        }
        .dark input[type='range'] {
            accent-color: #dfdfdf;
        }
        .container {
            max-width: 730px;
            margin: auto;
            padding-top: 1.5rem;
        }
        #gallery {
            min-height: 22rem;
            margin-bottom: 15px;
            margin-left: auto;
            margin-right: auto;
            border-bottom-right-radius: .5rem !important;
            border-bottom-left-radius: .5rem !important;
        }
        #gallery>div>.h-full {
            min-height: 20rem;
        }
        .details:hover {
            text-decoration: underline;
        }
        .gr-button {
            white-space: nowrap;
        }
        .gr-button:focus {
            border-color: rgb(147 197 253 / var(--tw-border-opacity));
            outline: none;
            box-shadow: var(--tw-ring-offset-shadow), var(--tw-ring-shadow), var(--tw-shadow, 0 0 #0000);
            --tw-border-opacity: 1;
            --tw-ring-offset-shadow: var(--tw-ring-inset) 0 0 0 var(--tw-ring-offset-width) var(--tw-ring-offset-color);
            --tw-ring-shadow: var(--tw-ring-inset) 0 0 0 calc(3px var(--tw-ring-offset-width)) var(--tw-ring-color);
            --tw-ring-color: rgb(191 219 254 / var(--tw-ring-opacity));
            --tw-ring-opacity: .5;
        }
        #advanced-btn {
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 12px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
        .footer {
            margin-bottom: 45px;
            margin-top: 35px;
            text-align: center;
            border-bottom: 1px solid #e5e5e5;
        }
        .footer>p {
            font-size: .8rem;
            display: inline-block;
            padding: 0 10px;
            transform: translateY(10px);
            background: white;
        }
        .dark .footer {
            border-color: #303030;
        }
        .dark .footer>p {
            background: #0b0f19;
        }
        .acknowledgments h4{
            margin: 1.25em 0 .25em 0;
            font-weight: bold;
            font-size: 115%;
        }
        .animate-spin {
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            from {
                transform: rotate(0deg);
            }
            to {
                transform: rotate(360deg);
            }
        }
        #share-btn-container {
            display: flex; padding-left: 0.5rem !important; padding-right: 0.5rem !important; background-color: #000000; justify-content: center; align-items: center; border-radius: 9999px !important; width: 13rem;
            margin-top: 10px;
            margin-left: auto;
        }
        #share-btn {
            all: initial; color: #ffffff;font-weight: 600; cursor:pointer; font-family: 'IBM Plex Sans', sans-serif; margin-left: 0.5rem !important; padding-top: 0.25rem !important; padding-bottom: 0.25rem !important;right:0;
        }
        #share-btn * {
            all: unset;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
        
        .gr-form{
            flex: 1 1 50%; border-top-right-radius: 0; border-bottom-right-radius: 0;
        }
        #prompt-container{
            gap: 0;
        }
        #prompt-text-input, #negative-prompt-text-input{padding: .45rem 0.625rem}
        #component-16{border-top-width: 1px!important;margin-top: 1em}
        .image_duplication{position: absolute; width: 100px; left: 50px}
"""

block = gr.Blocks(css=css)

examples = [
    [
        'A high tech solarpunk utopia in the Amazon rainforest',
        'low quality',
        9
    ],
    [
        'A pikachu fine dining with a view to the Eiffel Tower',
        'low quality',
        9
    ],
    [
        'A mecha robot in a favela in expressionist style',
        'low quality, 3d, photorealistic',
        9
    ],
    [
        'an insect robot preparing a delicious meal',
        'low quality, illustration',
        9
    ],
    [
        "A small cabin on top of a snowy mountain in the style of Disney, artstation",
        'low quality, ugly',
        9
    ],
]


with block:
    gr.HTML(
        """
            <div style="text-align: center; margin: 0 auto;">
              <div
                style="
                  display: inline-flex;
                  align-items: center;
                  gap: 0.8rem;
                  font-size: 1.75rem;
                "
              >
                <svg
                  width="0.65em"
                  height="0.65em"
                  viewBox="0 0 115 115"
                  fill="none"
                  xmlns="http://www.w3.org/2000/svg"
                >
                  <rect width="23" height="23" fill="white"></rect>
                  <rect y="69" width="23" height="23" fill="white"></rect>
                  <rect x="23" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="46" width="23" height="23" fill="white"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" width="23" height="23" fill="black"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="92" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="115" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="115" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="92" y="46" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="92" y="69" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="46" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="115" width="23" height="23" fill="white"></rect>
                  <rect x="69" y="69" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="46" y="46" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="115" width="23" height="23" fill="black"></rect>
                  <rect x="46" y="69" width="23" height="23" fill="black"></rect>
                  <rect x="23" y="46" width="23" height="23" fill="#D9D9D9"></rect>
                  <rect x="23" y="115" width="23" height="23" fill="#AEAEAE"></rect>
                  <rect x="23" y="69" width="23" height="23" fill="black"></rect>
                </svg>
                <h1 style="font-weight: 900; margin-bottom: 7px;margin-top:5px">
                  Stable Diffusion XL 0.9 Demo
                </h1>
              </div>
              <p style="margin-bottom: 10px; font-size: 94%; line-height: 23px;">
                Stable Diffusion XL 0.9 is the latest text-to-image model from StabilityAI. 
                <a style="text-decoration: underline;" href="https://huggingface.co/spaces/stabilityai/stable-diffusion">Access SD v2.1 Space</a> <a style="text-decoration: underline;" href="https://huggingface.co/spaces/stabilityai/stable-diffusion-1">SD v1 Space</a>
                <br/>
                For faster generation and API access you can try
                <a
                  href="http://beta.dreamstudio.ai/"
                  style="text-decoration: underline;"
                  target="_blank"
                  >DreamStudio Beta</a
                >.</a>
              </p>
            </div>
        """
    )
    with gr.Group():
        with gr.Box():
            with gr.Row(elem_id="prompt-container").style(mobile_collapse=False, equal_height=True):
                with gr.Column():
                    text = gr.Textbox(
                        label="Enter your prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt",
                        elem_id="prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                    negative = gr.Textbox(
                        label="Enter your negative prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter a negative prompt",
                        elem_id="negative-prompt-text-input",
                    ).style(
                        border=(True, False, True, True),
                        rounded=(True, False, False, True),
                        container=False,
                    )
                btn = gr.Button("Generate image").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )

        gallery = gr.Gallery(
            label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

        with gr.Group(elem_id="container-advanced-btns"):
            #advanced_button = gr.Button("Advanced options", elem_id="advanced-btn")
            with gr.Group(elem_id="share-btn-container"):
                community_icon = gr.HTML(community_icon_html)
                loading_icon = gr.HTML(loading_icon_html)
                share_button = gr.Button("Share to community", elem_id="share-btn")

        with gr.Accordion("Advanced settings", open=False):
        #    gr.Markdown("Advanced settings are temporarily unavailable")
        #    samples = gr.Slider(label="Images", minimum=1, maximum=4, value=4, step=1)
        #    steps = gr.Slider(label="Steps", minimum=1, maximum=50, value=45, step=1)
             guidance_scale = gr.Slider(
                label="Guidance Scale", minimum=0, maximum=50, value=9, step=0.1
             )
        #    seed = gr.Slider(
        #        label="Seed",
        #        minimum=0,
        #        maximum=2147483647,
        #        step=1,
        #        randomize=True,
        #    )

        ex = gr.Examples(examples=examples, fn=infer, inputs=[text, negative, guidance_scale], outputs=[gallery, community_icon, loading_icon, share_button], cache_examples=False)
        ex.dataset.headers = [""]
        negative.submit(infer, inputs=[text, negative, guidance_scale], outputs=[gallery], postprocess=False)
        text.submit(infer, inputs=[text, negative, guidance_scale], outputs=[gallery], postprocess=False)
        btn.click(infer, inputs=[text, negative, guidance_scale], outputs=[gallery], postprocess=False)
        
        #advanced_button.click(
        #    None,
        #    [],
        #    text,
        #    _js="""
        #    () => {
        #        const options = document.querySelector("body > gradio-app").querySelector("#advanced-options");
        #        options.style.display = ["none", ""].includes(options.style.display) ? "flex" : "none";
        #    }""",
        #)
        share_button.click(
            None,
            [],
            [],
            _js=share_js,
        )
        gr.HTML(
            """
                <div class="footer">
                    <p>Model by <a href="https://huggingface.co/stabilityai" style="text-decoration: underline;" target="_blank">StabilityAI</a> - Gradio Demo by 🤗 Hugging Face and <a style="text-decoration: underline;" href="https://tonylian.com/">Long (Tony) Lian</a>
                    </p>
                </div>
           """
        )
        with gr.Accordion(label="License", open=False):
            gr.HTML(
                """<div class="acknowledgments">
                    <p><h4>LICENSE</h4>
The model is licensed with a <a href="https://huggingface.co/stabilityai/stable-diffusion-2/blob/main/LICENSE-MODEL" style="text-decoration: underline;" target="_blank">CreativeML OpenRAIL++</a> license. The authors claim no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in this license. The license forbids you from sharing any content that violates any laws, produce any harm to a person, disseminate any personal information that would be meant for harm, spread misinformation and target vulnerable groups. For the full list of restrictions please <a href="https://huggingface.co/spaces/CompVis/stable-diffusion-license" target="_blank" style="text-decoration: underline;" target="_blank">read the license</a></p>
                    <p><h4>Biases and content acknowledgment</h4>
Despite how impressive being able to turn text into image is, beware to the fact that this model may output content that reinforces or exacerbates societal biases, as well as realistic faces, pornography and violence. The model was trained on the <a href="https://laion.ai/blog/laion-5b/" style="text-decoration: underline;" target="_blank">LAION-5B dataset</a>, which scraped non-curated image-text-pairs from the internet (the exception being the removal of illegal content) and is meant for research purposes. You can read more in the <a href="https://huggingface.co/CompVis/stable-diffusion-v1-4" style="text-decoration: underline;" target="_blank">model card</a></p>
               </div>
                """
            )
        

block.queue().launch()