# pip install torch, diffusers, transformers
# cmd -> git lfs install
# cmd -> git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

prompts = ["Tsunami Natural Disaster, trending in artstation"]
#prompts = ["A physical therapy shrek doctor, trending in artstation"]
#prompts = ["Origami Dragon, trending in artstation"]
#prompts = ["Weed smoked man plays twilight imperium in october"]
#prompts = ["Woodworking egg holder"]
#prompts = ["Landscape Lake Water Peaceful, trending in artstation"]
#prompts = ["House Plant Abundance Tree Girl , 4k, detailed, trending in artstation"]
#prompts = ["Landscape Georgraphy Mountain Range, 4k, detailed, trending in artstation"]
#prompts = ["pokemon Yughio Chicken, 4k, detailed, trending in artstation, vivid colors"]
#prompts = ["oil painting chicken best friends, 4k, detailed, trending in artstation, vivid colors"]
#prompts = ["Anime Japanese chicken, 4k, detailed, trending in artstation, vivid colors"]
#prompts = ["cyberpunk chicken, hyper realism, 4k, detailed, trending in artstation, fantasy vivid colors"]
#prompts = ["christmas ornament macrame"]
#prompts = ["donald trump machine gun america rock and roll"]
#prompts = ["barred rock hen sitting in chair"]
#prompts = ["Mr. Potato Head Satan Army"]
#prompts = ["brittany spears dancing in space"]
#prompts = ["eagle with donald trump face"]
#prompts = ["Watercolor peach face lovebird"]
#prompts = ["painting of fantasy chicken warrior"]

seeds = [1,2,3,4, 5,6,7,8, 9,10,11,12]
num_inference_steps = 30
guidance_scale=7.5
strength = 0.75
eta = 0.6
height= 512
width = 512
init_image_path = 'D:\Results2\InputImages\sketch-mountains.jpg';
mask_image_path = '',

use_init_image = True
use_mask_image = False
nsfw_filter_disabled = True
save_image = True
open_image = False
save_summary = True
open_summary = True

base_directory = "D:\Results2"
summary_dir = 'Summary'

from torch import autocast, float16, Generator
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image 
from os import mkdir, path
from math import sqrt, ceil

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def dummy(images, **kwargs):
    return images, False

# tags
#  -  Anime
#  -  Oil Painting
#  -  Steampunk
#  -  by josef thoma
#  -  BekSinSKi
#  -  Anthromorphic

# nouns
#  -  Pokemon
#  -  Yughio
#  -  Magic

# helpful 
#  - Trending in Artstation
#  - 4k 
#  - Detailed
#  - Hyper realism
#  - Vivid Colors


kwargs = {}

kwargs['num_inference_steps'] = num_inference_steps
kwargs['guidance_scale'] = guidance_scale
kwargs['strength'] = strength
kwargs['eta'] = eta

pipe_line = None
pipe_line_title = None
if use_init_image:
    pipe_line_title = 'Image'
    pipe_line = StableDiffusionImg2ImgPipeline
    init_image = Image.open(init_image_path).convert("RGB")
    init_image = init_image.resize((768, 512))
    kwargs['init_image'] = init_image

    if use_mask_image:
        mask_image = Image.open(mask_image_path).convert("RGB")
        kwargs['mask_image'] = mask_image
        

else:
    pipe_line_title = 'Text'
    pipe_line = StableDiffusionPipeline
    kwargs['height'] = height
    kwargs['width'] = width

# With Locally downloaded model
pipe = pipe_line.from_pretrained(
    "./stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=float16,
)
# Without locally downloaded model
'''
token = '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'
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    revision="fp16", 
    torch_dtype=float16,
    use_auth_token=token)
'''

pipe = pipe.to("cuda")
pipe.enable_attention_slicing()

if nsfw_filter_disabled:
    pipe.safety_checker = dummy

cols = int(ceil(sqrt(len(seeds))))
rows = int(ceil(len(seeds) / cols))

for prompt in prompts:
    kwargs['prompt'] = prompt

    prompt_title = (prompt).title().replace(" ", "")
    prompt_description = f'{pipe_line_title}_{prompt_title}_I{num_inference_steps}_GS{guidance_scale}_ETA{eta}_{height}x{width}'

    prompt_dir = path.join(base_directory, prompt_description)
    print (f'{prompt} in {prompt_dir}')
    if not path.isdir(prompt_dir):
        mkdir(prompt_dir)
    
    images = []
    for idx, seed in enumerate(seeds):
        generator = Generator("cuda").manual_seed(seed)
        kwargs['generator'] = generator

        file_name = path.join(prompt_dir,  f'{seed}.png')

        print (f'Generating {file_name} ({idx+1} of {len(seeds)})')

        with autocast("cuda"):

            result = pipe(
                **kwargs)  
            
            image = result.images[0]

            if save_image:
                image.save(file_name)

            if open_image:
                grid.show()

            images.append(image)

    if save_summary or open_summary:
        grid = image_grid(images, rows=rows, cols=cols)

        if save_summary:
            summary_path = path.join(base_directory, summary_dir)

            if not path.isdir(summary_path):
                mkdir(summary_path)

            summary_file_name = path.join(summary_path,  f'{prompt_description}.png')
            print (f'Summary {summary_file_name}')

            grid.save(summary_file_name)

        if open_summary:
            grid.show()

