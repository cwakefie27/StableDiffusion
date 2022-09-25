# pip install torch, diffusers, transformers
# cmd -> git lfs install
# cmd -> git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# tags
#  -  Anime
#  -  Origami
#  -  Oil Painting
#  -  Steampunk
#  -  by josef thoma
#  -  BekSinSKi
#  -  Anthromorphic
#  -  Futuristic

# nouns
#  -  Shrek
#  -  Pokemon
#  -  Yughio
#  -  Magic
#  -  Food
#  -  Sport
#  -  Board Game

# helpful 
#  - Trending in Artstation
#  - 4k 
#  - Detailed
#  - Hyper realism
#  - Vivid Colors

prompts = ["Mid-Mod Wooden Living Room, Hyper Realism"]
#prompts = ["gordan ramsay vampire video game, hyper realism"]
#prompts = ["Board Game of Shrek Cult, Religion, Hyper Realism"]
#prompts = ["Disc Golf Low Back Bag Small Ergonomic"]
#prompts = ["Cyborg Programmer Future Action binary"]
#prompts = ["Fairy cottage under big tree in woods. Peaceful Serene Painting, Trending in Artstation"]
#prompts = ["Muhammad Ali Wearing Jordan 1 and Michael Jackson Jacket"]
#prompts = ["Danny Devito Muscalar Spandex Dominatrix Whip, Hyper Realism"]
#prompts = ["Nightmare Horror Demon Satan Shrek, Hyper Realism, 4k, Trending in Artstation"]
#prompts = ["Shrek as donkey portrait, Hyper Realism, 4k, Trending in Artstation"]
#prompts = ["Shrek as Donald Trump portrait, Hyper Realism, 4k, Trending in Artstation"]
#prompts = ["Tsunami Natural Disaster, trending in artstation"]
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

seeds = [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16, 17,18,19,20,21,22,23,24,25, 26,27,28,29,30,31,32,33,34,35,36]
num_inference_steps = 30
guidance_scale=7.5
strength = 0.75
eta = 0.6
height= 512
width = 512

base_dir = "D:\Results"
summary_dir = 'D:\Results\Summary'

nsfw_filter_disabled = True
open_image = False
open_summary = True
save_image = True
save_summary = True

use_init_image = False
use_mask_image = False
init_image_path = 'D:\Results\InputImages\leanne.jpg';
mask_image_path = '',

 # cuda | cpu
pipeline_processor = 'cuda'

from torch import autocast, float16, Generator
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from PIL import Image 
from os import makedirs, path
from math import sqrt, ceil

def image_grid(imgs, rows, cols):
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def dummy(images, **kwargs):
    return images, False

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

    adjusted_x = 512
    adjusted_y = 512
    if init_image.width > init_image.height:
        adjusted_x = ceil(init_image.width/(init_image.height/512))
        adjusted_x = adjusted_x + adjusted_x % 2;
    elif init_image.width < init_image.height:
        adjusted_y = ceil(init_image.height/(init_image.width/512))
        adjusted_y = adjusted_y + adjusted_y % 2;

    init_image = init_image.resize((adjusted_x, adjusted_y))

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

pipe = pipe.to(pipeline_processor)
pipe.enable_attention_slicing()

if nsfw_filter_disabled:
    pipe.safety_checker = dummy

cols = int(ceil(sqrt(len(seeds))))
rows = int(ceil(len(seeds) / cols))

for prompt in prompts:
    kwargs['prompt'] = prompt

    prompt_title = (prompt).title().replace(" ", "")
    prompt_description = f'{prompt_title}_I{num_inference_steps}_GS{guidance_scale}_ETA{eta}_{height}x{width}'

    prompt_dir = path.join(base_dir, pipe_line_title, prompt_description)
    print (f'{prompt} in {prompt_dir}')
    if not path.isdir(prompt_dir):
        makedirs(prompt_dir)
    
    images = []
    for idx, seed in enumerate(seeds):
        generator = Generator(pipeline_processor).manual_seed(seed)
        kwargs['generator'] = generator

        file_name = path.join(prompt_dir,  f'{seed}.png')

        print (f'Generating {file_name} ({idx+1} of {len(seeds)})')

        with autocast(pipeline_processor):

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
            if not path.isdir(summary_dir):
                makedirs(summary_dir)

            summary_file_name = path.join(summary_dir,  f'{prompt_description}.png')
            print (f'Summary {summary_file_name}')

            grid.save(summary_file_name)

        if open_summary:
            grid.show()

