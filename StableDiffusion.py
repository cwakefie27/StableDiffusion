# pip install torch, diffusers, transformers 
# cmd -> git lfs install
# cmd -> git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# Test Torch
# Consider using CUDA
"""
import torch

x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
"""

# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=True)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt).images[0]  