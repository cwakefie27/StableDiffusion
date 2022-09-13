# pip install torch, diffusers, transformers, huggingface_hub
# cmd -> git lfs install
# cmd -> git clone https://huggingface.co/CompVis/stable-diffusion-v1-4

# For Developers

# https://huggingface.co/settings/tokens
# huggingface-cli login 
# username: cwakefield, password: Monkey27$

# Test Torch
# Consider using CUDA
"""
import torch

x = torch.rand(5, 3)
print(x)
print(torch.cuda.is_available())
"""

# Behind Proxy
'''
import os
os.environ['HTTP_PROXY'] = '192.168.232.5:3128';
os.environ['HTTPS_PROXY'] = '192.168.232.5:3128';
'''

# make sure you're logged in with `huggingface-cli login`
from torch import autocast
from diffusers import StableDiffusionPipeline

token = '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'

# With Locally downloaded model
#pipe = StableDiffusionPipeline.from_pretrained("./stable-diffusion-v1-4")
# Without locally downloaded model
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", use_auth_token=token)
pipe = pipe.to("cuda")

prompt = "a photo of an astronaut riding a horse on mars"
with autocast("cuda"):
    image = pipe(prompt).images[0]  