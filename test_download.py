from diffusers import StableDiffusionInpaintPipeline
import torch

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "CompVis/stable-diffusion-v-1-5-inpainting",  # 你可以在这里直接下载模型
    torch_dtype=torch.float16,
)
