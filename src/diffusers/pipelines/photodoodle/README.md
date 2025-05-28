# PhotoDoodle Pipeline

The PhotoDoodle pipeline is designed for image generation with conditional image input. It uses a combination of text and image conditioning to generate high-quality images.

## Model Architecture

The pipeline uses the following components:

1. **Transformer**: A FluxTransformer2DModel for denoising image latents
2. **VAE**: An AutoencoderKL for encoding/decoding images
3. **Text Encoders**: 
   - CLIP text encoder for initial text embedding
   - T5 encoder for additional text understanding
4. **Scheduler**: FlowMatchEulerDiscreteScheduler for the diffusion process

## Usage

```python
from diffusers import PhotoDoodlePipeline
import torch

pipeline = PhotoDoodlePipeline.from_pretrained("black-forest-labs/FLUX.1-dev")
pipeline = pipeline.to("cuda")
# Load initial model weights
pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle", weight_name="pretrain.safetensors")
pipeline.fuse_lora()
pipeline.unload_lora_weights()

pipeline.load_lora_weights("nicolaus-huang/PhotoDoodle",weight_name="sksmagiceffects.safetensors")

# Generate image with text prompt and condition image
prompt = "add a halo and wings for the cat by sksmagiceffects"
condition_image = load_image("path/to/condition.jpg")  # PIL Image
output = pipeline(
    prompt=prompt,
    condition_image=condition_image,
    num_inference_steps=28,
    guidance_scale=3.5
)

# Save the generated image
output.images[0].save("generated_image.png")
```

## Parameters

- `prompt`: Text prompt for image generation
- `prompt_2`: Optional secondary prompt for T5 encoder
- `condition_image`: Input image for conditioning
- `height`: Output image height (default: 512)
- `width`: Output image width (default: 512)
- `num_inference_steps`: Number of denoising steps (default: 28)
- `guidance_scale`: Classifier-free guidance scale (default: 3.5)
- `num_images_per_prompt`: Number of images to generate per prompt
- `generator`: Random number generator for reproducibility
- `output_type`: Output format ("pil", "latent", or "pt")

## Features

- Dual text encoder architecture (CLIP + T5)
- Image conditioning support
- Position encoding for better spatial understanding
- Support for LoRA fine-tuning
- VAE slicing and tiling for memory efficiency
- Progress bar during generation
- Callback support for step-by-step monitoring 