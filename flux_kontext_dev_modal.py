from io import BytesIO
from pathlib import Path
from typing import Optional

import modal

# Define the container image with all necessary dependencies
diffusers_commit_sha = "00f95b9755718aabb65456e791b8408526ae6e76"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .entrypoint([])  # remove verbose logging by base image on entry
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        f"uv pip install --system --compile-bytecode --index-strategy unsafe-best-match accelerate~=1.8.1 git+https://github.com/huggingface/diffusers.git@{diffusers_commit_sha} huggingface-hub[hf-transfer]~=0.33.1 Pillow~=11.2.1 safetensors~=0.5.3 transformers~=4.53.0 sentencepiece~=0.2.0 torch==2.7.1 optimum-quanto==0.2.7 fastapi[standard]==0.115.4 python-multipart==0.0.12 --extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

CACHE_DIR = Path("/cache")
cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
volumes = {CACHE_DIR: cache_volume}

secrets = [modal.Secret.from_name("flux-app-secrets", required_keys=["HF_TOKEN"])]

image = image.env(
    {
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # Allows faster model downloads
        "HF_HOME": str(CACHE_DIR),  # Points the Hugging Face cache to a Volume
    }
)

app = modal.App("flux-kontext-fastapi")

with image.imports():
    import torch
    import os
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

@app.cls(
    image=image,
    cpu="0.5",
    memory="2GiB",
    gpu="L40s",
    volumes=volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=10 * 60, # 10 minutes
)
class FluxKontextModel:
    @modal.enter()
    def enter(self):
        print(f"Downloading {MODEL_NAME} if necessary...")

        dtype = torch.bfloat16
        self.device = "cuda"

        self.pipe = FluxKontextPipeline.from_pretrained(
            MODEL_NAME,
            revision=MODEL_REVISION,
            torch_dtype=dtype,
            cache_dir=CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)

    @modal.method()
    def inference(
        self,
        prompt: str,
        image_bytes: Optional[bytes] = None,
        width: int = 1024,
        height: int = 1024,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 20,
        seed: Optional[int] = None,
    ) -> bytes:
        # Set seed
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)

        # Editing mode
        if image_bytes:
            init_image = load_image(Image.open(BytesIO(image_bytes)))
            result = self.pipe(
                image=init_image,
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                generator=generator,
            )
        else:
            # Generation mode
            result = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                output_type="pil",
                generator=generator,
            )

        image = result.images[0]
        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG")
        return byte_stream.getvalue()

# FastAPI web interface
@app.function(image=image, volumes=volumes, secrets=secrets, cpu="0.5", memory="2GiB")
@modal.asgi_app()
def fastapi_app():
    from fastapi import Depends, FastAPI, File, UploadFile, Form, HTTPException, status
    from fastapi.responses import Response
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from PIL import Image
    from io import BytesIO
    import os

    web_app = FastAPI(
        title="Flux Kontext Generator & Editor",
        description="Generate or edit images using Flux Kontext Dev model",
        version="1.0.0",
    )

    @web_app.post("/generate_or_edit")
    async def generate_or_edit(
        prompt: str = Form(..., description="Prompt for generation/editing"),
        image: Optional[UploadFile] = File(None, description="Optional input image"),
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        guidance_scale: float = Form(3.5),
        num_inference_steps: int = Form(20),
        seed: Optional[int] = Form(None),
        width: Optional[int] = Form(1024),
        height: Optional[int] = Form(1024),
    ):
        # Optional auth check
        if os.environ.get("BEARER_TOKEN", False):
            if not token or token.credentials != os.environ["BEARER_TOKEN"]:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect bearer token",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        image_bytes = None
        if image:
            if image.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
                raise HTTPException(status_code=400, detail="Invalid file type")
            image_bytes = await image.read()
            if len(image_bytes) > 10 * 1024 * 1024:
                raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")

        model = FluxKontextModel()
        result_bytes = model.inference.remote(
            prompt=prompt,
            image_bytes=image_bytes,
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        return Response(
            content=result_bytes,
            media_type="image/png",
            headers={"Content-Disposition": "inline; filename=result.png"},
        )

    return web_app

# CLI local test
@app.local_entrypoint()
def main(
    prompt: str = "A fantasy castle at sunset",
    image_path: Optional[str] = None,
    output_path: str = "/tmp/output.png",
    width: int = 1024,
    height: int = 1024,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 20,
    seed: Optional[int] = None,
):
    print(f"🎨 Prompt: {prompt}")

    image_bytes = None
    if image_path:
        try:
            image_bytes = Path(image_path).read_bytes()
        except FileNotFoundError:
            print(f"❌ Image file not found at {image_path}")
            return

    model = FluxKontextModel()
    result_bytes = model.inference.remote(
        prompt=prompt,
        image_bytes=image_bytes,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        seed=seed,
    )

    Path(output_path).write_bytes(result_bytes)
    print(f"✅ Saved result → {output_path}")
