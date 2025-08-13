import subprocess
import os
import json
import random
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import modal

# Build Modal Image with ComfyUI and dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("fastapi[standard]==0.115.4")
    .pip_install("comfy-cli==1.4.1")
    .run_commands(
        "comfy --skip-prompt install --fast-deps --nvidia --version 0.3.47"
    )
)

# Download custom nodes
image = image.run_commands(
    "comfy node install --fast-deps was-node-suite-comfyui@1.0.2",
    "git clone https://github.com/ChenDarYen/ComfyUI-NAG.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-NAG",
    "git clone https://github.com/kijai/ComfyUI-KJNodes.git /root/comfy/ComfyUI/custom_nodes/ComfyUI-KJNodes",
    "git clone https://github.com/cubiq/ComfyUI_essentials.git /root/comfy/ComfyUI/custom_nodes/ComfyUI_essentials",
)

def hf_download():
    from huggingface_hub import hf_hub_download
    
    # Download 14B text-to-image model (instead of video model)
    wan_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/diffusion_models/wan2.2_ti2v_14B_fp16.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {wan_model} /root/comfy/ComfyUI/models/diffusion_models/wan2.2_ti2v_14B_fp16.safetensors",
        shell=True,
        check=True,
    )
    
    # Download VAE
    vae_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.2_ComfyUI_Repackaged",
        filename="split_files/vae/wan2.2_vae.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {vae_model} /root/comfy/ComfyUI/models/vae/wan2.2_vae.safetensors",
        shell=True,
        check=True,
    )
    
    # Download text encoder
    t5_model = hf_hub_download(
        repo_id="Comfy-Org/Wan_2.1_ComfyUI_repackaged",
        filename="split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        cache_dir="/cache",
    )
    subprocess.run(
        f"ln -s {t5_model} /root/comfy/ComfyUI/models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        shell=True,
        check=True,
    )

vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
image = (
    image.pip_install("huggingface_hub[hf_transfer]>=0.34.0,<1.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        hf_download,
        volumes={"/cache": vol},
    )
)

app = modal.App(name="wan-14b-text-to-image", image=image)

# Request model for API
class ImageRequest(BaseModel):
    prompt: str
    negative_prompt: str = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
    seed: int = None
    steps: int = 20
    cfg: float = 5.0
    width: int = 1280
    height: int = 704

@app.function(
    gpu="L40S",  # Original GPU setting preserved
    memory=32768,
    timeout=600,
    volumes={"/cache": vol},
)
def generate_image(request: ImageRequest):
    import requests
    import time
    
    # Generate random seed if not provided
    if request.seed is None:
        request.seed = random.randint(1, 0xFFFFFFFF)
    
    # Text-to-image workflow for WAN 2.2 14B
    workflow = {
        "3": {
            "inputs": {
                "seed": request.seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": "uni_pc",
                "scheduler": "simple",
                "denoise": 1,
                "model": ["48", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0]
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler"
            }
        },
        "5": {
            "inputs": {
                "width": request.width,
                "height": request.height,
                "batch_size": 1
            },
            "class_type": "EmptyLatentImage",
            "_meta": {
                "title": "Empty Latent Image"
            }
        },
        "6": {
            "inputs": {
                "text": request.prompt,
                "clip": ["38", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Positive Prompt)"
            }
        },
        "7": {
            "inputs": {
                "text": request.negative_prompt,
                "clip": ["38", 0]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIP Text Encode (Negative Prompt)"
            }
        },
        "8": {
            "inputs": {
                "samples": ["3", 0],
                "vae": ["39", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAE Decode"
            }
        },
        "9": {
            "inputs": {
                "filename_prefix": "image/ComfyUI",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "Save Image"
            }
        },
        "37": {
            "inputs": {
                "unet_name": "wan2.2_ti2v_14B_fp16.safetensors",
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader",
            "_meta": {
                "title": "Load Diffusion Model"
            }
        },
        "38": {
            "inputs": {
                "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
                "type": "wan",
                "device": "default"
            },
            "class_type": "CLIPLoader",
            "_meta": {
                "title": "Load CLIP"
            }
        },
        "39": {
            "inputs": {
                "vae_name": "wan2.2_vae.safetensors"
            },
            "class_type": "VAELoader",
            "_meta": {
                "title": "Load VAE"
            }
        },
        "48": {
            "inputs": {
                "shift": 8,
                "model": ["37", 0]
            },
            "class_type": "ModelSamplingSD3",
            "_meta": {
                "title": "ModelSamplingSD3"
            }
        }
    }
    
    # Start ComfyUI server
    comfy_process = subprocess.Popen(
        "comfy launch -- --listen 0.0.0.0 --port 8000",
        shell=True
    )
    
    # Wait for ComfyUI to start
    time.sleep(30)
    
    try:
        # Submit workflow
        response = requests.post(
            "http://localhost:8000/prompt",
            json={
                "client_id": "modal",
                "prompt": workflow
            },
            timeout=30
        )
        response.raise_for_status()
        prompt_id = response.json()["prompt_id"]
        
        # Poll for completion
        max_wait_time = 500  # 8+ minutes for 14B model
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            history_response = requests.get(
                f"http://localhost:8000/history/{prompt_id}",
                timeout=30
            )
            history_response.raise_for_status()
            history = history_response.json()
            
            if prompt_id in history and history[prompt_id].get("status", {}).get("completed", False):
                if history[prompt_id]["status"]["status_str"] == "success":
                    # Get the generated image
                    outputs = history[prompt_id]["outputs"]
                    image_info = None
                    
                    # Find the saved image
                    for node_id, output in outputs.items():
                        if "images" in output:
                            image_info = output["images"][0]
                            break
                    
                    if image_info:
                        # Download the image
                        image_response = requests.get(
                            f"http://localhost:8000/view",
                            params={
                                "type": "output",
                                "filename": image_info["filename"],
                                "subfolder": image_info.get("subfolder", "")
                            },
                            timeout=30
                        )
                        image_response.raise_for_status()
                        
                        # Save image to volume for retrieval
                        image_path = f"/tmp/{prompt_id}.png"
                        with open(image_path, "wb") as f:
                            f.write(image_response.content)
                        
                        return {
                            "success": True,
                            "prompt_id": prompt_id,
                            "image_path": image_path,
                            "prompt": request.prompt,
                            "seed": request.seed
                        }
                    else:
                        raise HTTPException(status_code=500, detail="No image found in outputs")
                else:
                    raise HTTPException(status_code=500, detail="Generation failed")
            
            time.sleep(5)  # Check every 5 seconds
        
        raise HTTPException(status_code=408, detail="Generation timed out")
        
    finally:
        # Clean up ComfyUI process
        comfy_process.terminate()
        comfy_process.wait()

# Interactive ComfyUI server for development
@app.function(
    max_containers=1,
    gpu="L40S",  # Original GPU setting preserved
    volumes={"/cache": vol},
)
@modal.concurrent(max_inputs=10)
@modal.web_server(8000, startup_timeout=60)
def ui():
    subprocess.Popen("comfy launch -- --listen 0.0.0.0 --port 8000", shell=True)

# FastAPI app for API endpoints
web_app = FastAPI(title="WAN 2.2 14B Text-to-Image API")

@web_app.post("/generate")
async def api_generate_image(request: ImageRequest):
    """Generate an image from text using WAN 2.2 14B model"""
    result = generate_image.remote(request)
    return result

@app.function(
    memory=1024,
    volumes={"/cache": vol},
)
@modal.asgi_app()
def api():
    return web_app

if __name__ == "__main__":
    # For local testing
    with app.run():
        # Test the generation function
        test_request = ImageRequest(
            prompt="A beautiful landscape with mountains and a lake",
            negative_prompt="blurry, low quality, distorted",
            steps=20,
            cfg=5.0
        )
        result = generate_image.remote(test_request)
        print("Generation result:", result)
