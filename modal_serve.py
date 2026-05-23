"""
Deploy the inference server on Modal.

  modal serve modal_serve.py     — live reload dev mode
  modal deploy modal_serve.py    — persistent deployment
"""

import modal

app = modal.App("inference-engine-server")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install(
        "torch>=2.0.0", "transformers>=4.35.0", "accelerate>=0.24.0",
        "ninja", "fastapi>=0.110.0", "uvicorn>=0.29.0",
    )
    .add_local_dir("engine", remote_path="/root/engine")
    .add_local_dir("kernels", remote_path="/root/kernels")
)


@app.function(gpu="A10G", image=image, timeout=600, scaledown_window=300)
@modal.asgi_app()
def serve():
    import sys
    sys.path.insert(0, "/root")
    from engine.server import app as fastapi_app
    return fastapi_app
