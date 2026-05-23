from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

def load_model(model_name: str = DEFAULT_MODEL, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    model.eval()
    return model, tokenizer
