import torch
from transformers import AutoTokenizer

from engine.model.transformer import TinyLlamaModel, ModelConfig, TINYLLAMA_CONFIG

DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


def load_model(
    model_name: str = DEFAULT_MODEL,
    device: str = "cuda",
    config: ModelConfig = TINYLLAMA_CONFIG,
) -> tuple[TinyLlamaModel, AutoTokenizer]:
    from transformers import AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load HuggingFace weights, then transfer into our architecture
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16
    )
    hf_state = hf_model.state_dict()

    model = TinyLlamaModel(config)
    our_state = model.state_dict()
    mapped = {}

    # Top-level
    mapped["embed_tokens.weight"] = hf_state["model.embed_tokens.weight"]
    mapped["norm.weight"]         = hf_state["model.norm.weight"]
    mapped["lm_head.weight"]      = hf_state["lm_head.weight"]

    # Per-layer
    for i in range(config.num_hidden_layers):
        hf_prefix = f"model.layers.{i}"
        our_prefix = f"layers.{i}"

        mapped[f"{our_prefix}.input_layernorm.weight"]          = hf_state[f"{hf_prefix}.input_layernorm.weight"]
        mapped[f"{our_prefix}.post_attention_layernorm.weight"] = hf_state[f"{hf_prefix}.post_attention_layernorm.weight"]
        mapped[f"{our_prefix}.self_attn.q_proj.weight"]         = hf_state[f"{hf_prefix}.self_attn.q_proj.weight"]
        mapped[f"{our_prefix}.self_attn.k_proj.weight"]         = hf_state[f"{hf_prefix}.self_attn.k_proj.weight"]
        mapped[f"{our_prefix}.self_attn.v_proj.weight"]         = hf_state[f"{hf_prefix}.self_attn.v_proj.weight"]
        mapped[f"{our_prefix}.self_attn.o_proj.weight"]         = hf_state[f"{hf_prefix}.self_attn.o_proj.weight"]
        mapped[f"{our_prefix}.mlp.gate_proj.weight"]            = hf_state[f"{hf_prefix}.mlp.gate_proj.weight"]
        mapped[f"{our_prefix}.mlp.up_proj.weight"]              = hf_state[f"{hf_prefix}.mlp.up_proj.weight"]
        mapped[f"{our_prefix}.mlp.down_proj.weight"]            = hf_state[f"{hf_prefix}.mlp.down_proj.weight"]

    # Verify no keys are missing or unexpected
    missing = set(our_state.keys()) - set(mapped.keys())
    unexpected = set(mapped.keys()) - set(our_state.keys())
    if missing:
        raise RuntimeError(f"Missing weights: {missing}")
    if unexpected:
        raise RuntimeError(f"Unexpected weights: {unexpected}")

    model.load_state_dict(mapped)
    model.half().to(device).eval()  # match float16 weights + KV cache

    del hf_model, hf_state
    torch.cuda.empty_cache()

    return model, tokenizer
