import torch
import torch.nn.functional as F


def greedy(logits: torch.Tensor) -> int:
    return logits.argmax(dim=-1).item()


def top_p(logits: torch.Tensor, p: float = 0.9, temperature: float = 1.0) -> int:
    logits = logits / max(temperature, 1e-5)
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative - sorted_probs > p] = 0.0
    sorted_probs /= sorted_probs.sum()
    return sorted_indices[torch.multinomial(sorted_probs, 1)].item()
