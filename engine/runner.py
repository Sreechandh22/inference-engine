from dataclasses import dataclass
from typing import Iterator

import torch

from engine.sampling import greedy, top_p as top_p_sample


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    greedy: bool = False


class Runner:
    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        past_key_values = None

        for _ in range(config.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            logits = outputs.logits[0, -1, :]
            past_key_values = outputs.past_key_values

            if config.greedy:
                next_token = greedy(logits)
            else:
                next_token = top_p_sample(logits, p=config.top_p, temperature=config.temperature)

            if next_token == self.tokenizer.eos_token_id:
                break

            yield self.tokenizer.decode([next_token], skip_special_tokens=True)
            input_ids = torch.tensor([[next_token]], device=self.device)
