from dataclasses import dataclass
from typing import Iterator

import torch

from engine.block_manager import BlockManager
from engine.sampling import greedy, top_p as top_p_sample
from engine.model.transformer import TINYLLAMA_CONFIG


@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    greedy: bool = False


class Runner:
    def __init__(self, model, tokenizer, device: str = "cuda", num_blocks: int = 512):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        cfg = TINYLLAMA_CONFIG
        self.block_manager = BlockManager(
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=cfg.num_key_value_heads,
            head_dim=cfg.head_dim,
            num_blocks=num_blocks,
            device=device,
        )
        self._seq_counter = 0

    def _next_seq_id(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def generate(self, prompt: str, config: GenerationConfig) -> Iterator[str]:
        seq_id = self._next_seq_id()
        input_ids = self.tokenizer.encode(prompt)
        prompt_len = len(input_ids)

        if not self.block_manager.can_allocate(prompt_len):
            raise RuntimeError("Not enough KV cache blocks for prompt")

        self.block_manager.allocate(seq_id, prompt_len)

        prompt_tensor = torch.tensor(input_ids, dtype=torch.long, device=self.device)
        positions = torch.arange(prompt_len, dtype=torch.long, device=self.device)

        try:
            with torch.no_grad():
                logits = self.model(prompt_tensor, positions, seq_id, self.block_manager, is_prefill=True)

            generated_ids = []
            prev_text = ""

            for _ in range(config.max_new_tokens):
                if config.greedy:
                    next_token = greedy(logits)
                else:
                    next_token = top_p_sample(logits, p=config.top_p, temperature=config.temperature)

                if next_token == self.tokenizer.eos_token_id:
                    break

                generated_ids.append(next_token)
                # Decode the full sequence so far — SentencePiece needs context to place spaces
                new_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                delta = new_text[len(prev_text):]
                prev_text = new_text
                if delta:
                    yield delta

                if not self.block_manager.can_append(seq_id):
                    break

                self.block_manager.append_slot(seq_id)
                token_tensor = torch.tensor([next_token], dtype=torch.long, device=self.device)
                cur_pos = self.block_manager.seq_lengths[seq_id] - 1
                position = torch.tensor([cur_pos], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    logits = self.model(token_tensor, position, seq_id, self.block_manager, is_prefill=False)

        finally:
            self.block_manager.free(seq_id)
