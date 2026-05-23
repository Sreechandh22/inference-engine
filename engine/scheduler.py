from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from engine.block_manager import BlockManager
from engine.runner import GenerationConfig
from engine.sampling import greedy, top_p as top_p_sample


@dataclass
class Sequence:
    seq_id: int
    prompt_ids: List[int]
    config: GenerationConfig
    generated_ids: List[int] = field(default_factory=list)
    logits: Optional[torch.Tensor] = None
    prefilled: bool = False

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    def sample_next(self) -> int:
        if self.config.greedy:
            return greedy(self.logits)
        return top_p_sample(self.logits, p=self.config.top_p, temperature=self.config.temperature)

    def is_done(self, eos_token_id: int) -> bool:
        if not self.generated_ids:
            return False
        return (
            self.generated_ids[-1] == eos_token_id
            or len(self.generated_ids) >= self.config.max_new_tokens
        )


class Scheduler:
    """
    Continuous batching scheduler.
    Manages a waiting queue and a running set of sequences.
    Each step runs one token iteration for ALL running sequences —
    when one finishes, a waiting sequence takes its slot immediately.
    """

    def __init__(
        self,
        model,
        tokenizer,
        block_manager: BlockManager,
        device: str = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.block_manager = block_manager
        self.device = device
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []
        self._seq_counter = 0

    def add_request(self, prompt: str, config: GenerationConfig) -> int:
        seq_id = self._next_seq_id()
        prompt_ids = self.tokenizer.encode(prompt)
        self.waiting.append(Sequence(seq_id=seq_id, prompt_ids=prompt_ids, config=config))
        return seq_id

    def has_work(self) -> bool:
        return bool(self.waiting) or bool(self.running)

    def _next_seq_id(self) -> int:
        self._seq_counter += 1
        return self._seq_counter

    def _admit_waiting(self):
        """Pull sequences from waiting queue into running set as memory allows."""
        while self.waiting:
            seq = self.waiting[0]
            if self.block_manager.can_allocate(seq.prompt_len):
                self.waiting.popleft()
                self.block_manager.allocate(seq.seq_id, seq.prompt_len)
                self.running.append(seq)
            else:
                # Preempt the most recently admitted sequence to free space
                if self.running:
                    victim = self.running.pop()
                    self.block_manager.free(victim.seq_id)
                    victim.prefilled = False
                    victim.generated_ids = []
                    self.waiting.appendleft(victim)
                break

    def step(self) -> Dict[int, Optional[str]]:
        """
        Run one token-generation iteration across all running sequences.
        Returns {seq_id: token_str}.
        None value means the sequence finished this step.
        """
        self._admit_waiting()
        if not self.running:
            return {}

        outputs: Dict[int, Optional[str]] = {}
        finished: List[Sequence] = []

        for seq in self.running:
            if not seq.prefilled:
                # Prefill: process the full prompt in one forward pass
                input_ids = torch.tensor(seq.prompt_ids, dtype=torch.long, device=self.device)
                positions = torch.arange(seq.prompt_len, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    seq.logits = self.model(
                        input_ids, positions, seq.seq_id, self.block_manager, is_prefill=True
                    )
                seq.prefilled = True

            next_token = seq.sample_next()

            # Check termination
            if next_token == self.tokenizer.eos_token_id or len(seq.generated_ids) >= seq.config.max_new_tokens:
                outputs[seq.seq_id] = None
                finished.append(seq)
                self.block_manager.free(seq.seq_id)
                continue

            seq.generated_ids.append(next_token)

            # Check if we can append a KV slot; preempt if not
            if not self.block_manager.can_append(seq.seq_id):
                self.block_manager.free(seq.seq_id)
                seq.prefilled = False
                seq.generated_ids = []
                self.waiting.appendleft(seq)
                finished.append(seq)
                outputs[seq.seq_id] = None
                continue

            self.block_manager.append_slot(seq.seq_id)
            token_tensor = torch.tensor([next_token], dtype=torch.long, device=self.device)
            cur_pos = self.block_manager.seq_lengths[seq.seq_id] - 1
            position = torch.tensor([cur_pos], dtype=torch.long, device=self.device)
            with torch.no_grad():
                seq.logits = self.model(
                    token_tensor, position, seq.seq_id, self.block_manager, is_prefill=False
                )

            # Incremental decode (SentencePiece-aware)
            all_text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
            prev_text = self.tokenizer.decode(seq.generated_ids[:-1], skip_special_tokens=True)
            delta = all_text[len(prev_text):]
            outputs[seq.seq_id] = delta if delta else ""

        for seq in finished:
            if seq in self.running:
                self.running.remove(seq)

        return outputs
