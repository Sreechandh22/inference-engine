from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch

from engine.block_manager import BlockManager
from engine.runner import GenerationConfig
from engine.sampling import greedy, top_p as top_p_sample

_CHAT_MARKERS = ["<|assistant|>", "<|user|>", "<|system|>", "<|im_start|>", "<|im_end|>"]


@dataclass
class DecodeBatch:
    """Per-step batched decode metadata, shared across all transformer layers."""
    seq_ids: List[int]
    positions: torch.Tensor    # [B] long
    block_tables: torch.Tensor # [B, max_blocks] int32
    seq_lens: torch.Tensor     # [B] int32
    slot_mapping: torch.Tensor # [B] long — flat physical slot for KV write


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
        Run one token-generation iteration.
        Prefill any new sequences individually, then run ALL decode-ready
        sequences in a single batched forward pass.
        Returns {seq_id: token_str}; None means the sequence finished this step.
        """
        self._admit_waiting()
        if not self.running:
            return {}

        outputs: Dict[int, Optional[str]] = {}
        finished: List[Sequence] = []

        # --- Phase 1: prefill any sequence that hasn't been prefilled ---
        for seq in self.running:
            if not seq.prefilled:
                input_ids = torch.tensor(seq.prompt_ids, dtype=torch.long, device=self.device)
                positions = torch.arange(seq.prompt_len, dtype=torch.long, device=self.device)
                with torch.no_grad():
                    seq.logits = self.model(
                        input_ids, positions, seq.seq_id, self.block_manager, is_prefill=True
                    )
                seq.prefilled = True

        # --- Phase 2: sample next token, run termination/preempt, collect batch ---
        batch_seqs: List[Sequence] = []
        batch_tokens: List[int] = []
        for seq in self.running:
            next_token = seq.sample_next()

            eos_allowed = len(seq.generated_ids) >= 4
            if (eos_allowed and next_token == self.tokenizer.eos_token_id) \
                    or len(seq.generated_ids) >= seq.config.max_new_tokens:
                outputs[seq.seq_id] = None
                finished.append(seq)
                self.block_manager.free(seq.seq_id)
                continue

            seq.generated_ids.append(next_token)

            if not self.block_manager.can_append(seq.seq_id):
                self.block_manager.free(seq.seq_id)
                seq.prefilled = False
                seq.generated_ids = []
                self.waiting.appendleft(seq)
                finished.append(seq)
                outputs[seq.seq_id] = None
                continue

            batch_seqs.append(seq)
            batch_tokens.append(next_token)

        for seq in finished:
            if seq in self.running:
                self.running.remove(seq)

        if not batch_seqs:
            return outputs

        # --- Phase 3: reserve slots + build batch tensors ---
        tpb = self.block_manager.tokens_per_block
        slot_mapping = []
        for seq in batch_seqs:
            block_id, offset = self.block_manager.append_slot(seq.seq_id)
            slot_mapping.append(block_id * tpb + offset)

        positions = [self.block_manager.seq_lengths[s.seq_id] - 1 for s in batch_seqs]
        seq_lens = [self.block_manager.seq_lengths[s.seq_id] for s in batch_seqs]
        max_blocks = max(len(self.block_manager.block_tables[s.seq_id]) for s in batch_seqs)

        bt = torch.zeros(len(batch_seqs), max_blocks, dtype=torch.int32, device=self.device)
        for i, s in enumerate(batch_seqs):
            blocks = self.block_manager.block_tables[s.seq_id]
            bt[i, :len(blocks)] = torch.tensor(blocks, dtype=torch.int32, device=self.device)

        batch = DecodeBatch(
            seq_ids=[s.seq_id for s in batch_seqs],
            positions=torch.tensor(positions, dtype=torch.long, device=self.device),
            block_tables=bt,
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=self.device),
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=self.device),
        )
        input_ids = torch.tensor(batch_tokens, dtype=torch.long, device=self.device)

        # --- Phase 4: ONE batched forward pass over all decode sequences ---
        with torch.no_grad():
            logits = self.model.forward_decode(input_ids, batch, self.block_manager)  # [B, vocab]

        # --- Phase 5: stash next-step logits + emit decoded deltas ---
        for i, seq in enumerate(batch_seqs):
            seq.logits = logits[i]
            all_text = self.tokenizer.decode(seq.generated_ids, skip_special_tokens=True)
            prev_text = self.tokenizer.decode(seq.generated_ids[:-1], skip_special_tokens=True)
            delta = all_text[len(prev_text):]
            for m in _CHAT_MARKERS:
                delta = delta.replace(m, "")
            outputs[seq.seq_id] = delta if delta else ""

        return outputs
