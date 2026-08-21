"""Speculative decoding: a small "draft" model proposes several tokens
cheaply, then the big "target" model verifies all of them in a single
batched forward pass — verifying k tokens costs about the same as decoding
one token normally, since it's still one memory-bound pass over the target
model's weights (just with a slightly larger batch). If the draft's guesses
are usually right, you get several tokens for close to the price of one.

Each round: draft model decodes `gamma` tokens one at a time (sequentially —
it's autoregressive too), then the target model verifies all `gamma` in one
step. Accepted tokens (plus one guaranteed "correction" token sampled by the
target when a draft token is rejected) are emitted per round.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.gpu_specs import GpuSpec
from engine.inference import decode_step_seconds
from engine.memory import ModelShape


def expected_tokens_per_round(gamma: int, acceptance_rate: float) -> float:
    """Expected number of tokens emitted per speculative round, including the
    target model's guaranteed correction token. acceptance_rate=1 means every
    draft token is always accepted (gamma+1 tokens/round, the ceiling).
    """
    if acceptance_rate >= 1.0:
        return gamma + 1
    return (1 - acceptance_rate ** (gamma + 1)) / (1 - acceptance_rate)


@dataclass(frozen=True)
class SpeculativeResult:
    draft_step_s: float
    verify_step_s: float
    round_s: float
    expected_tokens_per_round: float
    speculative_tokens_per_sec: float
    baseline_tokens_per_sec: float
    speedup: float


def simulate_speculative_decoding(
    draft_model: ModelShape,
    target_model: ModelShape,
    gpu: GpuSpec,
    precision: str,
    gamma: int,
    acceptance_rate: float,
    avg_kv_tokens: float,
    batch_size: int = 1,
) -> SpeculativeResult:
    draft_step_s = decode_step_seconds(draft_model, gpu, precision, batch_size, avg_kv_tokens)
    verify_step_s = decode_step_seconds(target_model, gpu, precision, batch_size, avg_kv_tokens)
    round_s = gamma * draft_step_s + verify_step_s

    tokens_per_round = expected_tokens_per_round(gamma, acceptance_rate)
    speculative_tokens_per_sec = (tokens_per_round * batch_size) / round_s

    baseline_step_s = decode_step_seconds(target_model, gpu, precision, batch_size, avg_kv_tokens)
    baseline_tokens_per_sec = batch_size / baseline_step_s

    return SpeculativeResult(
        draft_step_s=draft_step_s,
        verify_step_s=verify_step_s,
        round_s=round_s,
        expected_tokens_per_round=tokens_per_round,
        speculative_tokens_per_sec=speculative_tokens_per_sec,
        baseline_tokens_per_sec=baseline_tokens_per_sec,
        speedup=speculative_tokens_per_sec / baseline_tokens_per_sec,
    )
