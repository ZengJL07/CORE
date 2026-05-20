from __future__ import annotations

import math
import random


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    scaled = [value / temperature for value in values]
    max_value = max(scaled)
    exps = [math.exp(value - max_value) for value in scaled]
    total = sum(exps)
    if total == 0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def gumbel_topk_indices(
    scores: list[float],
    k: int,
    *,
    tau: float = 1.0,
    rng: random.Random,
) -> list[int]:
    if k < 1:
        return []
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    perturbed = []
    for idx, score in enumerate(scores):
        u = min(max(rng.random(), 1e-9), 1.0 - 1e-9)
        gumbel_noise = -math.log(-math.log(u))
        perturbed.append((score / tau + gumbel_noise, idx))
    perturbed.sort(reverse=True)
    return [idx for _, idx in perturbed[: min(k, len(perturbed))]]


def weighted_sample_without_replacement(
    population: list[int],
    weights: list[float],
    k: int,
    *,
    rng: random.Random,
) -> list[int]:
    if k < 1 or not population:
        return []
    if len(population) != len(weights):
        raise ValueError("population and weights must have the same length")

    remaining_items = list(population)
    remaining_weights = [max(weight, 0.0) for weight in weights]
    sampled: list[int] = []
    sample_count = min(k, len(remaining_items))

    for _ in range(sample_count):
        total_weight = sum(remaining_weights)
        if total_weight <= 0:
            choice_idx = rng.randrange(len(remaining_items))
        else:
            threshold = rng.random() * total_weight
            running = 0.0
            choice_idx = len(remaining_items) - 1
            for idx, weight in enumerate(remaining_weights):
                running += weight
                if threshold <= running:
                    choice_idx = idx
                    break

        sampled.append(remaining_items.pop(choice_idx))
        remaining_weights.pop(choice_idx)

    return sampled
