import random
from types import SimpleNamespace

import pytest

from gepa.core.data_loader import ListDataLoader
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler


def test_epoch_sampler_refreshes_when_loader_expands():
    loader = ListDataLoader(["a", "b", "c", "d"])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    first_batch = sampler.next_minibatch_ids(loader, state)
    assert len(first_batch) == 2
    assert len(sampler.shuffled_ids) == 4
    assert sampler.last_trainset_size == 4

    state.i += 1
    loader.add_items(["e", "f"])

    second_batch = sampler.next_minibatch_ids(loader, state)
    assert len(second_batch) == 2
    assert sampler.last_trainset_size == 6
    assert len(sampler.shuffled_ids) == 6
    assert {4, 5}.issubset(set(sampler.shuffled_ids))


def test_epoch_sampler_errors_when_loader_empty():
    loader = ListDataLoader([])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(0))
    state = SimpleNamespace(i=0)

    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, state)


def test_epoch_sampler_is_reproducible_for_same_seed():
    loader = ListDataLoader([f"item-{i}" for i in range(12)])

    def sample_sequence(seed: int) -> list[list[int]]:
        sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(seed))
        return [sampler.next_minibatch_ids(loader, SimpleNamespace(i=i)) for i in range(4)]

    assert sample_sequence(7) == sample_sequence(7)


def test_epoch_sampler_varies_across_seeds():
    loader = ListDataLoader([f"item-{i}" for i in range(12)])

    sampler_a = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(7))
    sampler_b = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(8))

    batch_a = sampler_a.next_minibatch_ids(loader, SimpleNamespace(i=0))
    batch_b = sampler_b.next_minibatch_ids(loader, SimpleNamespace(i=0))

    assert batch_a != batch_b
