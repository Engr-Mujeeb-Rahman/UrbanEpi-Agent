"""Simulated secure aggregation: real pairwise masks, without key exchange or dropout recovery."""
import numpy as np

MODULUS_BITS = 62
_MASK = (1 << MODULUS_BITS) - 1


def encode(values: np.ndarray, scale_bits: int) -> np.ndarray:
    """Convert real values to fixed-point integers modulo 2^62."""
    scaled = np.rint(np.asarray(values, dtype=np.float64) * (1 << scale_bits)).astype(np.int64)
    return scaled.astype(np.uint64) & np.uint64(_MASK)


def decode(values: np.ndarray, scale_bits: int) -> np.ndarray:
    """Convert a modular sum back to real values."""
    unsigned = np.asarray(values, dtype=np.uint64) & np.uint64(_MASK)
    signed = unsigned.astype(np.int64)
    signed = np.where(signed >= (1 << (MODULUS_BITS - 1)), signed - (1 << MODULUS_BITS), signed)
    return signed / (1 << scale_bits)


def pairwise_masks(rng: np.random.Generator, parties: int, shape: tuple[int, ...]) -> list[np.ndarray]:
    """Create one mask per party. The masks add up to zero modulo 2^62."""
    masks = [np.zeros(shape, dtype=np.uint64) for _ in range(parties)]
    for first in range(parties):
        for second in range(first + 1, parties):
            shared = rng.integers(0, 1 << MODULUS_BITS, size=shape, dtype=np.uint64)
            masks[first] = (masks[first] + shared) & np.uint64(_MASK)
            masks[second] = (masks[second] - shared) & np.uint64(_MASK)
    return masks


def add_mask(encoded: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Hide one party's encoded values."""
    return (encoded + mask) & np.uint64(_MASK)


def aggregate(shares: list[np.ndarray]) -> np.ndarray:
    """Add masked shares. Only the total can be read."""
    total = np.zeros_like(shares[0])
    for share in shares:
        total = (total + share) & np.uint64(_MASK)
    return total
