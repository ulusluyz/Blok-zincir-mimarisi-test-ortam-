"""
Bitwise Utilities and Mathematical Similarity Functions.
Supports 64, 128, 256-bit representation vectors packed as tuples of 64-bit integers or Python arbitrary precision ints.

Methods implemented:
1. S_XOR(A, Q) = 1.0 - POPCOUNT(A XOR Q) / N  (Formula required in task)
2. S_BinarizedDot(A, Q) = POPCOUNT(A AND Q) / (POPCOUNT(A OR Q) + 1e-9)  (Jaccard/Binarized Dot)
3. S_SignCosine(A, Q) = 1.0 - POPCOUNT(A XOR Q) / (0.5 * N)  (Rescaled Hamming / Sign Cosine equivalent in [-1, 1])
"""

from typing import List, Tuple, Union
import math

class BinaryVector:
    """
    Represents an N-bit binary vector (N in {64, 128, 256, 512, 1024, ...}).
    Stored as a Python int or tuple of 64-bit ints.
    """

    def __init__(self, bits: int, bit_length: int = 128):
        self.bit_length = bit_length
        self.mask = (1 << bit_length) - 1
        self.bits = bits & self.mask

    @classmethod
    def from_bool_list(cls, bool_list: List[bool]) -> 'BinaryVector':
        bit_length = len(bool_list)
        val = 0
        for b in bool_list:
            val = (val << 1) | (1 if b else 0)
        return cls(val, bit_length=bit_length)

    @classmethod
    def from_float_list(cls, float_list: List[float], threshold: float = 0.0) -> 'BinaryVector':
        bool_list = [v > threshold for v in float_list]
        return cls.from_bool_list(bool_list)

    def to_bool_list(self) -> List[bool]:
        res = []
        for i in range(self.bit_length - 1, -1, -1):
            res.append(bool((self.bits >> i) & 1))
        return res

    def popcount(self) -> int:
        return self.bits.bit_count()

    def xor(self, other: 'BinaryVector') -> 'BinaryVector':
        assert self.bit_length == other.bit_length
        return BinaryVector(self.bits ^ other.bits, self.bit_length)

    def bit_and(self, other: 'BinaryVector') -> 'BinaryVector':
        assert self.bit_length == other.bit_length
        return BinaryVector(self.bits & other.bits, self.bit_length)

    def bit_or(self, other: 'BinaryVector') -> 'BinaryVector':
        assert self.bit_length == other.bit_length
        return BinaryVector(self.bits | other.bits, self.bit_length)


def similarity_xor_popcount(A: BinaryVector, Q: BinaryVector) -> float:
    """
    Formül: S(A, Q) = 1 - POPCOUNT(A XOR Q) / N
    Returns similarity in range [0.0, 1.0].
    """
    assert A.bit_length == Q.bit_length
    hamming_dist = (A.bits ^ Q.bits).bit_count()
    return 1.0 - (hamming_dist / A.bit_length)


def similarity_binarized_dot(A: BinaryVector, Q: BinaryVector) -> float:
    """
    Binarized Dot / Jaccard similarity: POPCOUNT(A AND Q) / POPCOUNT(A OR Q)
    Returns similarity in range [0.0, 1.0].
    """
    assert A.bit_length == Q.bit_length
    intersection = (A.bits & Q.bits).bit_count()
    union = (A.bits | Q.bits).bit_count()
    if union == 0:
        return 1.0
    return intersection / union


def similarity_sign_cosine(A: BinaryVector, Q: BinaryVector) -> float:
    """
    Rescaled Hamming / Sign Cosine similarity: (N - 2 * POPCOUNT(A XOR Q)) / N
    Maps [0, N] hamming distance to [+1, -1] cosine space.
    """
    assert A.bit_length == Q.bit_length
    hamming_dist = (A.bits ^ Q.bits).bit_count()
    return (A.bit_length - 2.0 * hamming_dist) / A.bit_length
