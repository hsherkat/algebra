import itertools as it
from dataclasses import dataclass
from enum import Enum
from numbers import Complex
from typing import Iterable, Union

ComplexT = Union[int, float, complex]

PHASE_STR = {1: "", -1: "-", 1j: "i", -1j: "-i"}


class SIGMA(Enum):
    """Enum representing the four sigma matrices."""

    I = "I"
    X = "X"
    Y = "Y"
    Z = "Z"

    def __str__(self):
        return f"\u03C3_{self.value}"


SIGMA_LIST = list(SIGMA)


def sign(permutation: Iterable[int]) -> int:
    """Calculates the sign of a permutation by counting inversions."""
    inversions = sum(x > y for x, y in it.combinations(permutation, 2))
    return (-1) ** inversions


def create_times_table() -> dict[tuple[int, int], tuple[ComplexT, int]]:
    """Creates a helper dictionary for computing products in the Pauli group.
    Makes a times table based on the subscript of the sigma.
    """
    table: dict[tuple[int, int], tuple[ComplexT, int]] = dict()
    phase: ComplexT
    for idx1, idx2 in it.product(range(4), repeat=2):
        if 0 in (idx1, idx2):
            phase = 1
            sigma_idx = idx1 + idx2
        elif idx1 == idx2:
            phase = 1
            sigma_idx = 0
        else:
            idx3 = 6 - idx1 - idx2
            phase = 1j * sign((idx1, idx2, idx3))
            sigma_idx = idx3
        table[(idx1, idx2)] = (phase, sigma_idx)
    return table


TIMES_TABLE = create_times_table()


@dataclass(frozen=True)
class PauliGroupElement:
    """Class for symbolically multiplying elements of the Pauli group.
    Pauli group elements are a sigma matrix with a phase that's a fourth root of unity.
    """

    phase: complex = 1
    sigma: SIGMA = SIGMA.I

    def __post_init__(self):
        if self.phase not in (1, -1, 1j, -1j):
            raise ValueError("Phase must be a fourth root of unity.")

    def __str__(self):
        return f"{PHASE_STR[self.phase]}{self.sigma!s}"

    def __mul__(self, other) -> "PauliGroupElement":
        """Multiply non-identity group elements according to the relation:
        sigma_j * sigma_k = i e_jkl sigma_l
        where e_jkl is the Levi-Civita symbol.
        """
        if isinstance(other, Complex):
            return PauliGroupElement(self.phase * other, self.sigma)

        if isinstance(other, PauliGroupElement):
            combined_phase = self.phase * other.phase
            self_idx, other_idx = (
                SIGMA_LIST.index(self.sigma),
                SIGMA_LIST.index(other.sigma),
            )
            phase, sigma_idx = TIMES_TABLE[(self_idx, other_idx)]
            return PauliGroupElement(combined_phase * phase, SIGMA_LIST[sigma_idx])

        return NotImplemented

    def __rmul__(self, other) -> "PauliGroupElement":
        if isinstance(other, Complex):
            return self * other
            # return PauliGroupElement(self.phase * other, self.sigma)

        return NotImplemented

    def __neg__(self):
        return -1 * self

    @classmethod
    def I(cls):
        """Convenient constructor for I."""
        return cls(1, SIGMA.I)

    @classmethod
    def X(cls):
        """Convenient constructor for X."""
        return cls(1, SIGMA.X)

    @classmethod
    def Y(cls):
        """Convenient constructor for Y."""
        return cls(1, SIGMA.Y)

    @classmethod
    def Z(cls):
        """Convenient constructor for Z."""
        return cls(1, SIGMA.Z)
