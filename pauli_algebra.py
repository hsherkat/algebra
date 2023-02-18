import itertools as it
import math
from collections import defaultdict
from dataclasses import dataclass
from numbers import Complex

from pauli_group import SIGMA, SIGMA_LIST, PauliGroupElement


@dataclass(frozen=True)
class PauliTensor:
    """Class representing tensors of Pauli group elements.
    These are the tensor products of linear operators, and thus multiply factorwise.
    """

    phase: complex = 1
    tensor: tuple[PauliGroupElement, ...] = (PauliGroupElement(1, SIGMA.I),)

    def __str__(self):
        s_tensor = " \u2297 ".join([str(factor) for factor in self.tensor])
        return f"{self.phase}({s_tensor})"

    @property
    def dim(self) -> int:
        """The dimension of the tensor."""
        return len(self.tensor)

    @property
    def overall_phase(self) -> complex:
        """Helper for computing phase of products."""
        return self.phase * math.prod(factor.phase for factor in self.tensor)

    def factor_phase(self) -> "PauliTensor":
        """Returns a tensor with no phase on any of its factors by factoring it out
        into an overall phase."""
        phaseless = [factor * (1 / factor.phase) for factor in self.tensor]
        return PauliTensor(self.overall_phase, tuple(phaseless))

    def __eq__(self, other) -> bool:
        if isinstance(other, PauliTensor):
            s_factored, o_factored = self.factor_phase(), other.factor_phase()
            return (
                s_factored.phase == o_factored.phase
                and s_factored.tensor == o_factored.tensor
            )

        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Complex):
            return PauliTensor(self.phase * other, self.tensor)

        if isinstance(other, PauliTensor):
            if self.dim != other.dim:
                raise ValueError("Tensors must have the same number of factors.")

            s_simple, o_simple = self.factor_phase(), other.factor_phase()
            phase = s_simple.phase * o_simple.phase
            tensor = PauliTensor(
                tensor=tuple(
                    s_factor * o_factor
                    for (s_factor, o_factor) in zip(s_simple.tensor, o_simple.tensor)
                ),
            )
            return phase * tensor.factor_phase()

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Complex):
            return PauliTensor(self.phase * other, self.tensor)

        return NotImplemented

    def __neg__(self):
        return -1 * self

    @classmethod
    def from_str(cls, string) -> "PauliTensor":
        """Convenience function for making simple tensors.
        Characters of input must be in {I, X, Y, Z}.
        """
        return cls(
            phase=1, tensor=tuple([PauliGroupElement(1, SIGMA[c]) for c in string])
        )


@dataclass(frozen=True)
class PauliAlgebraElement:
    """Class for manipulation of the group algebra of Pauli tensors over the complex numbers.
    Note: there is no validation preventing one from mixing tensors of different dimension.
    """

    coeffs: dict[PauliTensor, complex]

    @classmethod
    def from_str(cls, string) -> "PauliAlgebraElement":
        """Convenience function for making an algebra element with a single simple tensors.
        Characters of input must be in {I, X, Y, Z}.
        """
        tensor = PauliTensor.from_str(string)
        return PauliAlgebraElement({tensor: 1})

    def simplify(self) -> "PauliAlgebraElement":
        """Simplifies by making keys in coeffs simple tensors, moving phases to coefficient.
        Not in place: returns a new element.
        """
        new_coeffs: dict[PauliTensor, complex] = defaultdict(complex)
        for tensor, coeff in self.coeffs.items():
            factored_tensor = tensor.factor_phase()
            factored_phase = factored_tensor.phase
            new_coeffs[factored_tensor * (1 / factored_phase)] += coeff * factored_phase
        return PauliAlgebraElement(new_coeffs)

    @staticmethod
    def _str_for_lexicographic(tensor: PauliTensor) -> str:
        """Helper for __str__ so that tensors are sorted.
        Returns the string of indices of the sigma factors.
        """
        return "".join(str(SIGMA_LIST.index(factor.sigma)) for factor in tensor.tensor)

    def __str__(self):
        """Combines coefficient and tensor phase, sorts tensors."""
        return " + ".join(
            [
                f"{coeff*tensor.phase}{str(tensor)[(str(tensor).index('(')) :]}"
                for tensor, coeff in sorted(
                    self.coeffs.items(),
                    key=lambda x: PauliAlgebraElement._str_for_lexicographic(x[0]),
                )
            ]
        )

    def __eq__(self, other):
        if isinstance(other, PauliAlgebraElement):
            return self.simplify().coeffs == other.simplify().coeffs
        return NotImplemented

    def __add__(self, other):
        if isinstance(other, PauliAlgebraElement):
            sum_coeffs = defaultdict(complex)
            for coeffs in (self.coeffs, other.coeffs):
                for tensor, coeff in coeffs.items():
                    sum_coeffs[tensor] += coeff
            return PauliAlgebraElement(sum_coeffs)
        return NotImplemented

    def __mul__(self, other):
        if isinstance(other, Complex):
            return PauliAlgebraElement(
                {tensor: other * coeff for tensor, coeff in self.coeffs.items()}
            )

        if isinstance(other, PauliAlgebraElement):
            product_coeffs = defaultdict(int)
            for (s_tensor, s_coeff), (o_tensor, o_coeff) in it.product(
                self.coeffs.items(), other.coeffs.items()
            ):
                product_coeffs[s_tensor * o_tensor] += s_coeff * o_coeff
            return PauliAlgebraElement(product_coeffs)

        if isinstance(other, PauliTensor):
            other_algebra_elt = PauliAlgebraElement({other: 1})
            return self * other_algebra_elt

        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, Complex):
            return self * other

        if isinstance(other, PauliTensor):
            return self * other

        return NotImplemented

    def __neg__(self):
        return -1 * self
