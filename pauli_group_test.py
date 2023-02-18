from itertools import combinations

from pauli_group import SIGMA, PauliGroupElement

I, X, Y, Z = (
    PauliGroupElement.I(),
    PauliGroupElement.X(),
    PauliGroupElement.Y(),
    PauliGroupElement.Z(),
)

elts = [I, X, Y, Z]


def test_square_to_I():
    """Test everything squares to I."""
    for g in elts:
        assert g * g == I


def test_multiplication_by_I():
    """Test I is group identity."""
    for g in elts:
        assert g * I == g
        assert I * g == g


def test_some_multiplications():
    """Test a few multiplications."""
    assert X * Y == 1j * Z
    assert Y * Z == 1j * X
    assert Z * X == 1j * Y


def test_commutation_relation():
    """Test anti-commuting."""
    for g1, g2 in combinations([X, Y, Z], 2):
        assert g1 * g2 == -g2 * g1


def test_phase_multiplication():
    """Test multiplication by a phase."""
    assert 1j * I == I * 1j == PauliGroupElement(phase=1j, sigma=SIGMA.I)
    assert 1j * X == X * 1j == PauliGroupElement(phase=1j, sigma=SIGMA.X)
    assert 1j * Y == Y * 1j == PauliGroupElement(phase=1j, sigma=SIGMA.Y)
    assert 1j * Z == Z * 1j == PauliGroupElement(phase=1j, sigma=SIGMA.Z)
