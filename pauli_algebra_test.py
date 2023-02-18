from pauli_algebra import PauliAlgebraElement, PauliTensor
from pauli_group import PauliGroupElement

I, X, Y, Z = (
    PauliGroupElement.I(),
    PauliGroupElement.X(),
    PauliGroupElement.Y(),
    PauliGroupElement.Z(),
)

elts = [I, X, Y, Z]


def test_overall_phase():
    """Test the overall_phase property."""
    for g in elts:
        assert PauliTensor(tensor=(g,)).overall_phase == 1
        assert PauliTensor(tensor=(1j * g,)).overall_phase == 1j
        assert PauliTensor(tensor=(1j * g, 1j * g)).overall_phase == -1
        assert PauliTensor(phase=1j, tensor=(1j * g, 1j * g)).overall_phase == -1j


def test_factor_phase():
    """Test factor_phase."""
    for g in elts:
        assert PauliTensor(tensor=(1j * g,)).factor_phase() == PauliTensor(
            phase=1j, tensor=(g,)
        )

    for g in elts:
        assert PauliTensor(tensor=(1j * g, 1j * g)).factor_phase() == PauliTensor(
            phase=-1, tensor=(g, g)
        )


def test_from_str():
    """Test the from_str constructor for Pauli tensors."""
    assert PauliTensor.from_str("XYZ") == PauliTensor(tensor=(X, Y, Z))
    assert PauliTensor.from_str("XXX") == PauliTensor(tensor=(X, X, X))
    assert PauliTensor.from_str("III") == PauliTensor(tensor=(I, I, I))


III = PauliTensor.from_str("III")
XXX = PauliTensor.from_str("XXX")
XYZ = PauliTensor.from_str("XYZ")


def test_tensor_multiplication():
    """Test factorwise multiplication of Pauli tensors."""
    assert XXX * XXX == III
    assert XYZ * XYZ == III
    assert (XXX * XYZ) == PauliTensor.from_str("IZY")


def test_simplify():
    """Test simplify method for algebra elements."""
    d = {1j * III: 1, 2 * XXX: 2}
    d_simple = {III: 1j, XXX: 4}
    algebra_elt = PauliAlgebraElement(d)
    simplified = PauliAlgebraElement(d_simple)
    assert algebra_elt.simplify() == simplified


def test_equal():
    """Test __eq__ works."""
    d = {1j * III: 1, 2 * XXX: 2}
    d_simple = {III: 1j, XXX: 4}
    algebra_elt = PauliAlgebraElement(d)
    simplified = PauliAlgebraElement(d_simple)
    assert algebra_elt == simplified


def test_simplify_combine_coeffs():
    """Test simplify method for algebra elements with repeated simple tensor."""
    d = {1j * III: 1, 2 * XXX: 2, 1j * XXX: 1}
    d_simple = {III: 1j, XXX: 4 + 1j}
    algebra_elt = PauliAlgebraElement(d)
    simplified = PauliAlgebraElement(d_simple)
    assert algebra_elt.simplify() == simplified


def test_sum_of_algebra_elts():
    """Test addition of algebra elements."""
    assert PauliAlgebraElement({III: 1}) + PauliAlgebraElement(
        {XXX: 1}
    ) == PauliAlgebraElement({III: 1, XXX: 1})

    assert PauliAlgebraElement({III: 1, XXX: 1}) + PauliAlgebraElement(
        {XYZ: 1, XXX: 1}
    ) == PauliAlgebraElement({III: 1, XYZ: 1, XXX: 2})


def test_scalar_multiplication_of_algebra_elts():
    """Test scalar multiplication."""
    assert 2 * PauliAlgebraElement({III: 1, XXX: 1}) == PauliAlgebraElement(
        {III: 2, XXX: 2}
    )
    assert 1j * PauliAlgebraElement({III: 1, XXX: 1}) == PauliAlgebraElement(
        {III: 1j, XXX: 1j}
    )


def test_multiplication_of_algebra_elts():
    """Test multiplication."""
    assert PauliAlgebraElement({III: 1, XXX: 1}) * PauliAlgebraElement(
        {III: 1, XXX: 1}
    ) == PauliAlgebraElement({III: 2, XXX: 2})


def test_multiplication_by_tensor():
    """Test multiplication by a simple tensor."""
    IZY = PauliTensor.from_str("IZY")
    assert PauliAlgebraElement({III: 1, XYZ: 1}) * XXX == PauliAlgebraElement(
        {IZY: 1, XXX: 1}
    )
    assert XXX * PauliAlgebraElement({III: 1, XYZ: 1}) == PauliAlgebraElement(
        {IZY: 1, XXX: 1}
    )
