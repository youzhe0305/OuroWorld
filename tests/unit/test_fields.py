import pytest
import torch

from ouroworld.fields.cinemagraph import ViewContext
from ouroworld.fields.encoding import Triplane, TriplaneSpec, fourier_time_basis
from tests.helpers import small_model


def test_fourier_basis_is_periodic_only_with_period_one() -> None:
    time = torch.linspace(0, 1, 7).unsqueeze(1)
    periodic = fourier_time_basis(time, harmonics=3, period=1.0)
    assert torch.allclose(periodic[0], periodic[-1], atol=1e-6)
    aperiodic = fourier_time_basis(time, harmonics=3, period=2.0)
    assert not torch.allclose(aperiodic[0], aperiodic[-1], atol=1e-3)
    assert periodic.shape == aperiodic.shape == (7, 7)


def test_mirrored_coordinates_equal_flipped_planes() -> None:
    """Mirroring the coordinates equals flipping every plane."""
    torch.manual_seed(0)
    grid = Triplane(TriplaneSpec((5, 6, 7), (1, 2), 3), basis_size=2)
    points = torch.rand(50, 3) * 2 - 1
    mirrored = grid.coefficients(-points)
    with torch.no_grad():
        for plane in grid.planes():
            plane.copy_(torch.flip(plane, dims=(-2, -1)))
    assert torch.allclose(grid.coefficients(points), mirrored, atol=1e-6)


def test_periodic_field_is_identity_at_zero_and_periodic() -> None:
    model = small_model(with_drift=False)
    points = model.canonical.xyz.detach()
    at = lambda t: model.periodic(points, points.new_full((len(points), 1), t))  # noqa: E731
    zero, one, two, mid = at(0.0), at(1.0), at(2.0), at(0.3)
    assert torch.count_nonzero(zero.twist) == 0 and torch.count_nonzero(zero.sh_delta) == 0
    assert torch.count_nonzero(zero.log_scale_delta) == 0
    assert torch.allclose(one.twist, zero.twist, atol=1e-5) and torch.allclose(
        two.twist, zero.twist, atol=1e-5
    )
    assert mid.twist.abs().max() > 1e-4  # the perturbed heads make P non-trivial


def test_scale_change_is_bounded() -> None:
    model = small_model(with_drift=False)
    with torch.no_grad():
        for parameter in model.periodic.scale_head.parameters():
            parameter.mul_(1e4)
    points = model.canonical.xyz.detach()
    delta = model.periodic(points, points.new_full((len(points), 1), 0.4)).log_scale_delta
    assert delta.abs().max() <= model.periodic.spec.max_log_scale_delta + 1e-6


def test_drift_is_grounded_on_reference_view_and_at_t0() -> None:
    model = small_model()
    points = model.canonical.xyz.detach()
    count = len(points)
    views = torch.tensor([0, 1, 2, 3]).repeat_interleave(count)
    repeated = points.repeat(4, 1)
    for time, expect_zero_views in ((0.0, {0, 1, 2, 3}), (0.4, {1})):
        outputs = model.drift(repeated, repeated.new_full((4 * count, 1), time), views)
        for view in range(4):
            rows = views == view
            magnitude = sum(value[rows].abs().sum() for value in outputs.values())
            assert (magnitude == 0) == (view in expect_zero_views), (time, view)


def test_ungrounded_drift_acts_everywhere() -> None:
    model = small_model(zero_reference_view=False, zero_t0=False)
    assert model.applies_drift(ViewContext(0.0, 1))
    assert not small_model().applies_drift(ViewContext(0.0, 2))


def test_inference_context_ignores_drift() -> None:
    model = small_model()
    with torch.no_grad():
        with_view = model.deformed(ViewContext(0.4, 1))  # reference view: grounded
        without_view = model.deformed(ViewContext(0.4))
        drifted = model.deformed(ViewContext(0.4, 2))
    assert torch.equal(with_view.xyz, without_view.xyz)
    assert not torch.allclose(drifted.xyz, without_view.xyz)


@pytest.mark.parametrize("heads", [("pos",), ("rot",), ("scale", "opacity")])
def test_drift_heads_only_touch_their_attribute(heads: tuple[str, ...]) -> None:
    model = small_model(heads=heads)
    with torch.no_grad():
        base = model.deformed(ViewContext(0.4))
        drifted = model.deformed(ViewContext(0.4, 2))
    changed = {
        name
        for name in ("xyz", "rotation", "log_scale", "opacity_logit", "sh")
        if not torch.equal(getattr(base, name), getattr(drifted, name))
    }
    expected = {"pos": "xyz", "rot": "rotation", "scale": "log_scale", "opacity": "opacity_logit"}
    assert changed == {expected[head] for head in heads}
