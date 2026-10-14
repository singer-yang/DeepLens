"""Regression tests for aperture entry and sizing without surviving rays."""

import math

import pytest
import torch

from deeplens import GeoLens
from deeplens.geometric_surface import Aspheric, Plane
from deeplens.geometric_surface.base_surface import Surface
from deeplens.light import Ray


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("square", [False, True])
def test_aperture_entry_rejects_backside_and_preserves_front(dtype, reverse, square):
    """The synthetic sag -r^4 has a later root after an unphysical rim entry."""
    sign = -1 if reverse else 1
    surface = Aspheric(
        r=math.sqrt(2) if square else 1.0,
        d_next=0.0,
        c=0.0,
        k=0.0,
        ai=[-sign],
        mat2="air",
        is_square=square,
    ).astype(dtype)
    origins = torch.tensor([[2.0, 0.0, -1.8], [2.0, 0.0, -3.0]], dtype=dtype)
    directions = torch.tensor([[-1.0, 0.0, 1.0], [-1.5, 0.0, 2.9375]], dtype=dtype)
    origins[:, 2] *= sign
    directions[:, 2] *= sign
    ray = Ray(origins, directions, wvln=0.55)

    # The original solver accepts the later in-aperture root of the blocked ray.
    _, original_valid = Surface.newtons_method(surface, ray)
    assert original_valid.all()
    t, valid = surface.newtons_method(ray)
    assert valid.tolist() == [False, True]
    point = ray.o + ray.d * t.unsqueeze(-1)
    torch.testing.assert_close(
        point[1], point.new_tensor([0.5, 0.0, -sign / 16]), atol=2e-5, rtol=0
    )
    hit = surface.intersect(ray.clone())
    assert hit.is_valid.tolist() == [0.0, 1.0]
    torch.testing.assert_close(hit.o[0], origins[0])


@pytest.mark.parametrize("square", [False, True])
def test_entry_mask_handles_axial_inside_missing_and_scalar_rays(square):
    surface = Aspheric(
        r=1.0, d_next=0.0, c=0.0, k=0.0, ai=[-1.0], mat2="air", is_square=square
    ).astype(torch.float64)
    origins = torch.tensor(
        [[[0.0, 0.0, -1.0], [2.0, 0.0, -1.0], [2.0, 2.0, -1.0]]],
        dtype=torch.float64,
    )
    directions = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]])
    ray = Ray(origins, directions, wvln=0.55)
    assert not surface._enters_through_edge(ray).any()
    _, valid = surface.newtons_method(ray)
    assert valid.tolist() == [[True, False, False]]
    scalar = Ray(origins[0, 0], directions[0, 0], wvln=0.55)
    assert surface.newtons_method(scalar)[1].item()


def test_edge_check_preserves_valid_intersection_gradients():
    surface = Aspheric(r=1.0, d_next=0.0, c=0.0, k=0.0, ai=[-0.1], mat2="air")
    surface.astype(torch.float64)
    surface.ai4.requires_grad_(True)
    ray = Ray(
        torch.tensor([[0.3, 0.0, -1.0]], dtype=torch.float64),
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        wvln=0.55,
    )
    t, valid = surface.newtons_method(ray)
    assert valid.item()
    t.sum().backward()
    assert surface.ai4.grad.item() == pytest.approx(0.3**4, abs=1e-10)


@pytest.mark.parametrize("reverse", [False, True])
def test_entry_uses_conic_domain_when_it_is_smaller_than_aperture(reverse):
    sign = -1 if reverse else 1
    surface = Aspheric(
        r=3.0, d_next=0.0, c=sign * 0.5, k=3.0, ai=[-sign], mat2="air"
    ).astype(torch.float64)
    # The real conic domain ends at r=1, where sag tends to -sign/2.
    origins = torch.tensor([[2.0, 0.0, -0.11], [2.0, 0.0, -1.0]], dtype=torch.float64)
    directions = torch.tensor([[-1.0, 0.0, 0.01]] * 2, dtype=torch.float64)
    origins[:, 2] *= sign
    directions[:, 2] *= sign
    ray = Ray(origins, directions, wvln=0.55)
    assert surface._enters_through_edge(ray).tolist() == [True, False]


@pytest.mark.parametrize("mounting_margin", [None, 0.0, 0.1])
@pytest.mark.parametrize("missing", [float("nan"), float("inf")])
def test_prune_preserves_radii_without_survivors(monkeypatch, mounting_margin, missing):
    lens = GeoLens(device="cpu")
    lens.rfov = 0.2
    lens.surfaces = [
        Plane(r=0.7, d_next=1.0, mat2="air"),
        Plane(r=0.6, d_next=1.0, mat2="air"),
    ]
    records = [
        torch.tensor([[0.0, 0.0, -1.0]]),
        torch.tensor([[20.0, 0.0, 0.0]]),
        torch.tensor([[10.0, 0.0, 1.0]]),
        torch.full((1, 3), missing),
    ]
    monkeypatch.setattr(lens, "sample_from_fov", lambda **kwargs: object())
    monkeypatch.setattr(lens, "trace2sensor", lambda **kwargs: (None, records))
    for _ in range(3):
        lens.prune_surf(mounting_margin=mounting_margin)
        assert [surface.r for surface in lens.surfaces] == pytest.approx([0.7, 0.6])


def test_prune_uses_only_sensor_reaching_rays(monkeypatch):
    lens = GeoLens(device="cpu")
    lens.rfov = 0.2
    lens.surfaces = [Plane(r=0.7, d_next=1.0, mat2="air")]
    records = [
        torch.zeros((2, 3)),
        torch.tensor([[0.5, 0.0, 0.0], [20.0, 0.0, 0.0]]),
        torch.tensor([[0.5, 0.0, 1.0], [float("nan")] * 3]),
    ]
    monkeypatch.setattr(lens, "sample_from_fov", lambda **kwargs: object())
    monkeypatch.setattr(lens, "trace2sensor", lambda **kwargs: (None, records))
    lens.prune_surf(mounting_margin=0.1)
    assert lens.surfaces[0].r == pytest.approx(0.6)
