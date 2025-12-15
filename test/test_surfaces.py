"""
Tests for deeplens/optics/geometric_surface/ - Geometric surface classes.
"""

import pytest
import torch
import math

from deeplens.optics.geometric_surface import Spheric, Aspheric, Aperture, Plane
from deeplens.optics.geometric_surface.base import Surface
from deeplens.optics.ray import Ray


class TestSphericSurface:
    """Test Spheric surface class."""

    def test_spheric_init(self, device_auto):
        """Spheric surface should initialize with curvature."""
        surf = Spheric(
            c=0.1,  # curvature = 1/radius
            r=5.0,  # aperture radius
            d=0.0,  # distance from origin
            mat2="bk7",
            device=device_auto,
        )
        
        assert surf.c.item() == pytest.approx(0.1)
        assert surf.r == 5.0

    def test_spheric_sag_center(self, device_auto):
        """Sag at center should be zero."""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([0.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        z = surf.sag(x, y)
        
        assert torch.allclose(z, torch.tensor([0.0], device=device_auto), atol=1e-6)

    def test_spheric_sag_offaxis(self, device_auto):
        """Sag should increase with distance from axis."""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x1 = torch.tensor([1.0], device=device_auto)
        x2 = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        z1 = surf.sag(x1, y)
        z2 = surf.sag(x2, y)
        
        # For positive curvature, sag grows with radius
        assert z2.abs() > z1.abs()

    def test_spheric_sag_symmetry(self, device_auto):
        """Sag should be symmetric about optical axis."""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x_pos = torch.tensor([2.0], device=device_auto)
        x_neg = torch.tensor([-2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        z_pos = surf.sag(x_pos, y)
        z_neg = surf.sag(x_neg, y)
        
        assert torch.allclose(z_pos, z_neg)

    def test_spheric_intersect(self, device_auto):
        """Ray should intersect spheric surface."""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        # Use ray_reaction which handles coordinate transforms
        n1 = 1.0  # air
        n2 = surf.mat2.ior(torch.tensor([0.55], device=device_auto)).item()
        ray = surf.ray_reaction(ray, n1, n2)
        
        # Ray should hit the surface near z=10
        assert ray.o[0, 2].item() > 9.0
        assert ray.o[0, 2].item() < 11.0

    def test_spheric_refract(self, device_auto):
        """Ray should refract at spheric surface."""
        surf = Spheric(c=0.05, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        # Off-axis ray
        o = torch.tensor([[1.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        original_d = ray.d.clone()
        
        # Get refractive indices
        n1 = 1.0  # air
        n2 = surf.mat2.ior(torch.tensor([0.55], device=device_auto)).item()
        
        ray = surf.ray_reaction(ray, n1, n2)
        
        # Direction should change due to refraction at curved surface
        assert not torch.allclose(ray.d, original_d, atol=1e-3)

    def test_spheric_init_from_dict(self, device_auto):
        """Spheric should initialize from dictionary."""
        surf_dict = {
            "type": "Spheric",
            "c": 0.05,
            "r": 5.0,
            "d": 10.0,
            "mat2": "bk7",
        }
        
        surf = Spheric.init_from_dict(surf_dict)
        
        assert surf.c.item() == pytest.approx(0.05)
        assert surf.r == 5.0

    def test_spheric_surf_dict(self, device_auto):
        """Spheric should export to dictionary."""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        d = surf.surf_dict()
        
        assert d["type"] == "Spheric"
        assert "(c)" in d or "c" in d
        assert d["r"] == 5.0


class TestAsphericSurface:
    """Test Aspheric surface class."""

    def test_aspheric_init(self, device_auto):
        """Aspheric surface should initialize with coefficients."""
        surf = Aspheric(
            c=0.1,
            k=0.0,  # conic constant
            ai=[0.0] * 6,  # higher-order coefficients
            r=5.0,
            d=0.0,
            mat2="bk7",
            device=device_auto,
        )
        
        assert surf.c.item() == pytest.approx(0.1)
        assert len(surf.ai) == 6

    def test_aspheric_reduces_to_spheric(self, device_auto):
        """Aspheric with k=0 and ai=0 should equal spheric."""
        c = 0.05
        r = 5.0
        
        asph = Aspheric(c=c, k=0.0, ai=[0.0]*6, r=r, d=0.0, mat2="bk7", device=device_auto)
        sph = Spheric(c=c, r=r, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([1.0, 2.0, 3.0], device=device_auto)
        y = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        
        z_asph = asph.sag(x, y)
        z_sph = sph.sag(x, y)
        
        assert torch.allclose(z_asph, z_sph, atol=1e-5)

    def test_aspheric_conic_parabola(self, device_auto):
        """k=-1 should give parabolic sag z = c*r^2 / 2."""
        c = 0.1
        surf = Aspheric(c=c, k=-1.0, ai=[0.0]*6, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        r_sq = x**2 + y**2
        
        z = surf.sag(x, y)
        expected = c * r_sq / 2  # Parabolic formula
        
        assert torch.allclose(z, expected, atol=1e-5)

    def test_aspheric_higher_order(self, device_auto):
        """Higher-order coefficients should affect sag."""
        c = 0.0  # No base curvature
        ai = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0]  # Only ai2
        
        surf = Aspheric(c=c, k=0.0, ai=ai, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        r_sq = x**2 + y**2
        
        z = surf.sag(x, y)
        expected = ai[0] * r_sq  # ai2 * r^2
        
        assert torch.allclose(z, expected, atol=1e-5)

    def test_aspheric_init_from_dict(self, device_auto):
        """Aspheric should initialize from dictionary."""
        surf_dict = {
            "type": "Aspheric",
            "c": 0.05,
            "k": -0.5,
            "ai": [0.001, 0.0001, 0.0, 0.0, 0.0, 0.0],
            "r": 5.0,
            "d": 10.0,
            "mat2": "pmma",
        }
        
        surf = Aspheric.init_from_dict(surf_dict)
        
        assert surf.c.item() == pytest.approx(0.05)
        assert surf.k.item() == pytest.approx(-0.5)


class TestApertureSurface:
    """Test Aperture surface class."""

    def test_aperture_init(self, device_auto):
        """Aperture should initialize with radius."""
        aper = Aperture(r=2.0, d=5.0, device=device_auto)
        
        assert aper.r == 2.0
        assert aper.d.item() == pytest.approx(5.0)

    def test_aperture_clips_rays(self, device_auto):
        """Aperture should invalidate rays outside radius."""
        aper = Aperture(r=2.0, d=5.0, device=device_auto)
        
        # Ray inside aperture
        o_in = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray_in = Ray(o_in, d, wvln=0.55, device=device_auto)
        
        # Ray outside aperture
        o_out = torch.tensor([[5.0, 0.0, 0.0]], device=device_auto)
        ray_out = Ray(o_out, d.clone(), wvln=0.55, device=device_auto)
        
        ray_in = aper.ray_reaction(ray_in)
        ray_out = aper.ray_reaction(ray_out)
        
        assert ray_in.is_valid[0].item() == 1.0
        assert ray_out.is_valid[0].item() == 0.0

    def test_aperture_surf_dict(self, device_auto):
        """Aperture should export to dictionary."""
        aper = Aperture(r=2.0, d=5.0, device=device_auto)
        
        d = aper.surf_dict()
        
        assert d["type"] == "Aperture"
        assert d["r"] == 2.0


class TestPlaneSurface:
    """Test Plane surface class."""

    def test_plane_init(self, device_auto):
        """Plane surface should initialize."""
        plane = Plane(r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        assert plane.r == 5.0
        assert plane.d.item() == pytest.approx(10.0)

    def test_plane_sag_zero(self, device_auto):
        """Plane sag should be zero everywhere."""
        plane = Plane(r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device_auto)
        y = torch.tensor([0.0, 1.0, 2.0, 3.0], device=device_auto)
        
        z = plane.sag(x, y)
        
        assert torch.allclose(z, torch.zeros_like(z))

    def test_plane_intersect(self, device_auto):
        """Ray should intersect plane at z=d."""
        plane = Plane(r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        # Use ray_reaction which handles coordinate transforms
        n1 = 1.0
        n2 = plane.mat2.ior(torch.tensor([0.55], device=device_auto)).item()
        ray = plane.ray_reaction(ray, n1, n2)
        
        assert ray.o[0, 2].item() == pytest.approx(10.0, abs=0.1)


class TestSurfaceBase:
    """Test Surface base class functionality."""

    def test_surface_normal_vec(self, device_auto):
        """Normal vector should point toward light source."""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray = surf.intersect(ray)
        n = surf.normal_vec(ray)
        
        # At center, normal should point in -z direction (toward light)
        assert n[0, 2].item() < 0

    def test_surface_reflect(self, device_auto):
        """Reflection should obey law of reflection."""
        surf = Spheric(c=0.0, r=5.0, d=10.0, mat2="bk7", device=device_auto)  # Flat mirror
        
        # 45 degree incidence
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[1.0, 0.0, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray = surf.intersect(ray)
        ray = surf.reflect(ray)
        
        # Reflected ray should go in x, -z direction
        assert ray.d[0, 0].item() > 0  # +x
        assert ray.d[0, 2].item() < 0  # -z

    def test_surface_local_coord_transform(self, device_auto):
        """Local coordinate transform should be invertible."""
        surf = Spheric(c=0.1, r=5.0, d=10.0, mat2="bk7", device=device_auto)
        
        o = torch.tensor([[1.0, 2.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.1, 0.2, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        original_o = ray.o.clone()
        original_d = ray.d.clone()
        
        ray = surf.to_local_coord(ray)
        ray = surf.to_global_coord(ray)
        
        assert torch.allclose(ray.o, original_o, atol=1e-5)
        assert torch.allclose(ray.d, original_d, atol=1e-5)


class TestSurfaceDerivatives:
    """Test surface derivative calculations."""

    def test_spheric_dfdxy_center(self, device_auto):
        """Derivatives at center should be zero."""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([0.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        dfdx, dfdy = surf._dfdxy(x, y)
        
        assert torch.allclose(dfdx, torch.tensor([0.0], device=device_auto), atol=1e-5)
        assert torch.allclose(dfdy, torch.tensor([0.0], device=device_auto), atol=1e-5)

    def test_spheric_dfdxy_symmetry(self, device_auto):
        """Derivatives should have appropriate symmetry."""
        surf = Spheric(c=0.1, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([2.0], device=device_auto)
        y = torch.tensor([0.0], device=device_auto)
        
        dfdx1, dfdy1 = surf._dfdxy(x, y)
        dfdx2, dfdy2 = surf._dfdxy(-x, y)
        
        # dfdx should be antisymmetric
        assert torch.allclose(dfdx1, -dfdx2, atol=1e-5)

    def test_aspheric_dfdxy(self, device_auto):
        """Aspheric derivatives should be consistent with numerical gradient."""
        surf = Aspheric(c=0.05, k=-0.5, ai=[0.001]*6, r=5.0, d=0.0, mat2="bk7", device=device_auto)
        
        x = torch.tensor([1.5], device=device_auto, requires_grad=True)
        y = torch.tensor([0.5], device=device_auto, requires_grad=True)
        
        z = surf.sag(x, y)
        z.backward()
        
        dfdx_num = x.grad
        dfdy_num = y.grad
        
        x_detach = x.detach()
        y_detach = y.detach()
        dfdx_ana, dfdy_ana = surf._dfdxy(x_detach, y_detach)
        
        assert torch.allclose(dfdx_num, dfdx_ana, atol=1e-4)
        assert torch.allclose(dfdy_num, dfdy_ana, atol=1e-4)
