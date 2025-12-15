"""
Tests for deeplens/optics/ray.py - Ray class operations.
"""

import pytest
import torch

from deeplens.optics.ray import Ray
from deeplens.basics import DEFAULT_WAVE


class TestRayInit:
    """Test Ray initialization."""

    def test_ray_init_basic(self, device_auto):
        """Ray should initialize with origin and direction."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert ray.o.shape == (1, 3)
        assert ray.d.shape == (1, 3)
        assert ray.wvln.shape == ()  # 0D scalar tensor

    def test_ray_init_batch(self, device_auto):
        """Ray should support batch initialization."""
        batch_size = 100
        o = torch.zeros(batch_size, 3, device=device_auto)
        d = torch.zeros(batch_size, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert ray.o.shape == (batch_size, 3)
        assert ray.shape == (batch_size,)

    def test_ray_init_multidim(self, device_auto):
        """Ray should support multi-dimensional batches."""
        o = torch.zeros(5, 10, 3, device=device_auto)
        d = torch.zeros(5, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert ray.o.shape == (5, 10, 3)
        assert ray.shape == (5, 10)

    def test_ray_init_normalizes_direction(self, device_auto):
        """Ray direction should be normalized."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[3.0, 4.0, 0.0]], device=device_auto)  # Not normalized
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        norm = torch.norm(ray.d, dim=-1)
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-6)

    def test_ray_init_wavelength_validation(self, device_auto):
        """Ray should validate wavelength is in micrometers."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        
        # Valid wavelength (0.55 um = 550 nm)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        assert torch.isclose(ray.wvln, torch.tensor(0.55, device=device_auto)).item()
        
        # Wavelength out of range should raise
        with pytest.raises(AssertionError):
            Ray(o, d, wvln=550.0, device=device_auto)  # nm instead of um

    def test_ray_init_valid_mask(self, device_auto):
        """Ray should initialize with all-valid mask."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert torch.all(ray.is_valid == 1.0)

    def test_ray_init_opl_zero(self, device_auto):
        """Ray should initialize with zero optical path length."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        assert torch.all(ray.opl == 0.0)


class TestRayPropTo:
    """Test ray propagation."""

    def test_ray_prop_to_basic(self, device_auto):
        """Ray should propagate to z-plane."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.prop_to(z=10.0)
        
        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto))

    def test_ray_prop_to_angled(self, device_auto):
        """Ray should propagate correctly at an angle."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto)
        # 45 degree angle in xz plane
        d = torch.tensor([[1.0, 0.0, 1.0]], device=device_auto)
        d = d / torch.norm(d)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.prop_to(z=10.0)
        
        assert torch.allclose(ray.o[0, 0], torch.tensor(10.0, device=device_auto), atol=1e-5)
        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto), atol=1e-5)

    def test_ray_prop_to_backward(self, device_auto):
        """Ray should propagate backward."""
        o = torch.tensor([[0.0, 0.0, 10.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, -1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.prop_to(z=0.0)
        
        assert torch.allclose(ray.o[0, 2], torch.tensor(0.0, device=device_auto))

    def test_ray_prop_to_respects_valid(self, device_auto):
        """Propagation should respect valid mask."""
        o = torch.zeros(2, 3, device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.is_valid[1] = 0.0  # Invalidate second ray
        original_o = ray.o.clone()
        
        ray.prop_to(z=10.0)
        
        assert torch.allclose(ray.o[0, 2], torch.tensor(10.0, device=device_auto))
        assert torch.allclose(ray.o[1], original_o[1])  # Invalid ray unchanged

    def test_ray_prop_to_coherent_opl(self, device_auto):
        """Coherent ray should track OPL during propagation."""
        o = torch.tensor([[0.0, 0.0, 0.0]], device=device_auto, dtype=torch.float64)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto, dtype=torch.float64)
        ray = Ray(o, d, wvln=0.55, coherent=True, device=device_auto)
        
        ray.prop_to(z=10.0, n=1.5)
        
        # OPL = n * distance
        expected_opl = 1.5 * 10.0
        assert torch.allclose(ray.opl[0, 0], torch.tensor(expected_opl, device=device_auto, dtype=torch.float64))


class TestRayCentroid:
    """Test ray centroid calculation."""

    def test_ray_centroid_single(self, device_auto):
        """Centroid of single ray is the ray position."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        centroid = ray.centroid()
        
        assert torch.allclose(centroid, o.squeeze(0))

    def test_ray_centroid_batch(self, device_auto):
        """Centroid should be mean of valid rays."""
        o = torch.tensor([[0.0, 0.0, 0.0], [2.0, 4.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        centroid = ray.centroid()
        
        expected = torch.tensor([1.0, 2.0, 0.0], device=device_auto)
        assert torch.allclose(centroid, expected)

    def test_ray_centroid_respects_valid(self, device_auto):
        """Centroid should only consider valid rays."""
        o = torch.tensor([[0.0, 0.0, 0.0], [100.0, 100.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.is_valid[1] = 0.0  # Invalidate second ray
        
        centroid = ray.centroid()
        
        expected = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        assert torch.allclose(centroid, expected, atol=1e-5)


class TestRayRmsError:
    """Test RMS error calculation."""

    def test_ray_rms_error_zero(self, device_auto):
        """RMS error should be zero for coincident rays."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        rms = ray.rms_error()
        
        assert torch.allclose(rms, torch.tensor(0.0, device=device_auto), atol=1e-5)

    def test_ray_rms_error_nonzero(self, device_auto):
        """RMS error should be positive for spread rays."""
        # Rays forming a circle of radius 1
        n = 100
        theta = torch.linspace(0, 2 * 3.14159, n, device=device_auto)
        o = torch.stack([torch.cos(theta), torch.sin(theta), torch.zeros(n, device=device_auto)], dim=-1)
        d = torch.zeros(n, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        rms = ray.rms_error()
        
        # RMS of unit circle should be ~1
        assert rms > 0.9 and rms < 1.1

    def test_ray_rms_error_with_reference(self, device_auto):
        """RMS error should use provided reference center."""
        o = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]], device=device_auto)
        d = torch.zeros(2, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        center_ref = torch.tensor([0.0, 0.0, 0.0], device=device_auto)
        rms = ray.rms_error(center_ref=center_ref)
        
        # RMS from origin: sqrt((1^2 + 3^2) / 2) = sqrt(5)
        expected = torch.sqrt(torch.tensor(5.0, device=device_auto))
        assert torch.allclose(rms, expected, atol=1e-4)


class TestRayClone:
    """Test ray cloning."""

    def test_ray_clone_creates_copy(self, device_auto):
        """Clone should create independent copy."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        cloned = ray.clone()
        
        # Modify original
        ray.o[0, 0] = 999.0
        
        # Clone should be unchanged
        assert cloned.o[0, 0] != 999.0

    def test_ray_clone_to_cpu(self, device_auto):
        """Clone should allow device specification."""
        o = torch.tensor([[1.0, 2.0, 3.0]], device=device_auto)
        d = torch.tensor([[0.0, 0.0, 1.0]], device=device_auto)
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        cloned = ray.clone(device="cpu")
        
        assert cloned.o.device == torch.device("cpu")


class TestRaySqueezeUnsqueeze:
    """Test dimension manipulation."""

    def test_ray_squeeze(self, device_auto):
        """Squeeze should remove singleton dimensions."""
        o = torch.zeros(1, 10, 3, device=device_auto)
        d = torch.zeros(1, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.squeeze(dim=0)
        
        assert ray.o.shape == (10, 3)
        assert ray.d.shape == (10, 3)

    def test_ray_unsqueeze(self, device_auto):
        """Unsqueeze should add dimension."""
        o = torch.zeros(10, 3, device=device_auto)
        d = torch.zeros(10, 3, device=device_auto)
        d[:, 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ray.unsqueeze(dim=0)
        
        assert ray.o.shape == (1, 10, 3)
        assert ray.d.shape == (1, 10, 3)

    def test_ray_squeeze_unsqueeze_roundtrip(self, device_auto):
        """Squeeze then unsqueeze should restore shape."""
        o = torch.zeros(1, 10, 3, device=device_auto)
        d = torch.zeros(1, 10, 3, device=device_auto)
        d[..., 2] = 1.0
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        original_shape = ray.o.shape
        ray.squeeze(dim=0)
        ray.unsqueeze(dim=0)
        
        assert ray.o.shape == original_shape
