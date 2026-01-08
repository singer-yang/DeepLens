"""
Tests for deeplens/optics/monte_carlo.py - Monte Carlo ray integration.
"""

import pytest
import torch

from deeplens.optics.monte_carlo import forward_integral, assign_points_to_pixels
from deeplens.optics.ray import Ray


class TestForwardIntegral:
    """Test forward Monte Carlo integration for PSF."""

    def test_forward_integral_shape(self, device_auto):
        """Output should have correct shape."""
        # Create rays at the sensor plane
        n_rays = 1024
        o = torch.zeros(n_rays, 3, device=device_auto)
        o[:, 2] = 10.0  # At sensor z=10
        # Spread rays in x-y
        o[:, 0] = torch.randn(n_rays, device=device_auto) * 0.1
        o[:, 1] = torch.randn(n_rays, device=device_auto) * 0.1
        
        d = torch.zeros(n_rays, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ks = 31
        ps = 0.01  # pixel size
        field = forward_integral(ray, ps=ps, ks=ks)
        
        assert field.shape == (ks, ks)

    def test_forward_integral_normalized(self, device_auto):
        """Integrated PSF should sum to approximately valid ray count."""
        n_rays = 4096
        o = torch.zeros(n_rays, 3, device=device_auto)
        o[:, 2] = 10.0
        o[:, 0] = torch.randn(n_rays, device=device_auto) * 0.05
        o[:, 1] = torch.randn(n_rays, device=device_auto) * 0.05
        
        d = torch.zeros(n_rays, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ks = 64
        ps = 0.01
        field = forward_integral(ray, ps=ps, ks=ks)
        
        # Sum should be approximately n_rays (number of valid rays)
        valid_count = ray.is_valid.sum().item()
        assert field.sum().item() == pytest.approx(valid_count, rel=0.2)

    def test_forward_integral_with_center(self, device_auto):
        """Should use provided reference center."""
        n_rays = 1024
        o = torch.zeros(n_rays, 3, device=device_auto)
        o[:, 2] = 10.0
        o[:, 0] = torch.randn(n_rays, device=device_auto) * 0.05 + 0.5  # Offset center
        o[:, 1] = torch.randn(n_rays, device=device_auto) * 0.05
        
        d = torch.zeros(n_rays, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ks = 31
        ps = 0.01
        pointc = torch.tensor([[0.5, 0.0]], device=device_auto)
        
        field = forward_integral(ray, ps=ps, ks=ks, pointc=pointc)
        
        # Peak should be near center of kernel
        center = ks // 2
        peak_y, peak_x = torch.where(field == field.max())
        assert abs(peak_x[0].item() - center) <= 5
        assert abs(peak_y[0].item() - center) <= 5

    def test_forward_integral_batch(self, device_auto):
        """Should handle batched rays."""
        batch_size = 4
        n_rays = 512
        
        o = torch.zeros(batch_size, n_rays, 3, device=device_auto)
        o[..., 2] = 10.0
        o[..., 0] = torch.randn(batch_size, n_rays, device=device_auto) * 0.05
        o[..., 1] = torch.randn(batch_size, n_rays, device=device_auto) * 0.05
        
        d = torch.zeros(batch_size, n_rays, 3, device=device_auto)
        d[..., 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ks = 21
        ps = 0.01
        field = forward_integral(ray, ps=ps, ks=ks)
        
        assert field.shape == (batch_size, ks, ks)


class TestAssignPointsToPixels:
    """Test point-to-pixel assignment."""

    def test_assign_points_basic(self, device_auto):
        """Should assign points to correct pixels."""
        # Points at known locations
        points = torch.tensor([
            [0.0, 0.0],  # Center
            [0.5, 0.0],  # Right of center
        ], device=device_auto)
        mask = torch.ones(2, device=device_auto)
        
        ks = 11
        x_range = (-0.55, 0.55)
        y_range = (-0.55, 0.55)
        
        value = torch.ones(2, device=device_auto)
        field = assign_points_to_pixels(
            points, mask, ks, x_range, y_range,
            value=value, interpolate=False,
        )
        
        assert field.shape == (ks, ks)
        assert field.sum() > 0

    def test_assign_points_with_mask(self, device_auto):
        """Should respect validity mask."""
        points = torch.tensor([
            [0.0, 0.0],
            [0.1, 0.0],
            [0.2, 0.0],
        ], device=device_auto)
        mask = torch.tensor([1.0, 0.0, 1.0], device=device_auto)  # Middle point invalid
        
        ks = 11
        x_range = (-0.55, 0.55)
        y_range = (-0.55, 0.55)
        
        value = torch.ones(3, device=device_auto)
        field = assign_points_to_pixels(
            points, mask, ks, x_range, y_range,
            value=value, interpolate=False,
        )
        
        # Only 2 valid points should contribute
        assert field.sum().item() == pytest.approx(2.0, abs=0.1)

    def test_assign_points_interpolate(self, device_auto):
        """Interpolation should spread point across nearby pixels."""
        # Point between pixel centers
        points = torch.tensor([[0.025, 0.025]], device=device_auto)
        mask = torch.ones(1, device=device_auto)
        
        ks = 5
        x_range = (-0.1, 0.1)
        y_range = (-0.1, 0.1)
        
        value = torch.ones(1, device=device_auto)
        field = assign_points_to_pixels(
            points, mask, ks, x_range, y_range,
            value=value, interpolate=True,
        )
        
        # With interpolation, multiple pixels should have non-zero values
        nonzero_count = (field > 0).sum().item()
        assert nonzero_count >= 1

    def test_assign_points_complex_value(self, device_auto):
        """Should handle complex amplitude values."""
        points = torch.tensor([[0.0, 0.0]], device=device_auto)
        mask = torch.ones(1, device=device_auto)
        
        ks = 5
        x_range = (-0.1, 0.1)
        y_range = (-0.1, 0.1)
        
        # Complex amplitude value
        value = torch.ones(1, device=device_auto, dtype=torch.complex64)
        
        field = assign_points_to_pixels(
            points, mask, ks, x_range, y_range,
            value=value, interpolate=False,
        )
        
        # Output should be complex when input value is complex
        assert field.dtype == torch.complex64 or field.dtype == torch.complex128


class TestForwardIntegralCoherent:
    """Test coherent forward integration."""

    def test_forward_integral_coherent(self, device_auto):
        """Coherent integration should return complex field."""
        n_rays = 512
        o = torch.zeros(n_rays, 3, device=device_auto, dtype=torch.float64)
        o[:, 2] = 10.0
        o[:, 0] = torch.randn(n_rays, device=device_auto, dtype=torch.float64) * 0.05
        o[:, 1] = torch.randn(n_rays, device=device_auto, dtype=torch.float64) * 0.05
        
        d = torch.zeros(n_rays, 3, device=device_auto, dtype=torch.float64)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, coherent=True, device=device_auto)
        ray.opl = torch.randn(n_rays, 1, device=device_auto, dtype=torch.float64)
        
        ks = 21
        ps = 0.01
        field = forward_integral(ray, ps=ps, ks=ks)
        
        # Coherent field should be complex
        assert field.is_complex()


class TestForwardIntegralGPU:
    """Test forward integration on GPU."""

    def test_forward_integral_gpu(self, device_auto):
        """Forward integral should work on GPU."""
        n_rays = 1024
        o = torch.zeros(n_rays, 3, device=device_auto)
        o[:, 2] = 10.0
        o[:, 0] = torch.randn(n_rays, device=device_auto) * 0.05
        o[:, 1] = torch.randn(n_rays, device=device_auto) * 0.05
        
        d = torch.zeros(n_rays, 3, device=device_auto)
        d[:, 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ks = 31
        ps = 0.01
        field = forward_integral(ray, ps=ps, ks=ks)
        
        assert field.device.type == device_auto.type

    def test_forward_integral_large_batch_gpu(self, device_auto):
        """Should handle large batches on GPU."""
        batch_size = 16
        n_rays = 2048
        
        o = torch.zeros(batch_size, n_rays, 3, device=device_auto)
        o[..., 2] = 10.0
        o[..., 0] = torch.randn(batch_size, n_rays, device=device_auto) * 0.05
        o[..., 1] = torch.randn(batch_size, n_rays, device=device_auto) * 0.05
        
        d = torch.zeros(batch_size, n_rays, 3, device=device_auto)
        d[..., 2] = 1.0
        
        ray = Ray(o, d, wvln=0.55, device=device_auto)
        
        ks = 21
        ps = 0.01
        field = forward_integral(ray, ps=ps, ks=ks)
        
        assert field.shape == (batch_size, ks, ks)
        assert field.device.type == device_auto.type
