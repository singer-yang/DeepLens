"""
Tests for deeplens/optics/psf.py - PSF convolution functions.
"""

import pytest
import torch

from deeplens.optics.psf import (
    conv_psf,
    conv_psf_map,
    conv_psf_pixel,
    conv_psf_depth_interp,
    crop_psf_map,
    interp_psf_map,
    rotate_psf,
)


class TestConvPSF:
    """Test single PSF convolution."""

    def test_conv_psf_shape(self, device_auto):
        """Output should have same shape as input."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        psf = torch.rand(3, 11, 11, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)  # Normalize
        
        result = conv_psf(img, psf)
        
        assert result.shape == img.shape

    def test_conv_psf_normalized(self, device_auto):
        """Convolution with normalized PSF should preserve total energy."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        psf = torch.ones(3, 11, 11, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf(img, psf)
        
        # Total energy should be approximately preserved
        energy_in = img.sum()
        energy_out = result.sum()
        assert torch.allclose(energy_in, energy_out, rtol=0.1)

    def test_conv_psf_delta(self, device_auto):
        """Delta function PSF should return original image."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # Create delta PSF
        psf = torch.zeros(3, 11, 11, device=device_auto)
        psf[:, 5, 5] = 1.0
        
        result = conv_psf(img, psf)
        
        # Should be very close to original
        assert torch.allclose(result, img, atol=1e-5)

    def test_conv_psf_blur(self, device_auto):
        """Box PSF should blur the image."""
        # Create image with sharp edges
        img = torch.zeros(1, 3, 64, 64, device=device_auto)
        img[:, :, 20:44, 20:44] = 1.0
        
        # Box blur PSF
        psf = torch.ones(3, 5, 5, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf(img, psf)
        
        # Edges should be smoothed
        edge_sharpness_before = (img[:, :, 19, 32] - img[:, :, 20, 32]).abs()
        edge_sharpness_after = (result[:, :, 19, 32] - result[:, :, 20, 32]).abs()
        assert edge_sharpness_after.mean() < edge_sharpness_before.mean()


class TestConvPSFMap:
    """Test spatially-varying PSF convolution."""

    def test_conv_psf_map_shape(self, device_auto):
        """Output should have same shape as input."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # PSF map: [grid_h, grid_w, C, ks, ks]
        psf_map = torch.rand(4, 4, 3, 11, 11, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf_map(img, psf_map)
        
        assert result.shape == img.shape

    def test_conv_psf_map_uniform(self, device_auto):
        """Uniform PSF map should give same result as single PSF."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # Create uniform PSF (same at all grid points)
        single_psf = torch.rand(3, 11, 11, device=device_auto)
        single_psf = single_psf / single_psf.sum(dim=(-1, -2), keepdim=True)
        
        psf_map = single_psf.unsqueeze(0).unsqueeze(0).expand(4, 4, -1, -1, -1).clone()
        
        result_map = conv_psf_map(img, psf_map)
        result_single = conv_psf(img, single_psf)
        
        # Results should be similar
        assert torch.allclose(result_map, result_single, atol=0.1)


class TestConvPSFPixel:
    """Test per-pixel PSF convolution."""

    def test_conv_psf_pixel_shape(self, device_auto):
        """Output should have same shape as input."""
        img = torch.rand(1, 3, 32, 32, device=device_auto)
        
        # Per-pixel PSF: [H, W, C, ks, ks]
        psf = torch.rand(32, 32, 3, 5, 5, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf_pixel(img, psf)
        
        assert result.shape == img.shape


class TestConvPSFDepthInterp:
    """Test depth-interpolated PSF convolution."""

    def test_conv_psf_depth_interp_shape(self, device_auto):
        """Output should have same shape as input."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.rand(1, 1, 64, 64, device=device_auto)
        
        # PSF kernels at different depths
        psf_kernels = torch.rand(5, 3, 11, 11, device=device_auto)
        psf_kernels = psf_kernels / psf_kernels.sum(dim=(-1, -2), keepdim=True)
        
        # Depth values for each PSF
        psf_depths = torch.linspace(0, 1, 5, device=device_auto)
        
        result = conv_psf_depth_interp(img, depth, psf_kernels, psf_depths)
        
        assert result.shape == img.shape

    def test_conv_psf_depth_interp_extreme_depths(self, device_auto):
        """Should handle depth at boundaries."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # Depth at minimum value
        depth = torch.zeros(1, 1, 64, 64, device=device_auto)
        
        psf_kernels = torch.rand(5, 3, 11, 11, device=device_auto)
        psf_kernels = psf_kernels / psf_kernels.sum(dim=(-1, -2), keepdim=True)
        psf_depths = torch.linspace(0, 1, 5, device=device_auto)
        
        result = conv_psf_depth_interp(img, depth, psf_kernels, psf_depths)
        
        assert not torch.isnan(result).any()


class TestCropPSFMap:
    """Test PSF map cropping."""

    def test_crop_psf_map_shape(self, device_auto):
        """Should crop PSF patches correctly."""
        grid = 4
        ks = 21
        ks_crop = 11
        
        psf_map = torch.rand(3, grid * ks, grid * ks, device=device_auto)
        
        cropped = crop_psf_map(psf_map, grid=grid, ks_crop=ks_crop)
        
        assert cropped.shape == (3, grid * ks_crop, grid * ks_crop)

    def test_crop_psf_map_center(self, device_auto):
        """Cropping should take center of each patch."""
        grid = 2
        ks = 11
        ks_crop = 5
        
        # Create PSF map with known values
        psf_map = torch.zeros(3, grid * ks, grid * ks, device=device_auto)
        # Put value in center of each patch
        for i in range(grid):
            for j in range(grid):
                center_y = i * ks + ks // 2
                center_x = j * ks + ks // 2
                psf_map[:, center_y, center_x] = 1.0
        
        cropped = crop_psf_map(psf_map, grid=grid, ks_crop=ks_crop)
        
        # Center values should be preserved
        for i in range(grid):
            for j in range(grid):
                center_y = i * ks_crop + ks_crop // 2
                center_x = j * ks_crop + ks_crop // 2
                assert cropped[0, center_y, center_x] == 1.0


class TestInterpPSFMap:
    """Test PSF map interpolation."""

    def test_interp_psf_map_upsample(self, device_auto):
        """Should upsample PSF grid."""
        grid_old = 3
        grid_new = 6
        ks = 11
        
        psf_map = torch.rand(3, grid_old * ks, grid_old * ks, device=device_auto)
        
        interpolated = interp_psf_map(psf_map, grid_old=grid_old, grid_new=grid_new)
        
        assert interpolated.shape == (3, grid_new * ks, grid_new * ks)

    def test_interp_psf_map_identity(self, device_auto):
        """Same grid size should return similar map."""
        grid = 4
        ks = 11
        
        psf_map = torch.rand(3, grid * ks, grid * ks, device=device_auto)
        
        interpolated = interp_psf_map(psf_map, grid_old=grid, grid_new=grid)
        
        assert torch.allclose(interpolated, psf_map, atol=0.01)


class TestRotatePSF:
    """Test PSF rotation."""

    def test_rotate_psf_shape(self, device_auto):
        """Rotation should preserve shape."""
        psf = torch.rand(4, 3, 21, 21, device=device_auto)
        theta = torch.tensor([0.0, 0.5, 1.0, 1.5], device=device_auto)
        
        rotated = rotate_psf(psf, theta)
        
        assert rotated.shape == psf.shape

    def test_rotate_psf_zero(self, device_auto):
        """Zero rotation should return same PSF."""
        psf = torch.rand(1, 3, 21, 21, device=device_auto)
        theta = torch.tensor([0.0], device=device_auto)
        
        rotated = rotate_psf(psf, theta)
        
        assert torch.allclose(rotated, psf, atol=1e-4)

    def test_rotate_psf_symmetric(self, device_auto):
        """Symmetric PSF should be unchanged by rotation."""
        # Create circularly symmetric PSF (Gaussian-like)
        ks = 21
        center = ks // 2
        y, x = torch.meshgrid(torch.arange(ks), torch.arange(ks), indexing="ij")
        r = torch.sqrt((x - center).float()**2 + (y - center).float()**2)
        psf_single = torch.exp(-r**2 / 10)
        psf_single = psf_single / psf_single.sum()
        
        psf = psf_single.unsqueeze(0).unsqueeze(0).expand(1, 3, -1, -1).to(device_auto)
        theta = torch.tensor([1.57], device=device_auto)  # 90 degrees
        
        rotated = rotate_psf(psf, theta)
        
        # Should be approximately the same due to symmetry
        assert torch.allclose(rotated, psf, atol=0.05)


class TestPSFGPUPerformance:
    """Test PSF operations on GPU."""

    def test_conv_psf_gpu_batch(self, device_auto):
        """Should handle batched input on GPU."""
        batch_size = 4
        img = torch.rand(batch_size, 3, 128, 128, device=device_auto)
        psf = torch.rand(3, 21, 21, device=device_auto)
        psf = psf / psf.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf(img, psf)
        
        assert result.shape == img.shape
        assert result.device.type == device_auto.type

    def test_conv_psf_map_gpu(self, device_auto):
        """PSF map convolution should work on GPU."""
        img = torch.rand(1, 3, 128, 128, device=device_auto)
        psf_map = torch.rand(8, 8, 3, 15, 15, device=device_auto)
        psf_map = psf_map / psf_map.sum(dim=(-1, -2), keepdim=True)
        
        result = conv_psf_map(img, psf_map)
        
        assert result.device.type == device_auto.type
