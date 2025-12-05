"""
Tests for deeplens/utils.py - Utility functions.
"""

import pytest
import torch
import numpy as np

from deeplens.utils import (
    interp1d,
    grid_sample_xy,
    img2batch,
    batch_psnr,
    batch_ssim,
    normalize_ImageNet,
    denormalize_ImageNet,
    foc_dist_balanced,
)


class TestInterp1d:
    """Test 1D interpolation."""

    def test_interp1d_exact_keys(self, device_auto):
        """Query at key points should return exact values."""
        key = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        value = torch.tensor([0.0, 10.0, 20.0], device=device_auto)
        query = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert torch.allclose(result, value, atol=1e-5)

    def test_interp1d_midpoint(self, device_auto):
        """Midpoint query should give midpoint value."""
        key = torch.tensor([0.0, 2.0], device=device_auto)
        value = torch.tensor([0.0, 20.0], device=device_auto)
        query = torch.tensor([1.0], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert torch.allclose(result, torch.tensor([10.0], device=device_auto), atol=1e-5)

    def test_interp1d_batch(self, device_auto):
        """Should handle batched values."""
        key = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        value = torch.tensor([[0.0, 0.0], [10.0, 20.0], [20.0, 40.0]], device=device_auto)
        query = torch.tensor([0.5, 1.5], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert result.shape == (2, 2)


class TestGridSampleXY:
    """Test grid sampling with xy coordinates."""

    def test_grid_sample_xy_identity(self, device_auto):
        """Identity grid should return same image."""
        img = torch.rand(1, 3, 32, 32, device=device_auto)
        
        # Create identity grid
        y = torch.linspace(1, -1, 32, device=device_auto)
        x = torch.linspace(-1, 1, 32, device=device_auto)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        grid_xy = torch.stack([xx, yy], dim=-1).unsqueeze(0)
        
        result = grid_sample_xy(img, grid_xy, align_corners=True)
        
        assert torch.allclose(result, img, atol=1e-4)

    def test_grid_sample_xy_shape(self, device_auto):
        """Output shape should match grid shape."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        # Smaller output grid
        grid_xy = torch.rand(1, 32, 32, 2, device=device_auto) * 2 - 1
        
        result = grid_sample_xy(img, grid_xy)
        
        assert result.shape == (1, 3, 32, 32)


class TestImg2Batch:
    """Test image to batch conversion."""

    def test_img2batch_numpy_hwc(self, device_auto):
        """Should convert numpy HWC image to batch."""
        img = np.random.rand(64, 64, 3).astype(np.float32)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)
        assert isinstance(batch, torch.Tensor)

    def test_img2batch_tensor_chw(self, device_auto):
        """Should convert tensor CHW image to batch."""
        img = torch.rand(3, 64, 64, device=device_auto)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)

    def test_img2batch_tensor_hwc(self, device_auto):
        """Should convert tensor HWC image to batch."""
        img = torch.rand(64, 64, 3, device=device_auto)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)

    def test_img2batch_already_batch(self, device_auto):
        """Should handle already batched image."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        batch = img2batch(img)
        
        assert batch.shape == (1, 3, 64, 64)


class TestBatchPSNR:
    """Test batch PSNR calculation."""

    def test_batch_psnr_identical(self, device_auto):
        """PSNR of identical images should be very high."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        psnr = batch_psnr(img, img)
        
        assert psnr.item() > 40  # Very high PSNR

    def test_batch_psnr_different(self, device_auto):
        """PSNR of different images should be finite."""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = torch.rand(1, 3, 64, 64, device=device_auto)
        
        psnr = batch_psnr(img1, img2)
        
        assert psnr.item() > 0
        assert psnr.item() < 60

    def test_batch_psnr_batch(self, device_auto):
        """Should handle batched input."""
        pred = torch.rand(4, 3, 64, 64, device=device_auto)
        target = pred + torch.randn_like(pred) * 0.1
        target = target.clamp(0, 1)
        
        psnr = batch_psnr(pred, target)
        
        assert psnr.shape == (4,)


class TestBatchSSIM:
    """Test batch SSIM calculation."""

    def test_batch_ssim_identical(self, device_auto):
        """SSIM of identical images should be 1."""
        img = torch.rand(1, 3, 64, 64, device=device_auto)
        
        ssim = batch_ssim(img, img)
        
        assert ssim == pytest.approx(1.0, abs=0.01)

    def test_batch_ssim_different(self, device_auto):
        """SSIM of different images should be less than 1."""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = torch.rand(1, 3, 64, 64, device=device_auto)
        
        ssim = batch_ssim(img1, img2)
        
        assert ssim < 1.0
        assert ssim > -1.0

    def test_batch_ssim_range(self, device_auto):
        """SSIM should be in [-1, 1] range."""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = 1 - img1  # Inverted image
        
        ssim = batch_ssim(img1, img2)
        
        assert -1 <= ssim <= 1


class TestImageNetNormalization:
    """Test ImageNet normalization."""

    def test_normalize_imagenet_shape(self, device_auto):
        """Normalization should preserve shape."""
        batch = torch.rand(4, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        
        assert normalized.shape == batch.shape

    def test_normalize_imagenet_range(self, device_auto):
        """Normalized values should be roughly centered around 0."""
        batch = torch.rand(4, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        
        # Mean should be close to 0
        assert normalized.mean().abs() < 1.0

    def test_denormalize_imagenet(self, device_auto):
        """Denormalization should invert normalization."""
        batch = torch.rand(4, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        denormalized = denormalize_ImageNet(normalized)
        
        assert torch.allclose(denormalized, batch, atol=1e-5)


class TestFocDistBalanced:
    """Test focus distance calculation."""

    def test_foc_dist_balanced_symmetric(self, device_auto):
        """Equal distances should give geometric mean focus."""
        d1 = -1000.0
        d2 = -1000.0
        
        foc = foc_dist_balanced(d1, d2)
        
        assert foc == pytest.approx(d1, abs=1.0)

    def test_foc_dist_balanced_asymmetric(self, device_auto):
        """Asymmetric distances should give balanced focus."""
        d1 = -500.0
        d2 = -2000.0
        
        foc = foc_dist_balanced(d1, d2)
        
        # Result should be between d1 and d2
        assert min(d1, d2) < foc < max(d1, d2)

    def test_foc_dist_balanced_negative(self, device_auto):
        """Should handle negative distances (in front of lens)."""
        d1 = -100.0
        d2 = -10000.0
        
        foc = foc_dist_balanced(d1, d2)
        
        assert foc < 0


class TestUtilsGPU:
    """Test utility functions on GPU."""

    def test_interp1d_gpu(self, device_auto):
        """Interpolation should work on GPU."""
        key = torch.tensor([0.0, 1.0, 2.0], device=device_auto)
        value = torch.tensor([0.0, 10.0, 20.0], device=device_auto)
        query = torch.tensor([0.5, 1.5], device=device_auto)
        
        result = interp1d(query, key, value)
        
        assert result.device.type == device_auto.type

    def test_batch_psnr_gpu(self, device_auto):
        """PSNR should work on GPU."""
        img1 = torch.rand(1, 3, 64, 64, device=device_auto)
        img2 = torch.rand(1, 3, 64, 64, device=device_auto)
        
        psnr = batch_psnr(img1, img2)
        
        assert isinstance(psnr, torch.Tensor)

    def test_normalize_imagenet_gpu(self, device_auto):
        """Normalization should work on GPU."""
        batch = torch.rand(1, 3, 64, 64, device=device_auto)
        
        normalized = normalize_ImageNet(batch)
        
        assert normalized.device.type == device_auto.type
