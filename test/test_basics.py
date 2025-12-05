"""
Tests for deeplens/basics.py - Basic utilities and constants.
"""

import pytest
import torch


from deeplens.basics import (
    DEPTH,
    DEFAULT_WAVE,
    EPSILON,
    PSF_KS,
    SPP_PSF,
    WAVE_RGB,
    DeepObj,
    init_device,
    wave_rgb,
)


class TestConstants:
    """Test default constants are properly defined."""

    def test_depth_constant(self):
        """DEPTH should be a large negative value representing infinity."""
        assert DEPTH == -20000.0
        assert DEPTH < 0

    def test_wave_rgb(self):
        """WAVE_RGB should contain R, G, B wavelengths in micrometers."""
        assert len(WAVE_RGB) == 3
        assert WAVE_RGB[0] > WAVE_RGB[1] > WAVE_RGB[2]  # R > G > B
        # All wavelengths should be in visible range (0.38 - 0.78 um)
        for wvln in WAVE_RGB:
            assert 0.38 < wvln < 0.78

    def test_default_wave(self):
        """DEFAULT_WAVE should be green wavelength."""
        assert 0.5 < DEFAULT_WAVE < 0.6  # Green light

    def test_spp_psf(self):
        """SPP_PSF should be a power of 2."""
        assert SPP_PSF > 0
        assert (SPP_PSF & (SPP_PSF - 1)) == 0  # Check power of 2

    def test_psf_ks(self):
        """PSF_KS should be a reasonable kernel size."""
        assert PSF_KS > 0
        assert PSF_KS < 256

    def test_epsilon(self):
        """EPSILON should be a small positive value."""
        assert EPSILON > 0
        assert EPSILON < 1e-6


class TestInitDevice:
    """Test device initialization."""

    def test_init_device_returns_device(self):
        """init_device should return a torch device."""
        device = init_device()
        assert isinstance(device, torch.device)

    def test_init_device_cuda_or_cpu(self):
        """init_device should return cuda or cpu."""
        device = init_device()
        assert device.type in ["cuda", "cpu"]

    def test_init_device_matches_availability(self):
        """init_device result should match CUDA availability."""
        device = init_device()
        if torch.cuda.is_available():
            assert device.type == "cuda"
        else:
            assert device.type == "cpu"


class TestWaveRgb:
    """Test random wavelength sampling."""

    def test_wave_rgb_returns_three(self):
        """wave_rgb should return 3 wavelengths."""
        waves = wave_rgb()
        assert len(waves) == 3

    def test_wave_rgb_order(self):
        """wave_rgb should return [R, G, B] in decreasing wavelength order."""
        waves = wave_rgb()
        assert waves[0] > waves[1] > waves[2]

    def test_wave_rgb_range(self):
        """All wavelengths should be in visible range."""
        waves = wave_rgb()
        for w in waves:
            assert 0.4 < w < 0.75

    def test_wave_rgb_randomness(self):
        """wave_rgb should produce different results (probabilistic)."""
        results = [tuple(wave_rgb()) for _ in range(10)]
        # At least some should be different
        assert len(set(results)) > 1


class TestDeepObj:
    """Test DeepObj base class functionality."""

    def test_deep_obj_init(self):
        """DeepObj should initialize with default dtype."""
        obj = DeepObj()
        assert obj.dtype == torch.get_default_dtype()

    def test_deep_obj_init_custom_dtype(self):
        """DeepObj should accept custom dtype."""
        obj = DeepObj(dtype=torch.float64)
        assert obj.dtype == torch.float64

    def test_deep_obj_str(self):
        """DeepObj should have string representation."""
        obj = DeepObj()
        s = str(obj)
        assert "DeepObj" in s

    def test_deep_obj_clone(self):
        """DeepObj clone should create independent copy."""
        obj = DeepObj()
        obj.test_attr = torch.tensor([1.0, 2.0, 3.0])
        cloned = obj.clone()
        
        # Modify original
        obj.test_attr[0] = 999.0
        
        # Clone should be unchanged
        assert cloned.test_attr[0] != 999.0

    def test_deep_obj_to_device(self, device_auto):
        """DeepObj.to() should move tensors to device."""
        obj = DeepObj()
        obj.tensor_attr = torch.tensor([1.0, 2.0, 3.0])
        
        obj.to(device_auto)
        
        assert obj.device.type == device_auto.type
        assert obj.tensor_attr.device.type == device_auto.type

    def test_deep_obj_to_device_nested(self, device_auto):
        """DeepObj.to() should handle nested DeepObj."""
        outer = DeepObj()
        inner = DeepObj()
        inner.data = torch.tensor([1.0, 2.0])
        outer.child = inner
        
        outer.to(device_auto)
        
        assert inner.data.device.type == device_auto.type

    def test_deep_obj_to_device_list(self, device_auto):
        """DeepObj.to() should handle tensor lists."""
        obj = DeepObj()
        obj.tensor_list = [torch.tensor([1.0]), torch.tensor([2.0])]
        
        obj.to(device_auto)
        
        for t in obj.tensor_list:
            assert t.device.type == device_auto.type

    def test_deep_obj_astype_float32(self):
        """DeepObj.astype() should convert to float32."""
        obj = DeepObj(dtype=torch.float64)
        obj.data = torch.tensor([1.0, 2.0], dtype=torch.float64)
        
        obj.astype(torch.float32)
        
        assert obj.dtype == torch.float32
        assert obj.data.dtype == torch.float32

    def test_deep_obj_astype_float64(self):
        """DeepObj.astype() should convert to float64."""
        obj = DeepObj(dtype=torch.float32)
        obj.data = torch.tensor([1.0, 2.0], dtype=torch.float32)
        
        obj.astype(torch.float64)
        
        assert obj.dtype == torch.float64
        assert obj.data.dtype == torch.float64

    def test_deep_obj_astype_none(self):
        """DeepObj.astype(None) should be no-op."""
        obj = DeepObj(dtype=torch.float32)
        original_dtype = obj.dtype
        
        result = obj.astype(None)
        
        assert obj.dtype == original_dtype
        assert result is obj

    def test_deep_obj_astype_invalid(self):
        """DeepObj.astype() should reject invalid dtypes."""
        obj = DeepObj()
        
        with pytest.raises(AssertionError):
            obj.astype(torch.int32)

    def test_deep_obj_call_raises(self):
        """DeepObj.__call__() should raise if forward not implemented."""
        obj = DeepObj()
        
        with pytest.raises(AttributeError):
            obj(torch.tensor([1.0]))
