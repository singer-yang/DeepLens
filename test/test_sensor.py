"""
Tests for deeplens/sensor/ - Sensor noise and ISP.
"""

import pytest
import torch

from deeplens.sensor.sensor import Sensor


class TestSensorInit:
    """Test Sensor initialization."""

    def test_sensor_init_default(self, device_auto):
        """Should initialize with default parameters."""
        sensor = Sensor(device=device_auto)
        
        assert sensor.bit == 10
        assert sensor.black_level == 64
        assert sensor.nbit_max == 2**10 - 1

    def test_sensor_init_custom(self, device_auto):
        """Should initialize with custom parameters."""
        sensor = Sensor(
            bit=12,
            black_level=256,
            size=(8.0, 6.0),
            res=(4000, 3000),
            read_noise_std=1.0,
            shot_noise_std_alpha=0.5,
            device=device_auto,
        )
        
        assert sensor.bit == 12
        assert sensor.black_level == 256
        assert sensor.nbit_max == 2**12 - 1

    def test_sensor_to_device(self, device_auto):
        """Should move to device."""
        sensor = Sensor()
        sensor.to(device_auto)
        
        assert sensor.device.type == device_auto.type


class TestSensorNoise:
    """Test sensor noise simulation."""

    def test_sensor_simu_noise_shape(self, device_auto):
        """Noise simulation should preserve shape."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64  # In nbit range
        iso = torch.tensor([100], device=device_auto)
        
        img_noisy = sensor.simu_noise(img, iso)
        
        assert img_noisy.shape == img.shape

    def test_sensor_simu_noise_adds_variance(self, device_auto):
        """Noise should add variance to image."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        # Uniform input
        img = torch.ones(1, 3, 64, 64, device=device_auto) * 512
        iso = torch.tensor([400], device=device_auto)
        
        img_noisy = sensor.simu_noise(img, iso)
        
        # Output should have some variance
        assert img_noisy.std() > 0

    def test_sensor_simu_noise_clipped(self, device_auto):
        """Output should be clipped to valid range."""
        sensor = Sensor(bit=10, device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 64, 64, device=device_auto) * sensor.nbit_max
        iso = torch.tensor([800], device=device_auto)  # High ISO for more noise
        
        img_noisy = sensor.simu_noise(img, iso)
        
        assert img_noisy.min() >= 0
        assert img_noisy.max() <= sensor.nbit_max

    def test_sensor_simu_noise_quantized(self, device_auto):
        """Output should be quantized (integer values)."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        
        img_noisy = sensor.simu_noise(img, iso)
        
        # Check that values are quantized
        assert torch.allclose(img_noisy, img_noisy.round())

    def test_sensor_simu_noise_iso_effect(self, device_auto):
        """Higher ISO should mean more noise."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.ones(1, 3, 64, 64, device=device_auto) * 512
        
        iso_low = torch.tensor([100], device=device_auto)
        iso_high = torch.tensor([800], device=device_auto)
        
        # Run multiple times to get statistics
        noises_low = []
        noises_high = []
        for _ in range(10):
            img_low = sensor.simu_noise(img.clone(), iso_low)
            img_high = sensor.simu_noise(img.clone(), iso_high)
            noises_low.append((img_low - img).std().item())
            noises_high.append((img_high - img).std().item())
        
        avg_noise_low = sum(noises_low) / len(noises_low)
        avg_noise_high = sum(noises_high) / len(noises_high)
        
        assert avg_noise_high > avg_noise_low


class TestSensorForward:
    """Test sensor forward pass."""

    def test_sensor_forward_shape(self, device_auto):
        """Forward should preserve shape."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        
        output = sensor.forward(img, iso)
        
        assert output.shape == img.shape

    def test_sensor_forward_range(self, device_auto):
        """Forward output should be in [0, 1] after ISP."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        
        output = sensor.forward(img, iso)
        
        assert output.min() >= 0
        assert output.max() <= 1.0

    def test_sensor_call(self, device_auto):
        """__call__ should be alias for forward."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        
        output = sensor(img, iso)
        
        assert output.shape == img.shape


class TestSensorResponseCurve:
    """Test sensor response curve."""

    def test_response_curve_linear(self, device_auto):
        """Default response curve should be linear."""
        sensor = Sensor(device=device_auto)
        
        img_irr = torch.rand(1, 3, 64, 64, device=device_auto)
        img_raw = sensor.response_curve(img_irr)
        
        # For linear response, output equals input
        assert torch.allclose(img_irr, img_raw)


class TestSensorBatch:
    """Test sensor with batched input."""

    def test_sensor_batch_noise(self, device_auto):
        """Should handle batched images."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        batch_size = 4
        img = torch.rand(batch_size, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100] * batch_size, device=device_auto)
        
        img_noisy = sensor.simu_noise(img, iso)
        
        assert img_noisy.shape == img.shape

    def test_sensor_batch_different_iso(self, device_auto):
        """Should handle different ISO per batch."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.ones(2, 3, 64, 64, device=device_auto) * 512
        iso = torch.tensor([100, 400], device=device_auto)
        
        # This should work without error
        img_noisy = sensor.simu_noise(img, iso)
        
        assert img_noisy.shape == img.shape


class TestSensorPixelSize:
    """Test sensor pixel size calculation."""

    def test_sensor_pixel_size(self, device_auto):
        """Pixel size should be computed from resolution."""
        sensor = Sensor(
            size=(8.0, 6.0),
            res=(4000, 3000),
            device=device_auto,
        )
        
        # Pixel size is normalized
        assert sensor.pixel_size > 0


class TestSensorGPU:
    """Test sensor operations on GPU."""

    def test_sensor_noise_gpu(self, device_auto):
        """Noise simulation should work on GPU."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 128, 128, device=device_auto) * 500 + 64
        iso = torch.tensor([200], device=device_auto)
        
        img_noisy = sensor.simu_noise(img, iso)
        
        assert img_noisy.device.type == device_auto.type

    def test_sensor_forward_gpu(self, device_auto):
        """Full forward pass should work on GPU."""
        sensor = Sensor(device=device_auto)
        sensor.to(device_auto)
        
        img = torch.rand(1, 3, 128, 128, device=device_auto) * 500 + 64
        iso = torch.tensor([200], device=device_auto)
        
        output = sensor.forward(img, iso)
        
        assert output.device.type == device_auto.type
