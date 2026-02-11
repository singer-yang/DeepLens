"""
Tests for deeplens/sensor/ - Sensor base class and MonoSensor.
"""

import pytest
import torch

from deeplens.sensor.sensor import Sensor
from deeplens.sensor.mono_sensor import MonoSensor


# ===========================
# Base Sensor tests
# ===========================

class TestSensorInit:
    """Test Sensor initialization."""

    def test_sensor_init_default(self, device_auto):
        """Should initialize with default parameters."""
        sensor = Sensor()
        assert sensor.size == (8.0, 6.0)
        assert sensor.res == (4000, 3000)
        assert sensor.pixel_size > 0

    def test_sensor_init_custom(self, device_auto):
        """Should initialize with custom parameters."""
        sensor = Sensor(size=(10.0, 8.0), res=(5000, 4000))
        assert sensor.size == (10.0, 8.0)
        assert sensor.res == (5000, 4000)

    def test_sensor_to_device(self, device_auto):
        """Should move to device."""
        sensor = Sensor()
        sensor.to(device_auto)
        assert sensor.device.type == device_auto.type


class TestSensorForward:
    """Test base Sensor forward pass (gamma-only)."""

    def test_sensor_forward_shape(self, device_auto):
        """Forward should preserve shape."""
        sensor = Sensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto)
        output = sensor.forward(img)
        assert output.shape == img.shape

    def test_sensor_forward_range(self, device_auto):
        """Forward output should be in [0, 1] for input in [0, 1]."""
        sensor = Sensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto)
        output = sensor.forward(img)
        assert output.min() >= 0
        assert output.max() <= 1.0

    def test_sensor_forward_applies_gamma(self, device_auto):
        """Forward should apply gamma correction (output != input)."""
        sensor = Sensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto) * 0.5 + 0.1
        output = sensor.forward(img)
        # Gamma correction should change values (for non-trivial input)
        assert not torch.allclose(output, img, atol=1e-3)

    def test_sensor_call(self, device_auto):
        """__call__ should work as forward."""
        sensor = Sensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto)
        output = sensor(img)
        assert output.shape == img.shape


class TestSensorResponseCurve:
    """Test sensor response curve."""

    def test_response_curve_linear(self, device_auto):
        """Default response curve should be identity."""
        sensor = Sensor()

        img_irr = torch.rand(1, 3, 64, 64, device=device_auto)
        img_raw = sensor.response_curve(img_irr)
        assert torch.allclose(img_irr, img_raw)


class TestSensorSimuNoise:
    """Test base sensor noise simulation (identity)."""

    def test_simu_noise_identity(self, device_auto):
        """Default simu_noise should return input unchanged."""
        sensor = Sensor()

        img = torch.rand(1, 3, 64, 64, device=device_auto)
        img_out = sensor.simu_noise(img)
        assert torch.allclose(img, img_out)


class TestSensorPixelSize:
    """Test sensor pixel size calculation."""

    def test_sensor_pixel_size(self, device_auto):
        """Pixel size should be computed from resolution."""
        sensor = Sensor(size=(8.0, 6.0), res=(4000, 3000))
        assert sensor.pixel_size > 0


# ===========================
# MonoSensor tests
# ===========================

class TestMonoSensorInit:
    """Test MonoSensor initialization."""

    def test_mono_sensor_init_default(self, device_auto):
        """Should initialize with default parameters."""
        sensor = MonoSensor()
        assert sensor.bit == 10
        assert sensor.black_level == 64
        assert sensor.nbit_max == 2**10 - 1

    def test_mono_sensor_init_custom(self, device_auto):
        """Should initialize with custom parameters."""
        sensor = MonoSensor(
            bit=12,
            black_level=256,
            size=(8.0, 6.0),
            res=(4000, 3000),
            read_noise_std=1.0,
            shot_noise_std_alpha=0.5,
        )
        assert sensor.bit == 12
        assert sensor.black_level == 256
        assert sensor.nbit_max == 2**12 - 1


class TestMonoSensorNoise:
    """Test MonoSensor noise simulation."""

    def test_simu_noise_shape(self, device_auto):
        """Noise simulation should preserve shape."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert img_noisy.shape == img.shape

    def test_simu_noise_adds_variance(self, device_auto):
        """Noise should add variance to image."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.ones(1, 3, 64, 64, device=device_auto) * 512
        iso = torch.tensor([400], device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert img_noisy.std() > 0

    def test_simu_noise_clipped(self, device_auto):
        """Output should be clipped to valid range."""
        sensor = MonoSensor(bit=10)
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto) * sensor.nbit_max
        iso = torch.tensor([800], device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert img_noisy.min() >= 0
        assert img_noisy.max() <= sensor.nbit_max

    def test_simu_noise_quantized(self, device_auto):
        """Output should be quantized (integer values)."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert torch.allclose(img_noisy, img_noisy.round())

    def test_simu_noise_iso_effect(self, device_auto):
        """Higher ISO should mean more noise."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.ones(1, 3, 64, 64, device=device_auto) * 512

        iso_low = torch.tensor([100], device=device_auto)
        iso_high = torch.tensor([800], device=device_auto)

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


class TestMonoSensorForward:
    """Test MonoSensor forward pass."""

    def test_forward_shape(self, device_auto):
        """Forward should preserve shape."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        output = sensor.forward(img, iso)
        assert output.shape == img.shape

    def test_forward_range(self, device_auto):
        """Forward output should be in [0, 1] after ISP."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100], device=device_auto)
        output = sensor.forward(img, iso)
        assert output.min() >= 0
        assert output.max() <= 1.0


class TestMonoSensorBatch:
    """Test MonoSensor with batched input."""

    def test_batch_noise(self, device_auto):
        """Should handle batched images."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        batch_size = 4
        img = torch.rand(batch_size, 3, 64, 64, device=device_auto) * 500 + 64
        iso = torch.tensor([100] * batch_size, device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert img_noisy.shape == img.shape

    def test_batch_different_iso(self, device_auto):
        """Should handle different ISO per batch."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.ones(2, 3, 64, 64, device=device_auto) * 512
        iso = torch.tensor([100, 400], device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert img_noisy.shape == img.shape


class TestMonoSensorGPU:
    """Test MonoSensor operations on GPU."""

    def test_noise_gpu(self, device_auto):
        """Noise simulation should work on GPU."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 128, 128, device=device_auto) * 500 + 64
        iso = torch.tensor([200], device=device_auto)
        img_noisy = sensor.simu_noise(img, iso)
        assert img_noisy.device.type == device_auto.type

    def test_forward_gpu(self, device_auto):
        """Full forward pass should work on GPU."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 128, 128, device=device_auto) * 500 + 64
        iso = torch.tensor([200], device=device_auto)
        output = sensor.forward(img, iso)
        assert output.device.type == device_auto.type
