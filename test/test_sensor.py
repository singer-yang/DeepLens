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


class TestMonoSensorResponseCurve:
    """Test MonoSensor spectral response curve."""

    def test_response_curve_no_spectral(self, device_auto):
        """Without spectral response, single-channel input passes through."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 1, 64, 64, device=device_auto)
        img_raw = sensor.response_curve(img)
        assert torch.allclose(img_raw, img)

    def test_response_curve_multichannel_fallback(self, device_auto):
        """Without spectral response, multi-channel input is averaged."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 64, 64, device=device_auto)
        img_raw = sensor.response_curve(img)
        assert img_raw.shape == (1, 1, 64, 64)
        assert torch.allclose(img_raw, img.mean(dim=1, keepdim=True))

    def test_response_curve_spectral(self, device_auto):
        """With spectral response, should weight and sum spectral channels."""
        wavelengths = [400, 500, 600]
        spectral_response = [0.2, 0.6, 0.2]
        sensor = MonoSensor(wavelengths=wavelengths, spectral_response=spectral_response)
        sensor.to(device_auto)

        img = torch.ones(1, 3, 64, 64, device=device_auto)
        img_raw = sensor.response_curve(img)
        assert img_raw.shape == (1, 1, 64, 64)
        # Normalized response sums to 1, so uniform input -> 1.0
        assert torch.allclose(img_raw, torch.ones_like(img_raw), atol=1e-5)

    def test_response_curve_spectral_shape(self, device_auto):
        """Spectral response should reduce N wavelength channels to 1."""
        wavelengths = [400, 450, 500, 550, 600]
        spectral_response = [0.1, 0.3, 0.5, 0.3, 0.1]
        sensor = MonoSensor(wavelengths=wavelengths, spectral_response=spectral_response)
        sensor.to(device_auto)

        img = torch.rand(2, 5, 32, 32, device=device_auto)
        img_raw = sensor.response_curve(img)
        assert img_raw.shape == (2, 1, 32, 32)

    def test_response_curve_spectral_weighting(self, device_auto):
        """Spectral response should apply correct weights."""
        wavelengths = [400, 500, 600]
        spectral_response = [1.0, 0.0, 0.0]  # only responds to first channel
        sensor = MonoSensor(wavelengths=wavelengths, spectral_response=spectral_response)
        sensor.to(device_auto)

        img = torch.zeros(1, 3, 8, 8, device=device_auto)
        img[:, 0] = 0.5  # only first channel has signal
        img[:, 1] = 0.8
        img[:, 2] = 0.3

        img_raw = sensor.response_curve(img)
        # Should only see channel 0 contribution
        assert torch.allclose(img_raw, torch.full_like(img_raw, 0.5), atol=1e-5)


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

    def test_simu_noise_gpu(self, device_auto):
        """Noise simulation should produce correct device output on GPU."""
        sensor = MonoSensor()
        sensor.to(device_auto)

        img = torch.rand(1, 3, 128, 128, device=device_auto) * 500 + 64
        iso = torch.tensor([200], device=device_auto)
        output = sensor.simu_noise(img, iso)
        assert output.device.type == device_auto.type
