"""Tests for EventSensor."""

import torch
from deeplens.sensor.event_sensor import EventSensor


def test_init():
    """Test instantiation with default and custom params."""
    sensor = EventSensor()
    assert sensor.threshold_pos == 0.2
    assert sensor.threshold_neg == 0.2

    sensor2 = EventSensor(threshold_pos=0.5, threshold_neg=0.3)
    assert sensor2.threshold_pos == 0.5
    assert sensor2.threshold_neg == 0.3
    print("  [PASS] test_init")


def test_no_change():
    """Identical frames should produce zero events."""
    sensor = EventSensor(sigma_threshold=0.0)
    sensor.eval()
    img = torch.rand(1, 3, 64, 64)
    events = sensor(img, img)
    assert events.shape == (1, 2, 64, 64)
    assert events.sum().item() == 0.0
    print("  [PASS] test_no_change")


def test_step_change_positive():
    """A brightness increase should produce positive events."""
    sensor = EventSensor(threshold_pos=0.2, threshold_neg=0.2, sigma_threshold=0.0)
    sensor.eval()

    I_prev = torch.full((1, 1, 32, 32), 0.2)
    I_curr = torch.full((1, 1, 32, 32), 0.4)

    # delta = ln(0.4 + eps) - ln(0.2 + eps) ≈ ln(2) ≈ 0.693
    # expected_pos = floor(0.693 / 0.2) = 3
    events = sensor(I_curr, I_prev)
    assert events.shape == (1, 2, 32, 32)

    pos_counts = events[0, 0, 0, 0].item()
    neg_counts = events[0, 1, 0, 0].item()
    print(f"    pos={pos_counts}, neg={neg_counts}, expected pos=3, neg=0")
    assert pos_counts == 3.0
    assert neg_counts == 0.0
    print("  [PASS] test_step_change_positive")


def test_step_change_negative():
    """A brightness decrease should produce negative events."""
    sensor = EventSensor(threshold_pos=0.2, threshold_neg=0.2, sigma_threshold=0.0)
    sensor.eval()

    I_prev = torch.full((1, 1, 32, 32), 0.4)
    I_curr = torch.full((1, 1, 32, 32), 0.2)

    events = sensor(I_curr, I_prev)
    pos_counts = events[0, 0, 0, 0].item()
    neg_counts = events[0, 1, 0, 0].item()
    print(f"    pos={pos_counts}, neg={neg_counts}, expected pos=0, neg=3")
    assert pos_counts == 0.0
    assert neg_counts == 3.0
    print("  [PASS] test_step_change_negative")


def test_rgb_input():
    """Should handle 3-channel RGB input."""
    sensor = EventSensor(sigma_threshold=0.0)
    sensor.eval()

    I_prev = torch.full((2, 3, 16, 16), 0.1)
    I_curr = torch.full((2, 3, 16, 16), 0.5)
    events = sensor(I_curr, I_prev)
    assert events.shape == (2, 2, 16, 16)
    assert events[:, 0].sum() > 0  # positive events
    print("  [PASS] test_rgb_input")


def test_forward_video():
    """Video processing should produce correct shape."""
    sensor = EventSensor(sigma_threshold=0.0)
    sensor.eval()

    B, T, C, H, W = 2, 5, 1, 16, 16
    frames = torch.rand(B, T, C, H, W)
    events = sensor.forward_video(frames)
    assert events.shape == (B, T - 1, 2, H, W)
    print("  [PASS] test_forward_video")


def test_voxel_grid():
    """Voxel grid conversion should produce correct shape."""
    sensor = EventSensor(sigma_threshold=0.0)
    sensor.eval()

    events = torch.tensor([[[[3.0]], [[1.0]]]])  # (1, 2, 1, 1)
    voxel = sensor.events_to_voxel_grid(events, num_bins=5)
    assert voxel.shape == (1, 5, 1, 1)
    # net = 3 - 1 = 2, spread over 5 bins => 0.4 each
    assert abs(voxel[0, 0, 0, 0].item() - 0.4) < 1e-5
    print("  [PASS] test_voxel_grid")


def test_timestamp_image():
    """Timestamp image should be normalised to [0, 1]."""
    sensor = EventSensor()
    events = torch.tensor([[[[5.0, 2.0]], [[3.0, 0.0]]]])  # (1, 2, 1, 2)
    ts = sensor.events_to_timestamp_image(events)
    assert ts.max() <= 1.0
    assert ts.min() >= 0.0
    print("  [PASS] test_timestamp_image")


if __name__ == "__main__":
    print("Running EventSensor tests...")
    test_init()
    test_no_change()
    test_step_change_positive()
    test_step_change_negative()
    test_rgb_input()
    test_forward_video()
    test_voxel_grid()
    test_timestamp_image()
    print("\nAll tests passed!")
