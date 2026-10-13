# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Regressions for physical design targets, clearances, and field coverage."""

import math

import pytest
import torch

from deeplens import GeoLens
from deeplens.geometric_surface import Aspheric, Spheric
from deeplens.light import Ray


@pytest.mark.parametrize("degrees", [False, True])
def test_target_fnum_uses_entrance_pupil(sample_camera_lens, degrees):
    lens = sample_camera_lens
    assert lens.aper_idx > 0
    half_fov, target_fnum = 0.4, 2.8
    lens.set_target_fov_fnum(
        math.degrees(half_fov) if degrees else half_fov, target_fnum
    )
    assert lens.rfov == pytest.approx(half_fov)
    assert lens.rfov_eff == pytest.approx(half_fov)
    assert lens.real_dfov == pytest.approx(2 * half_fov)
    assert lens.foclen == pytest.approx(lens.r_sensor / math.tan(half_fov))
    assert lens.fnum == pytest.approx(target_fnum, rel=0.01)


@pytest.mark.parametrize("initial_scale,target", [(1.0, 16.0), (0.01, 2.8)])
def test_target_fnum_brackets_large_aperture_changes(
    sample_singlet_lens, initial_scale, target
):
    lens = sample_singlet_lens
    aperture = lens.surfaces[lens.aper_idx]
    aperture.update_r(aperture.r * initial_scale)
    lens.set_target_fov_fnum(0.4, target)
    assert lens.fnum == pytest.approx(target, rel=0.002)


@pytest.mark.parametrize("failure", ["limit", "clamped", "nan", "exception"])
def test_set_fnum_restores_aperture_on_failure(
    sample_singlet_lens, monkeypatch, failure
):
    lens = sample_singlet_lens
    aperture = lens.surfaces[lens.aper_idx]
    original_r, original_fnum = aperture.r, lens.fnum
    # "clamped": a stop loaded above its own max_height must still be restored.
    limit = 0.5 * original_r if failure == "clamped" else original_r
    monkeypatch.setattr(aperture, "max_height", lambda: limit)

    def pupil():
        if failure == "exception":
            raise RuntimeError("Pupil tracing failed")
        return 0.0, float("nan") if failure == "nan" else aperture.r

    monkeypatch.setattr(lens, "calc_entrance_pupil_rayaiming", pupil)
    with pytest.raises(RuntimeError):
        lens.set_fnum(lens.foclen / (4 * original_r))
    assert aperture.r == original_r
    assert lens.fnum == original_fnum


@pytest.mark.parametrize("target", [0.0, -1.0, float("nan"), float("inf")])
def test_set_fnum_rejects_invalid_target(sample_singlet_lens, target):
    lens = sample_singlet_lens
    aperture = lens.surfaces[lens.aper_idx]
    original_r = aperture.r
    with pytest.raises(ValueError, match="finite and positive"):
        lens.set_fnum(target)
    assert aperture.r == original_r


@pytest.mark.parametrize("interior_crossing", [False, True])
def test_bounds_sample_shared_profiles_and_physical_rims(
    interior_crossing, device_auto
):
    lens = GeoLens(device=device_auto)
    ai = [-400.0, 2500.0] if interior_crossing else [0.0] * 7 + [-0.001]
    lens.surfaces = [
        Spheric(c=0.0, r=1.0, d_next=1.0, mat2="air"),
        Aspheric(c=0.0, k=0.0, ai=ai, r=1.0, d_next=10000.0, mat2="air"),
    ]
    lens.to(device_auto)
    for name in ("air_center", "air_edge", "thick_center", "thick_edge", "bfl", "ttl"):
        setattr(lens, name + "_min", 0.0)
        setattr(lens, name + "_max", 20000.0)
    lens.surfaces[0].d_next.requires_grad_(True)
    clearance, envelope = lens.loss_bound()
    lens.surfaces[0].r = 2.0
    widened = lens.loss_bound()
    torch.testing.assert_close(widened[0], clearance)
    torch.testing.assert_close(widened[1], envelope)
    assert torch.isfinite(clearance + envelope)
    if interior_crossing:
        assert clearance > 0
        clearance.backward()
        assert lens.surfaces[0].d_next.grad < 0
    else:
        assert clearance == 0
        lens.surfaces[0].c = torch.tensor(0.49, device=device_auto)
        lens.surfaces[0].r = 1.0
        clear, _ = lens.loss_bound()
        lens.surfaces[0].r = 2.0
        crossed, _ = lens.loss_bound()
        assert clear == 0
        assert crossed > clear
        crossed.backward()
        assert lens.surfaces[0].d_next.grad < 0


@pytest.mark.parametrize(
    "case",
    ["fold", "dead", "clear", "none", "first_dead", "early_dead", "early_crossing"],
)
def test_fov_uses_first_crossing_or_last_surviving_field(
    monkeypatch, caplog, case, device_auto
):
    lens = GeoLens(device=device_auto)
    lens.foclen, lens.r_sensor, lens.sensor_size = 10.0, 5.0, (8.0, 6.0)
    seen = {}

    def sample(fov_x, fov_y, num_rays):
        fov = torch.tensor(fov_y, device=device_auto)
        seen["fov"] = fov
        t = (fov - fov[0]) / (fov[-1] - fov[0])
        height = 0.5 + 0.2 * t
        if case == "fold":
            height = torch.where(t <= 0.5, 0.5 + (0.5 / 0.37) * t, 2.0 - 1.5 * t)
        elif case == "dead":
            height = 0.5 + 0.4 * torch.sin(2 * math.pi * t)
        elif case == "early_crossing":
            height = t / 0.1
        o = torch.zeros(len(fov), num_rays, 3, device=device_auto)
        o[..., 1] = lens.r_sensor * height[:, None]
        d = torch.zeros_like(o)
        d[..., 2] = 1
        ray = Ray(o, d, wvln=0.587, device=device_auto)
        if case == "dead":
            ray.is_valid[t > 0.5] = 0
        elif case == "none":
            ray.is_valid[:] = 0
        elif case == "first_dead":
            ray.is_valid[0] = 0
        elif case == "early_dead":
            ray.is_valid[t > 0.1] = 0
        return ray

    monkeypatch.setattr(lens, "sample_from_fov", sample)
    monkeypatch.setattr(lens, "trace2sensor", lambda ray: ray)
    lens.calc_fov()
    fov = seen["fov"]
    if case == "fold":
        expected = math.radians(float(fov[0] + (fov[-1] - fov[0]) * 0.37))
    elif case == "dead":
        # Peak of the sine; the surviving fields past it fold back inward.
        expected = math.radians(float(fov[16]))
    elif case == "clear":
        expected = math.radians(float(fov[-1]))
    elif case == "early_dead":
        expected = math.radians(float(fov[6]))
    elif case == "early_crossing":
        expected = math.radians(float(fov[-1] * 0.1))
    else:
        expected = 0.0
    assert fov[0] == 0.0
    assert lens.rfov == pytest.approx(expected, rel=1e-5)
    assert lens.real_dfov == pytest.approx(2 * expected)
    assert ("rfov = 0" in caplog.text) == (expected == 0.0)


def test_spot_analysis_reports_lost_field_throughput(monkeypatch, device_auto):
    lens = GeoLens(device=device_auto)
    lens.rfov = 0.4

    def sample(**kwargs):
        o = torch.zeros(3, 4, 3, device=device_auto)
        d = torch.zeros_like(o)
        d[..., 2] = 1
        ray = Ray(o, d, wvln=0.587, device=device_auto)
        ray.is_valid[1, 2:] = 0
        ray.is_valid[2] = 0
        return ray

    monkeypatch.setattr(lens, "sample_radial_rays", sample)
    monkeypatch.setattr(lens, "trace2sensor", lambda ray: ray)
    result = lens.analysis_spot()
    assert [field["throughput"] for field in result.values()] == [1.0, 0.5, 0.0]
    assert result["fov1.0"]["rms"] == 0.0
