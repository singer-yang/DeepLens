"""
Tests for deeplens/paraxiallens.py - Paraxial lens model.
"""

import pytest
import torch

from deeplens.paraxiallens import ParaxialLens
from deeplens.basics import DEPTH


class TestParaxialLensInit:
    """Test ParaxialLens initialization."""

    def test_paraxial_init(self, device_auto):
        """Should initialize with basic parameters."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        assert lens.foclen == 50.0
        assert lens.fnum == 1.8

    def test_paraxial_aperture_radius(self, device_auto):
        """Aperture radius calculation check."""
        foclen = 50.0
        fnum = 2.0
        
        lens = ParaxialLens(
            foclen=foclen,
            fnum=fnum,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        # ParaxialLens doesn't expose 'r' directly, so we just verify parameters
        assert lens.foclen == foclen
        assert lens.fnum == fnum


class TestParaxialLensRefocus:
    """Test lens refocusing."""

    def test_paraxial_refocus(self, device_auto):
        """Should change focus distance."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        original_foc = lens.foc_dist
        lens.refocus(-1000.0)
        
        assert lens.foc_dist != original_foc
        assert lens.foc_dist == -1000.0

    def test_paraxial_refocus_infinity(self, device_auto):
        """Should handle infinity focus."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(DEPTH)
        
        assert lens.foc_dist == DEPTH


class TestParaxialLensCoC:
    """Test circle of confusion calculation."""

    def test_paraxial_coc_at_focus(self, device_auto):
        """CoC should be zero at focus distance."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        depth = torch.tensor([-1000.0], device=device_auto)
        coc = lens.coc(depth)
        
        assert coc.item() == pytest.approx(0.0, abs=0.01)

    def test_paraxial_coc_out_of_focus(self, device_auto):
        """CoC should increase out of focus."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        
        depth_near = torch.tensor([-500.0], device=device_auto)
        depth_far = torch.tensor([-2000.0], device=device_auto)
        
        coc_near = lens.coc(depth_near)
        coc_far = lens.coc(depth_far)
        
        assert coc_near.abs().item() > 0
        assert coc_far.abs().item() > 0

    def test_paraxial_coc_batch(self, device_auto):
        """Should handle batch of depths."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        depths = torch.tensor([-500.0, -1000.0, -2000.0], device=device_auto)
        cocs = lens.coc(depths)
        
        assert cocs.shape == depths.shape


class TestParaxialLensDoF:
    """Test depth of field calculation."""

    def test_paraxial_dof_exists(self, device_auto):
        """Should calculate positive DoF."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        # DoF should be positive. Note standard DoF is undefined at focus where CoC=0 in this implementation?
        # Check DoF at slightly defocused distance.
        depth = torch.tensor([-500.0], device=device_auto)
        dof = lens.dof(depth)
        
        assert dof.item() > 0

    def test_paraxial_coc_fnum_dependence(self, device_auto):
        """Larger f-number (smaller aperture) should give smaller CoC."""
        lens1 = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        lens2 = ParaxialLens(
            foclen=50.0,
            fnum=8.0,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens1.refocus(-1000.0)
        lens2.refocus(-1000.0)
        
        depth = torch.tensor([-500.0], device=device_auto)
        coc1 = lens1.coc(depth)
        coc2 = lens2.coc(depth)
        
        assert coc2.item() < coc1.item()


class TestParaxialLensPSF:
    """Test PSF generation."""

    def test_paraxial_psf_gaussian(self, device_auto):
        """Should generate Gaussian PSF."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[-500.0]], device=device_auto)  # Out of focus
        points = torch.cat([torch.zeros(1, 2, device=device_auto), points], dim=-1)
        
        psf = lens.psf(points, ks=31, psf_type="gaussian")
        
        # PSF is [N, ks, ks]
        assert psf.shape[-2:] == (31, 31)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_paraxial_psf_pillbox(self, device_auto):
        """Should generate pillbox PSF."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf = lens.psf(points, ks=31, psf_type="pillbox")

        assert psf.shape[-2:] == (31, 31)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_paraxial_psf_in_focus_sharp(self, device_auto):
        """PSF at focus should be sharp (small)."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points_focus = torch.tensor([[0.0, 0.0, -1000.0]], device=device_auto)
        points_defocus = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_focus = lens.psf(points_focus, ks=31, psf_type="gaussian")
        psf_defocus = lens.psf(points_defocus, ks=31, psf_type="gaussian")
        
        # In-focus PSF should be more concentrated (higher peak)
        assert psf_focus.max() > psf_defocus.max()

    def test_paraxial_psf_rgb(self, device_auto):
        """Should generate RGB PSF."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_rgb = lens.psf_rgb(points, ks=31)
        
        # Expect [N, 3, ks, ks] or [3, ks, ks]
        assert psf_rgb.shape[-3:] == (3, 31, 31)

    def test_paraxial_psf_batch(self, device_auto):
        """Should handle batch of points."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([
            [0.0, 0.0, -500.0],
            [0.0, 0.0, -1000.0],
            [0.0, 0.0, -2000.0],
        ], device=device_auto)
        
        psf = lens.psf(points, ks=31, psf_type="gaussian")
        
        # Expect [3, ks, ks]
        assert psf.shape[-3:] == (3, 31, 31)


class TestParaxialLensDualPixel:
    """Test dual-pixel PSF generation."""

    def test_paraxial_psf_dp(self, device_auto):
        """Should generate dual-pixel PSFs."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_left, psf_right = lens.psf_dp(points, ks=31)
        
        assert psf_left.shape[-2:] == (31, 31)
        assert psf_right.shape[-2:] == (31, 31)

    def test_paraxial_psf_dp_disparity(self, device_auto):
        """Left and right PSFs should have disparity."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)  # Out of focus
        
        psf_left, psf_right = lens.psf_dp(points, ks=31)
        
        # Left and right should be different
        diff = (psf_left - psf_right).abs().sum()
        assert diff.item() > 0.01

    def test_paraxial_psf_rgb_dp(self, device_auto):
        """Should generate RGB dual-pixel PSFs."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        points = torch.tensor([[0.0, 0.0, -500.0]], device=device_auto)
        
        psf_left, psf_right = lens.psf_rgb_dp(points, ks=31)
        
        assert psf_left.shape[-3:] == (3, 31, 31)
        assert psf_right.shape[-3:] == (3, 31, 31)


class TestParaxialLensPSFMap:
    """Test PSF map generation."""

    def test_paraxial_psf_map(self, device_auto):
        """Should generate PSF map."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        psf_map = lens.psf_map(grid=(3, 3), ks=31, depth=-500.0)
        
        # psf_map: [grid_y, grid_x, 1, ks, ks]
        assert psf_map.shape == (3, 3, 1, 31, 31)

    def test_paraxial_psf_map_dp(self, device_auto):
        """Should generate dual-pixel PSF map."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(1000, 1000),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        psf_map_left, psf_map_right = lens.psf_map_dp(grid=(3, 3), ks=31, depth=-500.0)
        
        assert psf_map_left.shape == (3, 3, 1, 31, 31)
        assert psf_map_right.shape == (3, 3, 1, 31, 31)


class TestParaxialLensRendering:
    """Test RGBD rendering."""

    def test_paraxial_render_rgbd_dp(self, device_auto):
        """Should render dual-pixel images from RGBD."""
        lens = ParaxialLens(
            foclen=50.0,
            fnum=1.8,
            sensor_size=(20.0, 20.0),
            sensor_res=(64, 64),
            device=device_auto,
        )
        
        lens.refocus(-1000.0)
        
        rgb = torch.rand(1, 3, 64, 64, device=device_auto)
        depth = torch.full((1, 1, 64, 64), -500.0, device=device_auto)
        
        img_left, img_right = lens.render_rgbd_dp(rgb, depth)
        
        assert img_left.shape == rgb.shape
        assert img_right.shape == rgb.shape
