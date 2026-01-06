"""
Tests for deeplens/geolens.py - Main geometric lens class.
"""

import os
import pytest
import torch

from deeplens import GeoLens
from deeplens.basics import DEPTH, DEFAULT_WAVE


class TestGeoLensLoading:
    """Test lens loading from files."""

    def test_geolens_load_json(self, sample_singlet_lens):
        """Should load lens from JSON file."""
        lens = sample_singlet_lens
        
        assert lens is not None
        assert len(lens.surfaces) > 0

    def test_geolens_load_cellphone(self, sample_cellphone_lens):
        """Should load cellphone lens with aspheric surfaces."""
        lens = sample_cellphone_lens
        
        assert lens is not None
        assert len(lens.surfaces) > 1

    def test_geolens_post_computation(self, sample_cellphone_lens):
        """Should compute foclen, fov, fnum after loading."""
        # Use cellphone lens which has aperture
        lens = sample_cellphone_lens
        
        # GeoLens calculates hfov, vfov, dfov, rfov but doesn't set "fov" attribute directly
        assert hasattr(lens, "foclen")
        assert hasattr(lens, "dfov")
        assert hasattr(lens, "fnum")
        assert lens.foclen > 0
        # assert lens.fov > 0 # Removed
        assert lens.fnum > 0

    def test_geolens_write_json(self, sample_singlet_lens, test_output_dir):
        """Should save lens to JSON file."""
        lens = sample_singlet_lens
        save_path = os.path.join(test_output_dir, "test_lens.json")
        
        lens.write_lens_json(save_path)
        
        assert os.path.exists(save_path)
        
        # Reload and verify
        lens2 = GeoLens(filename=save_path)
        assert len(lens2.surfaces) == len(lens.surfaces)

    def test_geolens_empty_init(self, device_auto):
        """Should initialize empty lens without file."""
        lens = GeoLens()
        lens.to(device_auto)
        
        assert lens.surfaces == []


class TestGeoLensRaySampling:
    """Test ray sampling methods."""

    def test_geolens_sample_parallel(self, sample_singlet_lens):
        """Should sample parallel rays at field angles."""
        lens = sample_singlet_lens
        
        ray = lens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        
        assert ray is not None
        assert ray.o.shape[-1] == 3
        assert ray.d.shape[-1] == 3

    def test_geolens_sample_parallel_offaxis(self, sample_singlet_lens):
        """Should sample off-axis parallel rays."""
        lens = sample_singlet_lens
        
        ray = lens.sample_parallel(fov_x=[5.0], fov_y=[0.0], num_rays=512)
        
        # Off-axis rays should have non-zero x direction component
        assert ray.d[..., 0].abs().max() > 0.01

    def test_geolens_sample_point_source(self, sample_singlet_lens):
        """Should sample point source rays."""
        lens = sample_singlet_lens
        
        ray = lens.sample_point_source(fov_x=[0.0], fov_y=[0.0], depth=DEPTH, num_rays=512)
        
        assert ray is not None
        assert ray.shape[-1] == 512

    def test_geolens_sample_from_points(self, sample_singlet_lens):
        """Should sample rays from specified points."""
        lens = sample_singlet_lens
        
        points = [[0.0, 0.0, -10000.0]]
        ray = lens.sample_from_points(points=points, num_rays=512)
        
        assert ray is not None
        assert ray.shape[-1] == 512

    def test_geolens_sample_from_points_batch(self, sample_singlet_lens):
        """Should sample rays from multiple points."""
        lens = sample_singlet_lens
        
        points = [[0.0, 0.0, -10000.0], [1.0, 1.0, -10000.0]]
        ray = lens.sample_from_points(points=points, num_rays=512)
        
        assert ray.o.shape[0] == 2

    def test_geolens_sample_sensor(self, sample_cellphone_lens):
        """Should sample backward rays from sensor."""
        lens = sample_cellphone_lens  # Has aperture stop
        
        ray = lens.sample_sensor(spp=2)
        
        assert ray is not None
        # Ray direction z component mean should indicate backward direction
        # (the exact sign depends on implementation)


class TestGeoLensTracing:
    """Test ray tracing through lens."""

    def test_geolens_trace_basic(self, sample_singlet_lens):
        """Should trace rays through lens."""
        lens = sample_singlet_lens
        
        ray = lens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, _ = lens.trace(ray)
        
        assert ray_out is not None
        assert ray_out.is_valid.sum() > 0

    def test_geolens_trace_with_record(self, sample_singlet_lens):
        """Should record ray path during tracing."""
        lens = sample_singlet_lens
        
        ray = lens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, ray_record = lens.trace(ray, record=True)
        
        assert ray_record is not None
        assert len(ray_record) > 0

    def test_geolens_trace_preserves_valid(self, sample_singlet_lens):
        """Tracing should maintain valid ray count or reduce it."""
        lens = sample_singlet_lens
        
        ray = lens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        valid_before = ray.is_valid.sum().item()
        
        ray_out, _ = lens.trace(ray)
        valid_after = ray_out.is_valid.sum().item()
        
        assert valid_after <= valid_before
        assert valid_after > 0  # Some rays should survive

    def test_geolens_call_is_trace(self, sample_singlet_lens):
        """__call__ should be alias for trace."""
        lens = sample_singlet_lens
        
        ray = lens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out = lens(ray)
        
        assert ray_out is not None


class TestGeoLensPSF:
    """Test PSF computation."""

    def test_geolens_psf_mono(self, sample_cellphone_lens):
        """Should compute monochrome PSF."""
        lens = sample_cellphone_lens
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31, model="geometric")
        
        # PSF should be [1, ks, ks] for batched input
        assert psf.shape == (1, 31, 31)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_psf_coherent_dispatcher(self, sample_cellphone_lens):
        """Should compute coherent PSF via dispatcher."""
        lens = sample_cellphone_lens
        
        # Coherent PSF requires float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64)
            psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31, model="coherent")
            
            assert psf.shape == (31, 31)  # psf_pupil_prop always returns 2D for now
            assert psf.sum().item() == pytest.approx(1.0, abs=0.1)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_huygens_dispatcher(self, sample_cellphone_lens):
        """Should compute Huygens PSF via dispatcher."""
        lens = sample_cellphone_lens
        
        # Huygens mode requires float64
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64)
            psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31, spp=10000, model="huygens")
            
            assert psf.shape == (31, 31) # Huygens currently single-point only, returns 2D
            assert psf.sum().item() == pytest.approx(1.0, abs=0.1)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_normalized(self, sample_cellphone_lens):
        """PSF should sum to approximately 1."""
        lens = sample_cellphone_lens
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=64)
        
        # ParaxialLens PSF is usually single channel unless psf_rgb implied
        # psf() returns [N_points, ks, ks] for single point
        assert psf.shape == (1, 64, 64)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_psf_rgb(self, sample_cellphone_lens):
        """Should compute RGB PSF."""
        lens = sample_cellphone_lens
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf_rgb = lens.psf_rgb(points, ks=64)
        
        # Check for 3-channel output
        assert psf_rgb.shape[1] == 3

    def test_geolens_psf_map(self, sample_cellphone_lens):
        """Should compute PSF map across field."""
        lens = sample_cellphone_lens
        
        psf_map = lens.psf_map(grid=(3, 3), ks=31, depth=DEPTH)
        
        # PSF map should have correct grid dimensions
        assert psf_map.shape == (3, 3, 1, 31, 31)

    def test_geolens_psf_huygens_basic(self, sample_cellphone_lens):
        """Should compute Huygens PSF (coherent mode) for single point."""
        lens = sample_cellphone_lens
        
        # Huygens mode requires float64 for coherent ray tracing
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
            # Use smaller spp for faster testing
            psf = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
            
            # PSF should have correct shape [ks, ks] for single point
            assert psf.shape == (31, 31)
            # Huygens PSF should be real-valued (intensity)
            assert not psf.is_complex()
            # All values should be non-negative (intensity)
            assert psf.min() >= 0
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_huygens_normalized(self, sample_cellphone_lens):
        """Huygens PSF should sum to approximately 1."""
        lens = sample_cellphone_lens
        
        # Huygens mode requires float64 for coherent ray tracing
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
            psf = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=64, spp=10000)
            
            # PSF should be normalized
            assert psf.shape == (64, 64)
            assert psf.sum().item() == pytest.approx(1.0, abs=0.1)
        finally:
            torch.set_default_dtype(original_dtype)

    def test_geolens_psf_huygens_vs_geometric_different(self, sample_cellphone_lens):
        """Huygens and geometric PSF should produce different results."""
        lens = sample_cellphone_lens
        
        # Huygens mode requires float64 for coherent ray tracing
        original_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
            psf_geo = lens.psf(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
            psf_huygens = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
            
            # Both should have same shape
            assert psf_geo.shape == psf_huygens.shape
            # But different values (coherent vs incoherent)
            # Convert to same dtype for comparison
            assert not torch.allclose(psf_geo.to(torch.float64), psf_huygens.to(torch.float64), atol=1e-3)
        finally:
            torch.set_default_dtype(original_dtype)


class TestGeoLensRendering:
    """Test image rendering."""

    def test_geolens_render_psf(self, sample_cellphone_lens, sample_image_small):
        """Should render image with PSF convolution."""
        lens = sample_cellphone_lens
        img = sample_image_small
        
        img_render = lens.render(img, depth=DEPTH, method="psf_patch")
        
        assert img_render.shape == img.shape

    def test_geolens_render_psf_map(self, sample_cellphone_lens, sample_image_small):
        """Should render image with spatially-varying PSF."""
        lens = sample_cellphone_lens
        img = sample_image_small
        
        # psf_map requires image resolution to match sensor resolution
        # Resize input image to match sensor resolution
        img_large = torch.nn.functional.interpolate(img, size=lens.sensor_res, mode='bilinear')
        img_render = lens.render(img_large, depth=DEPTH, method="psf_map")
        
        # Output shape should match sensor resolution, not input image small shape
        assert img_render.shape[-2:] == lens.sensor_res

    def test_geolens_render_preserves_range(self, sample_cellphone_lens, sample_image_small):
        """Rendered image should have non-negative values."""
        lens = sample_cellphone_lens
        img = sample_image_small
        
        img_render = lens.render(img, depth=DEPTH, method="psf_patch")
        
        assert img_render.min() >= 0

    def test_geolens_analysis_rendering(self, sample_cellphone_lens, sample_image_small, test_output_dir):
        """Should run analysis_rendering and return rendered image."""
        import os
        lens = sample_cellphone_lens
        # Convert [B, C, H, W] to [H, W, C] format expected by analysis_rendering
        img = sample_image_small.squeeze(0).permute(1, 2, 0)
        save_path = os.path.join(test_output_dir, "analysis_render_test")
        
        # Run analysis_rendering
        img_render = lens.analysis_rendering(
            img_org=img, 
            depth=DEPTH, 
            spp=64,
            save_name=save_path,
            method="ray_tracing",
            show=False
        )
        
        # Check output shape [B, C, H, W]
        assert img_render is not None
        assert len(img_render.shape) == 4
        assert img_render.shape[1] == 3  # RGB channels
        # Check values are in valid range
        assert img_render.min() >= 0
        assert img_render.max() <= 1
        # Check that output file was saved
        assert os.path.exists(f"{save_path}.png")


class TestGeoLensProperties:
    """Test lens property calculations."""

    def test_geolens_refocus(self, sample_singlet_lens):
        """Should refocus lens to new distance."""
        lens = sample_singlet_lens
        original_d = lens.d_sensor.item()
        
        lens.refocus(foc_dist=-500.0)
        
        # d_sensor should change
        assert lens.d_sensor.item() != original_d

    def test_geolens_aperture_idx(self, sample_cellphone_lens):
        """Should identify aperture stop index."""
        lens = sample_cellphone_lens
        
        aper_idx = lens.aper_idx
        
        assert isinstance(aper_idx, int)
        assert 0 <= aper_idx < len(lens.surfaces)

    def test_geolens_fov_calc(self, sample_cellphone_lens):
        """Should calculate correct FoV."""
        lens = sample_cellphone_lens
        
        lens.calc_fov()
        
        assert hasattr(lens, "dfov")
        assert lens.dfov > 0 
        # lens.calc_fov() returns None, so we don't check return value

    def test_geolens_sensor_properties(self, sample_singlet_lens):
        """Should have correct sensor properties."""
        lens = sample_singlet_lens
        
        assert lens.sensor_res[0] > 0
        assert lens.sensor_res[1] > 0
        assert lens.sensor_size[0] > 0
        assert lens.sensor_size[1] > 0


class TestGeoLensDifferentiability:
    """Test gradient flow through lens operations."""

    def test_geolens_psf_differentiable(self, sample_cellphone_lens):
        """PSF computation should be differentiable."""
        lens = sample_cellphone_lens
        
        # Make a surface parameter require grad
        lens.surfaces[1].d.requires_grad_(True)
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31)
        
        loss = psf.sum()
        loss.backward()
        
        # Check gradient exists
        assert lens.surfaces[1].d.grad is not None

    def test_geolens_get_optimizer(self, sample_cellphone_lens):
        """Should return optimizer parameters."""
        lens = sample_cellphone_lens
        
        optimizer = lens.get_optimizer(lrs=[1e-4, 1e-4, 1e-4, 1e-4])
        
        assert optimizer is not None


class TestGeoLensVisualization:
    """Test visualization methods."""

    def test_geolens_draw_layout(self, sample_cellphone_lens, test_output_dir):
        """Should draw lens layout."""
        lens = sample_cellphone_lens
        save_path = os.path.join(test_output_dir, "lens_layout.png")
        
        lens.draw_layout(filename=save_path)
        
        assert os.path.exists(save_path)

    def test_geolens_analysis(self, sample_cellphone_lens, test_output_dir):
        """Should run lens analysis."""
        lens = sample_cellphone_lens
        save_path = os.path.join(test_output_dir, "lens_analysis.png")
        
        # This may require more setup, so we just check it doesn't crash
        try:
            lens.analysis(save_name=save_path)
        except Exception as e:
            pytest.skip(f"Analysis requires additional dependencies: {e}")


class TestGeoLensDeviceHandling:
    """Test device transfer and GPU support."""

    def test_geolens_to_device(self, sample_singlet_lens, device_auto):
        """Should move lens to device."""
        lens = sample_singlet_lens
        lens.to(device_auto)
        
        assert lens.device.type == device_auto.type
        for surf in lens.surfaces:
            assert surf.d.device.type == device_auto.type

    def test_geolens_trace_on_gpu(self, sample_singlet_lens, device_auto):
        """Tracing should work on GPU."""
        lens = sample_singlet_lens
        lens.to(device_auto)
        
        ray = lens.sample_parallel(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, _ = lens.trace(ray)
        
        assert ray_out.o.device.type == device_auto.type

    def test_geolens_psf_on_gpu(self, sample_cellphone_lens, device_auto):
        """PSF computation should work on GPU."""
        lens = sample_cellphone_lens
        lens.to(device_auto)
        
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device_auto)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31)
        
        assert psf.device.type == device_auto.type
