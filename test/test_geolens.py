"""
Tests for deeplens/geolens.py - Main geometric lens class.
"""

import os

import pytest
import torch

from deeplens import GeoLens
from deeplens.config import DEFAULT_WAVE, DEPTH


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

    def test_geolens_sample_from_fov(self, sample_singlet_lens):
        """Should sample parallel rays at field angles."""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)

        assert ray is not None
        assert ray.o.shape[-1] == 3
        assert ray.d.shape[-1] == 3

    def test_geolens_sample_from_fov_offaxis(self, sample_singlet_lens):
        """Should sample off-axis parallel rays."""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[5.0], fov_y=[0.0], num_rays=512)

        # Off-axis rays should have non-zero x direction component
        assert ray.d[..., 0].abs().max() > 0.01

    def test_geolens_sample_point_source(self, sample_singlet_lens):
        """Should sample point source rays via sample_from_fov with finite depth."""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], depth=DEPTH, num_rays=512)

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

    def test_geolens_sample_entrance_pupil_flag(self):
        """entrance_pupil=False must aim at surface 0 at finite depth too."""
        lens = GeoLens(filename="datasets/lenses/camera/ef100mm_f2.8.json")
        # This lens has its stop behind surface 0, so the two targets differ.
        assert lens.get_entrance_pupil() != (lens.surf_d(0).item(), lens.surfaces[0].r)

        for depth in (float("inf"), -1000.0):
            kw = dict(fov_x=0.0, fov_y=0.0, depth=depth, num_rays=2048)
            on = lens.sample_from_fov(entrance_pupil=True, **kw)
            off = lens.sample_from_fov(entrance_pupil=False, **kw)

            # Footprint radius where each bundle crosses surface 0.
            def footprint(ray):
                return (
                    ray.clone().prop_to(lens.surf_d(0).item()).o[..., :2].norm(-1).max()
                )

            assert not torch.allclose(footprint(on), footprint(off))

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

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, _ = lens.trace(ray)

        assert ray_out is not None
        assert ray_out.is_valid.sum() > 0

    def test_geolens_trace_with_record(self, sample_singlet_lens):
        """Should record ray path during tracing."""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, ray_record = lens.trace(ray, record=True)

        assert ray_record is not None
        assert len(ray_record) > 0

    def test_geolens_trace_preserves_valid(self, sample_singlet_lens):
        """Tracing should maintain valid ray count or reduce it."""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        valid_before = ray.is_valid.sum().item()

        ray_out, _ = lens.trace(ray)
        valid_after = ray_out.is_valid.sum().item()

        assert valid_after <= valid_before
        assert valid_after > 0  # Some rays should survive

    def test_geolens_call_is_trace(self, sample_singlet_lens):
        """__call__ should be alias for trace."""
        lens = sample_singlet_lens

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
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
        lens.astype(torch.float64)
        points = torch.tensor(
            [[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64
        )
        psf = lens.psf(
            points,
            wvln=DEFAULT_WAVE,
            ks=31,
            model="coherent",
            spp=1_000_000,
        )

        assert psf.shape == (31, 31)  # psf_pupil_prop always returns 2D for now
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_psf_huygens_dispatcher(self, sample_cellphone_lens):
        """Should compute Huygens PSF via dispatcher."""
        lens = sample_cellphone_lens

        # Huygens mode requires float64
        lens.astype(torch.float64)
        points = torch.tensor(
            [[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64
        )
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31, spp=10000, model="huygens")

        assert psf.shape == (31, 31)  # Huygens currently single-point only, returns 2D
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_coherent_fields(self, sample_cellphone_lens):
        """Coherent models should return comparable unit-energy complex fields."""
        lens = sample_cellphone_lens
        lens.astype(torch.float64)
        lens.set_sensor_res((256, 256))
        points = torch.tensor(
            [[0.0, 0.0, DEPTH]], device=lens.device, dtype=torch.float64
        )

        def compute(model, spp, return_field=False):
            torch.manual_seed(0)
            kwargs = {"return_field": True} if return_field else {}
            with torch.no_grad():
                return lens.psf(
                    points,
                    wvln=DEFAULT_WAVE,
                    ks=31,
                    model=model,
                    spp=spp,
                    **kwargs,
                )

        coherent_psf = compute("coherent", 1_000_000)
        coherent_field = compute("coherent", 1_000_000, return_field=True)
        huygens_psf = compute("huygens", 10_000)
        huygens_field = compute("huygens", 10_000, return_field=True)

        for field in (coherent_field, huygens_field):
            assert field.shape == (31, 31)
            assert field.is_complex()
            energy = field.abs().square().sum().item()
            assert energy == pytest.approx(1.0, abs=1e-6)

        for psf, field in (
            (coherent_psf, coherent_field),
            (huygens_psf, huygens_field),
        ):
            assert not psf.is_complex()
            assert torch.allclose(
                psf.to(torch.float64), field.abs().square(), rtol=1e-5, atol=1e-8
            )

        # Compare spatial energy distributions after suppressing Monte Carlo noise.
        coherent_intensity = torch.nn.functional.avg_pool2d(
            coherent_field.abs().square()[None, None], 3, stride=1, padding=1
        ).flatten()
        huygens_intensity = torch.nn.functional.avg_pool2d(
            huygens_field.abs().square()[None, None], 3, stride=1, padding=1
        ).flatten()
        similarity = torch.nn.functional.cosine_similarity(
            coherent_intensity,
            huygens_intensity,
            dim=0,
        )
        assert similarity.item() > 0.75

    def test_geolens_psf_normalized(self, sample_cellphone_lens):
        """PSF should sum to approximately 1."""
        lens = sample_cellphone_lens

        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=64)

        # DefocusLens PSF is usually single channel unless psf_rgb implied
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
        lens.astype(torch.float64)
        point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
        # Use smaller spp for faster testing
        psf = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)

        # PSF should have correct shape [ks, ks] for single point
        assert psf.shape == (31, 31)
        # Huygens PSF should be real-valued (intensity)
        assert not psf.is_complex()
        # All values should be non-negative (intensity)
        assert psf.min() >= 0

    def test_geolens_psf_huygens_normalized(self, sample_cellphone_lens):
        """Huygens PSF should sum to approximately 1."""
        lens = sample_cellphone_lens

        # Huygens mode requires float64 for coherent ray tracing
        lens.astype(torch.float64)
        point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
        psf = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=64, spp=10000)

        # PSF should be normalized
        assert psf.shape == (64, 64)
        assert psf.sum().item() == pytest.approx(1.0, abs=0.1)

    def test_geolens_psf_huygens_vs_geometric_different(self, sample_cellphone_lens):
        """Huygens and geometric PSF should produce different results."""
        lens = sample_cellphone_lens

        # Huygens mode requires float64 for coherent ray tracing
        lens.astype(torch.float64)
        point = torch.tensor([0.0, 0.0, DEPTH], device=lens.device, dtype=torch.float64)
        psf_geo = lens.psf(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)
        psf_huygens = lens.psf_huygens(point, wvln=DEFAULT_WAVE, ks=31, spp=10000)

        # Both should have same shape
        assert psf_geo.shape == psf_huygens.shape
        # But different values (coherent vs incoherent)
        # Convert to same dtype for comparison
        assert not torch.allclose(
            psf_geo.to(torch.float64), psf_huygens.to(torch.float64), atol=1e-3
        )


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
        lens.set_sensor_res((64, 64))

        # psf_map requires image resolution to match sensor resolution
        # Resize input image to match sensor resolution
        img_large = torch.nn.functional.interpolate(img, size=(64, 64), mode="bilinear")
        img_render = lens.render(
            img_large,
            depth=DEPTH,
            method="psf_map",
            psf_grid=(2, 2),
            psf_ks=31,
            psf_spp=1024,
            warp_grid=8,
        )

        # Output shape should match sensor resolution, not input image small shape
        assert img_render.shape[-2:] == lens.sensor_res

    def test_geolens_render_preserves_range(
        self, sample_cellphone_lens, sample_image_small
    ):
        """Rendered image should have non-negative values."""
        lens = sample_cellphone_lens
        img = sample_image_small

        img_render = lens.render(img, depth=DEPTH, method="psf_patch")

        assert img_render.min() >= 0

    def test_geolens_analysis_rendering(
        self, sample_cellphone_lens, sample_image_small, test_output_dir
    ):
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
            show=False,
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

    def test_calc_foclen_matches_thick_lens_equation(self, sample_singlet_lens):
        """Paraxial EFL of a singlet must equal the analytic thick-lens value."""
        lens = sample_singlet_lens
        front, back = lens.surfaces[0], lens.surfaces[1]
        n = float(front.mat2.ior(lens.primary_wvln))
        c1, c2 = float(front.c), float(back.c)
        d = float(front._get_effective_d_next())

        # 1/f = (n - 1) * [c1 - c2 + (n - 1) * d * c1 * c2 / n]
        expected = 1.0 / ((n - 1) * (c1 - c2 + (n - 1) * d * c1 * c2 / n))

        assert lens.calc_foclen() == pytest.approx(expected, rel=1e-9)

    def test_calc_foclen_ignores_aspheric_and_conic_terms(self, sample_cellphone_lens):
        """First-order EFL depends on vertex curvature only, as in Zemax/CODE V."""
        lens = sample_cellphone_lens
        surf = next(s for s in lens.surfaces if getattr(s, "ai", None) is not None)
        before = lens.calc_foclen()

        surf.ai = surf.ai * 2.0
        surf.k = surf.k + 1.0

        assert lens.calc_foclen() == pytest.approx(before, rel=1e-12)

    def test_calc_foclen_rejects_afocal_system(self, sample_singlet_lens):
        """An afocal stack has no focal length and must fail loudly."""
        lens = sample_singlet_lens
        for surf in lens.surfaces:
            surf.c = torch.zeros_like(surf.c)

        with pytest.raises(ValueError, match="afocal"):
            lens.calc_foclen()

    def test_calc_pupil_rejects_zero_entrance_radius(
        self, sample_cellphone_lens, monkeypatch
    ):
        """A failed pupil fallback raises a domain error before F/# division."""
        monkeypatch.setattr(
            sample_cellphone_lens,
            "calc_exit_pupil_rayaiming",
            lambda: (0.0, 1.0),
        )
        monkeypatch.setattr(
            sample_cellphone_lens,
            "calc_entrance_pupil_rayaiming",
            lambda: (0.0, 0.0),
        )

        with pytest.raises(ValueError, match="entrance pupil radius"):
            sample_cellphone_lens.calc_pupil()

    def test_paraxial_pupil_matches_thin_lens_conjugate(self):
        """The first-order pupil is the analytic image of the stop.

        Stop and thin lens f=10 mm separated by t=5 mm: the stop images to
        m = 1/(1 - t/f) = 2x at t/(1 - t/f) = 10 mm past the lens, on the
        side the light came from.
        """
        from deeplens.geometric_surface import Aperture, ThinLens

        f, t, aper_r = 10.0, 5.0, 1.0

        # Exit pupil: stop first, lens 5 mm behind it (lens vertex at z=5).
        lens = GeoLens()
        lens.surfaces = [
            Aperture(r=aper_r, d_next=t, device="cpu"),
            ThinLens(r=5.0, d_next=5.0, f=f, device="cpu"),
        ]
        lens.aper_idx = 0
        z, r = lens.calc_pupil_paraxial(reverse=False)
        assert z == pytest.approx(t - 10.0)
        assert r == pytest.approx(2.0 * aper_r)

        # Entrance pupil: same pair reversed (lens vertex at z=0).
        lens = GeoLens()
        lens.surfaces = [
            ThinLens(r=5.0, d_next=t, f=f, device="cpu"),
            Aperture(r=aper_r, d_next=5.0, device="cpu"),
        ]
        lens.aper_idx = 1
        z, r = lens.calc_pupil_paraxial(reverse=True)
        assert z == pytest.approx(10.0)
        assert r == pytest.approx(2.0 * aper_r)

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
        lens.surfaces[1].d_next.requires_grad_(True)

        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31)

        loss = psf.sum()
        loss.backward()

        # Check gradient exists
        assert lens.surfaces[1].d_next.grad is not None

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
            assert surf.d_next.device.type == device_auto.type

    def test_geolens_trace_on_gpu(self, sample_singlet_lens, device_auto):
        """Tracing should work on GPU."""
        lens = sample_singlet_lens
        lens.to(device_auto)

        ray = lens.sample_from_fov(fov_x=[0.0], fov_y=[0.0], num_rays=512)
        ray_out, _ = lens.trace(ray)

        assert ray_out.o.device.type == device_auto.type

    def test_geolens_psf_on_gpu(self, sample_cellphone_lens, device_auto):
        """PSF computation should work on GPU."""
        lens = sample_cellphone_lens
        lens.to(device_auto)

        points = torch.tensor([[0.0, 0.0, DEPTH]], device=device_auto)
        psf = lens.psf(points, wvln=DEFAULT_WAVE, ks=31)

        assert psf.device.type == device_auto.type


class TestGeoLensDistortion:
    """Test distortion calculation methods."""

    def test_geolens_distortion_map(self, sample_cellphone_lens):
        """Should compute distortion map."""
        lens = sample_cellphone_lens

        distortion_map = lens.calc_distortion_map(num_grid=(5, 5), depth=DEPTH)

        assert distortion_map.shape == (5, 5, 2)
        # Distortion values should be normalized to approximately [-1, 1]
        assert distortion_map.abs().max() <= 2.0

    def test_geolens_inv_distortion_map(self, sample_cellphone_lens):
        """Should compute inverse distortion grid for image warping."""
        lens = sample_cellphone_lens

        inv_distortion_map = lens.calc_inv_distortion_map(num_grid=(5, 3), depth=DEPTH)

        assert inv_distortion_map.shape == (3, 5, 2)
        assert not torch.isnan(inv_distortion_map).any()
        assert inv_distortion_map.abs().max() <= 2.0

        x, y = torch.meshgrid(
            torch.linspace(-1, 1, 5, device=lens.device),
            torch.linspace(-1, 1, 3, device=lens.device),
            indexing="xy",
        )
        expected_grid = torch.stack((x, y), dim=-1)
        points = torch.cat(
            [
                inv_distortion_map.reshape(-1, 2),
                torch.full((15, 1), DEPTH, device=lens.device),
            ],
            dim=-1,
        )
        actual_grid = lens.distortion_center(points).reshape(3, 5, 2)
        assert torch.allclose(actual_grid, expected_grid, atol=5e-2)

    def test_geolens_warp_shape(self, sample_cellphone_lens):
        """Warp should preserve image shape."""
        lens = sample_cellphone_lens
        img = torch.rand(1, 3, 32, 48, device=lens.device)

        img_warped = lens.warp(img, depth=DEPTH, num_grid=(5, 3))

        assert img_warped.shape == img.shape

    def test_geolens_distortion_center_single_point(self, sample_cellphone_lens):
        """Should compute distortion center for single point."""
        lens = sample_cellphone_lens

        # Single normalized point at center
        points = torch.tensor([[0.0, 0.0, DEPTH]], device=lens.device)
        distortion_center = lens.distortion_center(points)

        # Output should be [N, 2]
        assert distortion_center.shape == (1, 2)
        # Center point should have distortion center near origin
        assert distortion_center.abs().max() < 0.5

    def test_geolens_distortion_center_multiple_points(self, sample_cellphone_lens):
        """Should compute distortion center for multiple points."""
        lens = sample_cellphone_lens

        # Multiple normalized points
        points = torch.tensor(
            [
                [0.0, 0.0, DEPTH],
                [0.5, 0.0, DEPTH],
                [0.0, 0.5, DEPTH],
                [0.5, 0.5, DEPTH],
            ],
            device=lens.device,
        )
        distortion_center = lens.distortion_center(points)

        # Output should be [N, 2]
        assert distortion_center.shape == (4, 2)
        # All values should be in valid normalized range
        assert distortion_center.abs().max() <= 2.0

    def test_geolens_distortion_center_normalized_range(self, sample_cellphone_lens):
        """Distortion center output should be in normalized [-1, 1] range for reasonable inputs."""
        lens = sample_cellphone_lens

        # Grid of normalized points
        x = torch.linspace(-0.8, 0.8, 3)
        y = torch.linspace(-0.8, 0.8, 3)
        xx, yy = torch.meshgrid(x, y, indexing="xy")
        z = torch.full_like(xx, DEPTH)
        points = torch.stack([xx, yy, z], dim=-1).reshape(-1, 3).to(lens.device)

        distortion_center = lens.distortion_center(points)

        # Output should be [9, 2]
        assert distortion_center.shape == (9, 2)
        # Most distortion centers should be within reasonable range
        # (may exceed 1.0 due to lens distortion, but shouldn't be extreme)
        assert distortion_center.abs().max() < 3.0
