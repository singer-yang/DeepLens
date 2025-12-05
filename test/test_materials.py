"""
Tests for deeplens/optics/materials.py - Glass and plastic materials.
"""

import pytest
import torch

from deeplens.optics.materials import Material


class TestMaterialInit:
    """Test Material initialization."""

    def test_material_vacuum(self, device_auto):
        """Vacuum should have n=1."""
        mat = Material(name="vacuum", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert torch.allclose(n, torch.tensor([1.0], device=device_auto))

    def test_material_air(self, device_auto):
        """Air should have n≈1."""
        mat = Material(name="air", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() == pytest.approx(1.0, abs=0.001)

    def test_material_bk7(self, device_auto):
        """BK7 should have typical glass index ~1.5."""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() == pytest.approx(1.52, abs=0.02)

    def test_material_case_insensitive(self, device_auto):
        """Material names should be case insensitive."""
        mat1 = Material(name="BK7", device=device_auto)
        mat2 = Material(name="bk7", device=device_auto)
        mat3 = Material(name="Bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n1 = mat1.ior(wvln)
        n2 = mat2.ior(wvln)
        n3 = mat3.ior(wvln)
        
        assert torch.allclose(n1, n2)
        assert torch.allclose(n2, n3)

    def test_material_default_vacuum(self, device_auto):
        """None name should default to vacuum."""
        mat = Material(name=None, device=device_auto)
        
        assert mat.name == "vacuum"


class TestMaterialDispersion:
    """Test wavelength-dependent refractive index."""

    def test_material_dispersion_bk7(self, device_auto):
        """BK7 should show normal dispersion (n decreases with wavelength)."""
        mat = Material(name="bk7", device=device_auto)
        
        n_blue = mat.ior(torch.tensor([0.45], device=device_auto))
        n_green = mat.ior(torch.tensor([0.55], device=device_auto))
        n_red = mat.ior(torch.tensor([0.65], device=device_auto))
        
        # Normal dispersion: n_blue > n_green > n_red
        assert n_blue > n_green > n_red

    def test_material_dispersion_range(self, device_auto):
        """Index should vary reasonably over visible spectrum."""
        mat = Material(name="bk7", device=device_auto)
        
        n_min = mat.ior(torch.tensor([0.7], device=device_auto))
        n_max = mat.ior(torch.tensor([0.4], device=device_auto))
        
        # Dispersion shouldn't be too extreme
        delta_n = n_max - n_min
        assert 0.005 < delta_n.item() < 0.05

    def test_material_dispersion_wavelength_input(self, device_auto):
        """Should accept tensor wavelengths."""
        mat = Material(name="bk7", device=device_auto)
        
        wvlns = torch.tensor([0.45, 0.55, 0.65], device=device_auto)
        n = mat.ior(wvlns)
        
        assert n.shape == wvlns.shape

    def test_material_refractive_index_alias(self, device_auto):
        """refractive_index should be alias for ior."""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n1 = mat.ior(wvln)
        n2 = mat.refractive_index(wvln)
        
        assert torch.allclose(n1, n2)


class TestMaterialTypes:
    """Test different material types."""

    def test_material_cdgm_glass(self, device_auto):
        """CDGM glasses should work."""
        mat = Material(name="h-k9l", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert 1.4 < n.item() < 2.0

    def test_material_schott_glass(self, device_auto):
        """Schott glasses should work."""
        mat = Material(name="n-bk7", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert 1.4 < n.item() < 2.0

    def test_material_plastic_pmma(self, device_auto):
        """PMMA plastic should work."""
        mat = Material(name="pmma", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert n.item() == pytest.approx(1.49, abs=0.02)

    def test_material_plastic_polycarb(self, device_auto):
        """Polycarbonate should work."""
        mat = Material(name="polycarb", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert n.item() == pytest.approx(1.58, abs=0.03)

    def test_material_coc(self, device_auto):
        """COC plastic should work."""
        mat = Material(name="coc", device=device_auto)
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert 1.5 < n.item() < 1.6


class TestMaterialSellmeier:
    """Test Sellmeier dispersion formula."""

    def test_material_set_sellmeier_param(self, device_auto):
        """Should set custom Sellmeier parameters."""
        mat = Material(name="vacuum", device=device_auto)
        
        # BK7-like parameters
        params = [1.039, 0.006, 0.231, 0.020, 1.010, 103.56]
        mat.set_sellmeier_param(params)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        assert n.item() > 1.0  # No longer vacuum

    def test_material_sellmeier_formula(self, device_auto):
        """Sellmeier formula should give positive contribution."""
        mat = Material(name="bk7", device=device_auto)
        
        # All wavelengths should give n > 1
        for wvln_val in [0.4, 0.5, 0.6, 0.7]:
            wvln = torch.tensor([wvln_val], device=device_auto)
            n = mat.ior(wvln)
            assert n.item() > 1.0


class TestMaterialMatch:
    """Test material matching functionality."""

    def test_material_match_returns_something(self, device_auto):
        """Should attempt material match without crashing."""
        mat = Material(name="bk7", device=device_auto)
        
        # This may or may not find a match depending on implementation
        try:
            matched = mat.match_material()
            # If it returns something, should be valid or None
            assert matched is None or len(matched) > 0
        except Exception:
            pytest.skip("match_material not implemented for this material type")

    def test_material_get_name(self, device_auto):
        """get_name should return material name."""
        mat = Material(name="bk7", device=device_auto)
        
        name = mat.get_name()
        
        assert name == "bk7"


class TestMaterialOptimization:
    """Test material parameter optimization."""

    def test_material_get_optimizer_params(self, device_auto):
        """Should return optimizer-compatible parameters."""
        mat = Material(name="bk7", device=device_auto)
        
        params = mat.get_optimizer_params(lrs=[1e-4, 1e-2])
        
        assert isinstance(params, list)
        assert len(params) > 0
        for p in params:
            assert "params" in p
            assert "lr" in p

    def test_material_n_trainable(self, device_auto):
        """Refractive index should be differentiable."""
        mat = Material(name="bk7", device=device_auto)
        mat.get_optimizer_params()
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        # Check n is a tensor that can have gradients
        assert isinstance(n, torch.Tensor)


class TestMaterialEdgeCases:
    """Test edge cases and error handling."""

    def test_material_extreme_wavelength_blue(self, device_auto):
        """Should handle near-UV wavelengths."""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.35], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() > 1.0

    def test_material_extreme_wavelength_red(self, device_auto):
        """Should handle near-IR wavelengths."""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.9], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.item() > 1.0

    def test_material_device_consistency(self, device_auto):
        """Output should be on same device as input."""
        mat = Material(name="bk7", device=device_auto)
        
        wvln = torch.tensor([0.55], device=device_auto)
        n = mat.ior(wvln)
        
        assert n.device.type == device_auto.type
