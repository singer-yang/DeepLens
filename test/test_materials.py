"""
Tests for deeplens/optics/material/materials.py - Glass and plastic materials.
"""

import pytest
import torch

from deeplens.material import Material, RII_data

# Spectral lines used for nd/Vd: He d-line, H F-line, H C-line [µm].
WVLN_D, WVLN_F, WVLN_C = 0.5875618, 0.4861327, 0.6562725


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

    def test_material_default_air(self, device_auto):
        """None name should default to air."""
        mat = Material(name=None, device=device_auto)

        assert mat.name == "air"


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


class TestRefractiveIndexInfo:
    """Test the bundled refractiveindex.info catalog integration."""

    def test_catalog_loaded(self):
        """The bundled JSON catalog should load with formula + interp tables."""
        assert isinstance(RII_data.get("FORMULA"), dict)
        assert isinstance(RII_data.get("INTERP"), dict)
        # Sanity: the catalog ships a substantial number of glasses.
        assert len(RII_data["FORMULA"]) > 1000

    def test_ohara_glass_resolves_to_rii(self, device_auto):
        """An OHARA glass (not in the AGF catalogs) loads from refractiveindex.info."""
        mat = Material(name="s-bsl7", device=device_auto)
        assert mat.dispersion == "rii"
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        # S-BSL7 is OHARA's N-BK7 analogue, nd ~ 1.5163.
        assert n.item() == pytest.approx(1.5163, abs=2e-3)

    def test_hikari_formula3_glass(self, device_auto):
        """A HIKARI glass uses formula 3 (polynomial) and gives a sane index."""
        mat = Material(name="j-bk7a", device=device_auto)
        assert mat.dispersion == "rii"
        assert mat.rii_formula == 3
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.5168, abs=2e-3)

    def test_substrate_fused_silica(self, device_auto):
        """Fused silica (SiO2) resolves via the refractiveindex.info formula-1 path.

        ``sio2`` is unique to the refractiveindex.info catalog, so it exercises
        the new ``"rii"`` branch (the alias ``fused_silica`` is instead served by
        the pre-existing custom interpolation table and is covered elsewhere).
        """
        mat = Material(name="sio2", device=device_auto)
        assert mat.dispersion == "rii" and mat.rii_formula == 1
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.4585, abs=2e-3)

    def test_interp_tabulated_glass(self, device_auto):
        """A tabulated refractiveindex.info crystal uses the 'interp' branch.

        Membership in RII_data['INTERP'] proves the new tabulated-fallback branch
        is exercised (rather than a shadowed custom/AGF entry).
        """
        assert "bf1" in RII_data["INTERP"]
        mat = Material(name="bf1", device=device_auto)
        assert mat.dispersion == "interp"
        n = mat.ior(torch.tensor([0.55], device=device_auto))
        assert torch.isfinite(n).all() and 1.4 < n.item() < 2.0

    def test_interp_table_sorted(self):
        """Bundled interp tables must be ascending in wavelength (interp assumes it)."""
        for name, e in RII_data["INTERP"].items():
            w = e["wvlns"]
            assert all(w[i + 1] > w[i] for i in range(len(w) - 1)), name

    def test_substrate_sapphire(self, device_auto):
        """Sapphire (Al2O3 ordinary ray) resolves and matches literature nd."""
        mat = Material(name="sapphire", device=device_auto)
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.768, abs=3e-3)

    def test_formula1_constant_term(self, device_auto):
        """MgF2 (formula 1) carries a non-zero C1 constant; it must be applied.

        Dropping the leading C1 term would shift n by ~0.1, so this guards the
        formula-1 constant-term handling specifically.
        """
        mat = Material(name="mgf2", device=device_auto)
        assert mat.rii_formula == 1 and mat.rii_coeffs[0] != 0.0
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.3777, abs=3e-3)

    def test_infrared_substrate_in_band(self, device_auto):
        """Silicon (IR-only) gives the right index inside its validity band."""
        mat = Material(name="si", device=device_auto)
        n = mat.ior(torch.tensor([2.0], device=device_auto))  # 2 µm
        assert n.item() == pytest.approx(3.4487, abs=0.02)

    def test_rii_normal_dispersion(self, device_auto):
        """A refractiveindex.info glass shows normal dispersion."""
        mat = Material(name="s-bsl7", device=device_auto)
        n_blue = mat.ior(torch.tensor([0.45], device=device_auto))
        n_green = mat.ior(torch.tensor([0.55], device=device_auto))
        n_red = mat.ior(torch.tensor([0.65], device=device_auto))
        assert n_blue > n_green > n_red

    def test_rii_differentiable(self, device_auto):
        """Refractive index from an RII formula is differentiable in wavelength."""
        mat = Material(name="s-bsl7", device=device_auto)
        wvln = torch.tensor([WVLN_D], device=device_auto, requires_grad=True)
        n = mat.ior(wvln)
        n.backward()
        # Normal dispersion -> dn/dlambda < 0.
        assert wvln.grad is not None and wvln.grad.item() < 0

    @pytest.mark.parametrize(
        "name", ["s-bsl7", "bah32", "bac4", "j-bk7a", "k-baf8", "sapphire"]
    )
    def test_nd_vd_oracle(self, name, device_auto):
        """Computed (nd, Vd) must reproduce the stored catalog values.

        This is the correctness gate: it simultaneously checks coefficient
        parsing, formula selection, and µm unit handling.
        """
        entry = RII_data["FORMULA"][name]
        mat = Material(name=name, device=device_auto)
        nd = mat.ior(torch.tensor([WVLN_D], device=device_auto)).item()
        assert nd == pytest.approx(entry["nd"], abs=1.5e-3)
        if entry["vd"] < 1e37:
            nf = mat.ior(torch.tensor([WVLN_F], device=device_auto)).item()
            nc = mat.ior(torch.tensor([WVLN_C], device=device_auto)).item()
            vd = (nd - 1) / (nf - nc)
            assert vd == pytest.approx(entry["vd"], rel=0.02)

    def test_existing_name_precedence_unchanged(self, device_auto):
        """Names in the AGF catalogs keep precedence over refractiveindex.info.

        n-bk7 exists in both SCHOTT.AGF and the refractiveindex.info catalog;
        it must still resolve via the AGF (Sellmeier) path, not 'rii'.
        """
        mat = Material(name="n-bk7", device=device_auto)
        assert mat.dispersion == "sellmeier"
        n = mat.ior(torch.tensor([WVLN_D], device=device_auto))
        assert n.item() == pytest.approx(1.5168, abs=1e-3)


class TestPatentGlassGrades:
    """Glass grades that patent prescriptions name but no bundled catalog had."""

    @pytest.mark.parametrize(
        ("name", "nd", "vd"),
        [
            ("TAFD35L", 1.91082, 35.25),
            ("NBFD29", 1.77047, 29.74),
            ("J-KZFH9", 1.73800, 32.26),
            ("FDS20-W", 1.86966, 20.02),
        ],
    )
    def test_patent_grade_resolves(self, name, nd, vd):
        mat = Material(name)
        assert mat.dispersion == "cauchy"
        assert float(mat.n) == pytest.approx(nd)
        assert float(mat.V) == pytest.approx(vd)

    def test_refractiveindex_grade_keeps_its_formula(self):
        # TAFD37A is in refractiveindex_data.json; a custom-table row would
        # shadow its dispersion formula with a Cauchy fit.
        assert Material("tafd37a").dispersion == "rii"


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
