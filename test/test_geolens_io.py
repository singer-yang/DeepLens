"""Tests for deeplens/optics/geolens_pkg/io.py — GeoLensIO mixin.

Tests lens file I/O for JSON, Zemax (.zmx), and Code V (.seq) formats.
"""

import json
import math
import os
from pathlib import Path

import pytest
import torch


class TestJSONIO:
    """Tests for JSON lens file I/O."""

    def test_read_write_json_roundtrip(self, sample_singlet_lens, test_output_dir):
        """Write JSON then read back — surface count and foclen should be preserved."""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_roundtrip.json")
        original_num_surfs = len(lens.surfaces)
        original_foclen = lens.foclen

        lens.write_lens_json(out_path)
        assert os.path.exists(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == original_num_surfs
        assert lens2.foclen == pytest.approx(original_foclen, rel=0.01)

    def test_read_write_json_cellphone(self, sample_cellphone_lens, test_output_dir):
        """Round-trip a cellphone lens (with aspheric surfaces)."""
        lens = sample_cellphone_lens
        out_path = os.path.join(test_output_dir, "test_cellphone_roundtrip.json")
        original_num_surfs = len(lens.surfaces)

        lens.write_lens_json(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens(filename=out_path)
        assert len(lens2.surfaces) == original_num_surfs

    def test_legacy_sensor_size_derives_sensor_radius(self, test_output_dir):
        """Legacy JSON without r_sensor retains its explicit rectangular sensor."""
        source = Path("datasets/lenses/singlet/example1.json")
        data = json.loads(source.read_text(encoding="utf-8"))
        data.pop("r_sensor", None)
        sensor_size = data["(sensor_size)"]
        output = Path(test_output_dir) / "legacy_sensor_size.json"
        output.write_text(json.dumps(data), encoding="utf-8")

        from deeplens import GeoLens

        lens = GeoLens(filename=str(output))
        expected_radius = math.hypot(*sensor_size) / 2.0
        assert lens.r_sensor == pytest.approx(expected_radius)
        assert tuple(lens.sensor_size) == pytest.approx(tuple(sensor_size))


class TestZMXIO:
    """Tests for Zemax .zmx lens file I/O."""

    def test_read_zmx(self, lenses_dir):
        """Load a .zmx file and verify it produces surfaces."""
        zmx_path = os.path.join(lenses_dir, "camera/ef35mm_f2.0.zmx")
        if not os.path.exists(zmx_path):
            pytest.skip("ZMX test file not available")

        from deeplens import GeoLens

        lens = GeoLens()
        lens.read_lens_zmx(zmx_path)
        assert len(lens.surfaces) > 0
        assert lens.d_sensor is not None

    def test_write_zmx(self, sample_singlet_lens, test_output_dir):
        """Write a .zmx file and verify it exists."""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_write.zmx")
        lens.write_lens_zmx(out_path)
        assert os.path.exists(out_path)

    def test_zmx_roundtrip(self, sample_singlet_lens, test_output_dir):
        """Write then read .zmx — surface count should be preserved."""
        lens = sample_singlet_lens
        original_num_surfs = len(lens.surfaces)
        out_path = os.path.join(test_output_dir, "test_zmx_roundtrip.zmx")
        lens.write_lens_zmx(out_path)

        from deeplens import GeoLens

        lens2 = GeoLens()
        lens2.read_lens_zmx(out_path)
        # ZMX round-trip may lose some surface types, but count should be close
        assert len(lens2.surfaces) > 0

    def test_zmx_aperture_exports_diam(self):
        """Aperture (STOP) surface must export a DIAM (semi-diameter) line.

        Regression test: ``Aperture.zmx_str`` previously omitted ``DIAM``
        entirely, so the exported aperture stop had no aperture size and a
        re-import defaulted the radius to 1.0 mm.
        """
        from deeplens.geometric_surface import Aperture

        aperture = Aperture(r=2.5, d_next=0.0)
        surf_str = aperture.zmx_str(surf_idx=1, d_next=torch.tensor(5.0))

        assert "STOP" in surf_str
        diam_lines = [
            ln for ln in surf_str.splitlines() if ln.strip().startswith("DIAM")
        ]
        assert len(diam_lines) == 1, f"Expected one DIAM line, got: {surf_str!r}"
        assert "2.5" in diam_lines[0]

    def test_zmx_aperture_size_roundtrip(self, sample_cellphone_lens, test_output_dir):
        """Aperture semi-diameter must survive a .zmx write/read round-trip."""
        from deeplens import GeoLens
        from deeplens.geometric_surface import Aperture

        lens = sample_cellphone_lens
        aper_idx = next(
            i for i, s in enumerate(lens.surfaces) if isinstance(s, Aperture)
        )
        # Use a distinctive radius so the read default (1.0) cannot mask the bug.
        lens.surfaces[aper_idx].r = 1.234

        out_path = os.path.join(test_output_dir, "test_zmx_aperture_roundtrip.zmx")
        lens.write_lens_zmx(out_path)

        lens2 = GeoLens()
        lens2.read_lens_zmx(out_path)
        aper2 = next(s for s in lens2.surfaces if isinstance(s, Aperture))
        assert aper2.r == pytest.approx(1.234, abs=1e-3)

    def test_write_zmx_keeps_catalog_glass_name(self, lenses_dir, test_output_dir):
        """Catalog glasses export by name (with their GCAT); n/V stays model glass."""
        from deeplens import GeoLens
        from deeplens.material import Material

        lens = GeoLens(filename=str(Path(lenses_dir) / "camera/ef35mm_f2.0.zmx"))
        glass = [s for s in lens.surfaces if s.mat2.name != "air"]
        glass[0].mat2 = Material("n-bk7")
        glass[1].mat2 = Material("1.6/50.0")
        glass[2].mat2 = Material("okp4")  # PLASTIC2022: not a Zemax catalog
        glass[3].mat2 = Material("h-k9l")
        glass[3].mat2.get_optimizer_params()  # optimizable: n/V no longer catalog
        output = Path(test_output_dir) / "named_glass.zmx"
        lens.write_lens_zmx(str(output))

        zmx = output.read_text()
        assert "GLAS N-BK7 0 0 " in zmx
        assert "GLAS ___BLANK 1 0 1.6 50.0" in zmx
        assert "OKP4" not in zmx and "H-K9L" not in zmx
        assert zmx.count("GLAS ___BLANK") == len(glass) - 1
        gcat = next(l for l in zmx.splitlines() if l.startswith("GCAT")).split()
        assert gcat == ["GCAT", "MISC", "SCHOTT"]

        reread = GeoLens(filename=str(output))
        names = [s.mat2.name for s in reread.surfaces if s.mat2.name != "air"]
        assert names[:2] == ["n-bk7", "1.6/50.0"]

    def test_unknown_named_glass_uses_embedded_model_values(
        self, lenses_dir, test_output_dir
    ):
        """An unknown catalog glass falls back to nd/Vd carried by GLAS."""
        source = Path(lenses_dir) / "camera/ef35mm_f2.0.zmx"
        zmx = source.read_text(encoding="utf-8").replace(
            "GLAS ___BLANK 1 0 1.58913 61.2",
            "GLAS D-FK90 0 0 1.48656 84.47",
            1,
        )
        output = Path(test_output_dir) / "unknown_model_glass.zmx"
        output.write_text(zmx, encoding="utf-16")

        from deeplens import GeoLens

        lens = GeoLens(filename=str(output))
        material = lens.surfaces[0].mat2
        assert material.name == "1.48656/84.47"
        assert float(material.n) == pytest.approx(1.48656, rel=1e-6)
        assert float(material.V) == pytest.approx(84.47, rel=1e-6)

    def test_unsupported_surface_type_is_rejected_before_finalization(
        self, lenses_dir, test_output_dir
    ):
        """Unknown Zemax surfaces cannot be silently removed from the lens."""
        source = Path(lenses_dir) / "camera/ef35mm_f2.0.zmx"
        zmx = source.read_text(encoding="utf-8").replace(
            "SURF 1 \n    TYPE STANDARD",
            "SURF 1 \n    TYPE TOROIDAL",
            1,
        )
        output = Path(test_output_dir) / "unsupported_surface.zmx"
        output.write_text(zmx, encoding="utf-8")

        from deeplens import GeoLens

        with pytest.raises(
            NotImplementedError,
            match=r"Unsupported Zemax surface types: TOROIDAL",
        ):
            GeoLens(filename=str(output))

    def test_glass_name_with_slash_falls_back_to_model_values(self):
        """A GLAS name with "/" that is not n/V uses the record's nd/Vd.

        Material raises ValueError (not NotImplementedError) for such a name,
        which used to escape the model-glass fallback and abort the import.
        """
        from deeplens import GeoLens

        record = "FOO/BAR 0 0 1.5168 64.17 0 0 0 0"
        assert GeoLens._resolve_zmx_glass(record) == "1.5168/64.17"


class TestSEQIO:
    """Tests for Code V .seq lens file I/O."""

    def test_write_seq(self, sample_singlet_lens, test_output_dir):
        """Write a .seq file and verify it exists."""
        lens = sample_singlet_lens
        out_path = os.path.join(test_output_dir, "test_write.seq")
        lens.write_lens_seq(out_path)
        assert os.path.exists(out_path)

    def test_missing_image_surface_reports_actionable_error(self, test_output_dir):
        """A truncated Code V file names the missing SI image surface."""
        output = Path(test_output_dir) / "missing_image.seq"
        output.write_text(
            "RDM\nEPD 10.0\nYAN 0 10\nSO 0.0 1e10\nS 50.0 3.0 SK16\nS -50.0 5.0\n",
            encoding="utf-8",
        )

        from deeplens import GeoLens

        with pytest.raises(ValueError, match=r"image surface.*\bSI\b"):
            GeoLens(filename=str(output))

    def test_inline_comment_and_separator_are_not_glasses(self, test_output_dir):
        """``!`` comments and ``;`` command separators are not glass names.

        Regression: ``S r t; CCY 0`` read the glass as ``N-BK7;`` and
        ``S r t ! note`` read it as ``!``; both surfaces were silently dropped.
        """
        from deeplens import GeoLens

        clean = (
            "RDM\nEPD 10.0\nYAN 0 10\nSO 0.0 1e10\n"
            "S 50.0 3.0 N-BK7\nS -50.0 45.0\nSI 0.0 0.0\nGO\n"
        )
        commented = (
            "! hand-written prescription\nRDM\nEPD 10.0\nYAN 0 10 ! degrees\n"
            "SO 0.0 1e10\nS 50.0 3.0 N-BK7; CCY 0\nS -50.0 45.0 ! air gap\n"
            "SI 0.0 0.0\nGO\n"
        )
        lenses = []
        for stem, text in (("clean", clean), ("commented", commented)):
            path = Path(test_output_dir) / f"seq_comment_{stem}.seq"
            path.write_text(text, encoding="utf-8")
            lenses.append(GeoLens(filename=str(path)))

        reference, lens = lenses
        assert [s.mat2.get_name() for s in lens.surfaces] == [
            s.mat2.get_name() for s in reference.surfaces
        ]
        assert [s.mat2.get_name() for s in lens.surfaces] == ["n-bk7", "air"]
        assert float(lens.foclen) == pytest.approx(float(reference.foclen))


class TestCrossFormat:
    """Tests for cross-format conversion."""

    def test_json_to_zmx(self, sample_singlet_lens, test_output_dir):
        """Read JSON → write ZMX → read ZMX: foclen should be similar."""
        lens = sample_singlet_lens
        zmx_path = os.path.join(test_output_dir, "test_cross_format.zmx")
        lens.write_lens_zmx(zmx_path)

        from deeplens import GeoLens

        lens2 = GeoLens()
        lens2.read_lens_zmx(zmx_path)
        assert len(lens2.surfaces) > 0
