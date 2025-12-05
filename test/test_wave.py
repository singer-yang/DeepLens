"""
Tests for deeplens/optics/wave.py - Wave optics and propagation.
"""

import pytest
import torch


from deeplens.optics.wave import ComplexWave, AngularSpectrumMethod


class TestComplexWaveInit:
    """Test ComplexWave initialization."""

    def test_complex_wave_init_default(self, device_auto):
        """Should initialize with default parameters."""
        wave = ComplexWave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        assert wave.wvln == 0.55
        assert (wave.z == 0.0).all().item()
        assert wave.phy_size == (4.0, 4.0)
        assert wave.res == (256, 256)

    def test_complex_wave_init_with_field(self, device_auto):
        """Should initialize with custom field."""
        u = torch.ones(256, 256, dtype=torch.complex64, device=device_auto)
        
        wave = ComplexWave(
            u=u,
            wvln=0.55,
            phy_size=(4.0, 4.0),
        )
        
        assert wave.u.shape[-2:] == (256, 256)


class TestComplexWavePointWave:
    """Test point source spherical wave."""

    def test_point_wave_center(self, device_auto):
        """Point wave at center should be symmetric."""
        wave = ComplexWave.point_wave(
            point=(0, 0, -1000.0),
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        assert wave.u.shape[-2:] == (256, 256)
        # Check approximate symmetry
        irr = torch.abs(wave.u)**2
        assert torch.allclose(irr[0, 0, 128, 64], irr[0, 0, 128, 192], rtol=0.1)

    def test_point_wave_intensity(self, device_auto):
        """Point wave should have non-zero intensity."""
        wave = ComplexWave.point_wave(
            point=(0, 0, -1000.0),
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        irr = torch.abs(wave.u)**2
        irr = torch.abs(wave.u)**2
        assert irr.sum().item() > 0


class TestComplexWavePlaneWave:
    """Test plane wave initialization."""

    def test_plane_wave_uniform(self, device_auto):
        """Plane wave should have uniform amplitude."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        amp = torch.abs(wave.u)
        # All amplitudes should be equal (within numerical precision)
        # All amplitudes should be equal (within numerical precision)
        assert torch.allclose(amp, amp[0, 0, 0, 0].expand_as(amp), atol=1e-5)

    def test_plane_wave_with_valid_r(self, device_auto):
        """Plane wave should respect valid radius."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
            valid_r=1.0,
        )
        
        # Corners should be zero (outside valid radius)
        # Corners should be zero (outside valid radius)
        assert wave.u[0, 0, 0, 0].abs().item() == pytest.approx(0.0, abs=1e-5)


class TestComplexWaveImageWave:
    """Test image-modulated wave."""

    def test_image_wave(self, device_auto):
        """Should create wave from image."""
        img = torch.rand(256, 256, device=device_auto)
        
        wave = ComplexWave.image_wave(
            img=img,
            wvln=0.55,
            phy_size=(4.0, 4.0),
        )
        
        # Amplitude should match sqrt(image)
        amp = torch.abs(wave.u)
        expected = torch.sqrt(img)
        assert torch.allclose(amp, expected, atol=1e-4)


class TestComplexWavePropagation:
    """Test wave propagation."""

    def test_wave_prop_distance(self, device_auto):
        """Propagation should update z coordinate."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        wave.prop(prop_dist=10.0)
        
        assert (wave.z == 10.0).all().item()

    def test_wave_prop_to(self, device_auto):
        """prop_to should propagate to specific z."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        wave.prop_to(z=10.0)
        
        assert (wave.z == 10.0).all().item()

    def test_wave_prop_energy_conservation(self, device_auto):
        """Propagation should conserve energy."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        energy_before = (torch.abs(wave.u)**2).sum()
        wave.prop(prop_dist=10.0)
        energy_after = (torch.abs(wave.u)**2).sum()
        
        assert torch.allclose(energy_before, energy_after, rtol=0.1)

    def test_wave_prop_with_refractive_index(self, device_auto):
        """Propagation should account for refractive index."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            z=0.0,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        # Propagate in medium with n=1.5
        wave.prop(prop_dist=10.0, n=1.5)
        
        assert (wave.z == 10.0).all().item()


class TestAngularSpectrumMethod:
    """Test Angular Spectrum Method propagation."""

    def test_asm_basic(self, device_auto):
        """ASM should propagate field."""
        u = torch.ones(256, 256, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=10.0,
            wvln=0.55,
            ps=0.01,  # pixel size [mm]
        )
        
        assert u_prop.shape == u.shape

    def test_asm_zero_distance(self, device_auto):
        """Zero propagation should return same field."""
        u = torch.rand(256, 256, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=0.0,
            wvln=0.55,
            ps=0.01,
        )
        
        assert torch.allclose(u_prop, u, atol=1e-5)

    def test_asm_with_padding(self, device_auto):
        """ASM with padding should avoid aliasing."""
        u = torch.ones(128, 128, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=10.0,
            wvln=0.55,
            ps=0.01,
            padding=True,
        )
        
        assert u_prop.shape == u.shape

    def test_asm_batch(self, device_auto):
        """ASM should support batch dimension."""
        u = torch.ones(1, 1, 256, 256, dtype=torch.complex64, device=device_auto)
        
        u_prop = AngularSpectrumMethod(
            u=u,
            z=10.0,
            wvln=0.55,
            ps=0.01,
        )
        
        assert u_prop.shape == u.shape


class TestComplexWaveGrids:
    """Test grid generation."""

    def test_gen_xy_grid(self, device_auto):
        """Should generate correct xy grid."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        wave.gen_xy_grid()
        
        assert hasattr(wave, "x")
        assert hasattr(wave, "y")
        assert wave.x.shape == (256, 256)
        assert wave.y.shape == (256, 256)

    def test_gen_freq_grid(self, device_auto):
        """Should generate frequency grid."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        fx, fy = wave.gen_freq_grid()
        
        assert fx is not None
        assert fy is not None


class TestComplexWaveOperations:
    """Test wave manipulation operations."""

    def test_wave_pad(self, device_auto):
        """Padding should increase resolution and physical size."""
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(256, 256),
        )
        
        original_size = wave.phy_size
        wave.pad(Hpad=64, Wpad=64)
        
        assert wave.u.shape[-2:] == (256 + 128, 256 + 128)
        assert wave.phy_size[0] > original_size[0]

    def test_wave_flip(self, device_auto):
        """Flip should reverse field."""
        wave = ComplexWave(
            u=torch.arange(16).reshape(4, 4).float().to(dtype=torch.complex64),
            wvln=0.55,
            phy_size=(4.0, 4.0),
        )
        
        original_u = wave.u.clone()
        wave.flip()
        
        # Check that corners swapped
        # Check that corners swapped
        assert wave.u[0, 0, 0, 0] == original_u[0, 0, -1, -1]


class TestComplexWaveIO:
    """Test save/load functionality."""

    def test_wave_save_load(self, device_auto, test_output_dir):
        """Should save and load wave."""
        import os
        
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(128, 128),
        )
        
        filepath = os.path.join(test_output_dir, "test_wave.npz")
        wave.save(filepath)
        
        assert os.path.exists(filepath)
        
        # Load back
        wave2 = ComplexWave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(128, 128),
        )
        wave2.load(filepath)
        
        assert wave2.u.shape == wave.u.shape


class TestComplexWaveVisualization:
    """Test visualization methods."""

    def test_wave_show_irradiance(self, device_auto, test_output_dir):
        """Should save irradiance image."""
        import os
        
        wave = ComplexWave.plane_wave(
            wvln=0.55,
            phy_size=(4.0, 4.0),
            res=(128, 128),
        )
        
        save_path = os.path.join(test_output_dir, "wave_irr.png")
        wave.show(save_name=save_path, data="irr")
        
        assert os.path.exists(save_path)
