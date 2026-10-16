# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""PSF computation methods for geometric lens systems.

Supports three PSF models:
    1. Geometric PSF (``psf_geometric``): incoherent intensity ray tracing — fast and differentiable.
    2. Exit-pupil PSF (``psf_pupil_prop`` / ``psf_coherent``): coherent ray tracing to exit pupil,
       then Angular Spectrum Method (ASM) free-space propagation — accurate and differentiable.
    3. Huygens PSF (``psf_huygens``): coherent ray tracing to exit pupil, then Huygens-Fresnel
       integration — accurate but not differentiable.

Functions:
    - psf(): Dispatcher selecting between geometric, coherent, and Huygens models.
    - psf_geometric(): Incoherent geometric PSF via ray binning.
    - psf_coherent(): Alias for psf_pupil_prop.
    - psf_pupil_prop(): Exit-pupil diffraction PSF via coherent tracing + ASM.
    - pupil_field(): Complex wavefront at the exit pupil plane.
    - psf_huygens(): Huygens-Fresnel PSF via secondary point source integration.
    - psf_map(): Geometric PSF map across the field of view.
    - psf_center(): Reference PSF centre via chief ray or pinhole projection.
"""

import logging

import torch
import torch.nn.functional as F

from ..config import (
    EPSILON,
    PSF_KS,
    SPP_CALC,
    SPP_COHERENT,
    SPP_PSF,
)
from ..imgsim import forward_integral
from ..light import AngularSpectrumMethod
from ..utils import diff_float

logger = logging.getLogger(__name__)


class GeoLensPSF:
    """Mixin providing PSF computation for `GeoLens`.

    Exposes three PSF models through a single `psf` dispatcher: incoherent
    geometric ray tracing, coherent exit-pupil diffraction (ASM propagation),
    and Huygens-Fresnel integration. The geometric and coherent models are
    differentiable; Huygens is not. This class is not instantiated directly;
    it is mixed into `GeoLens`.
    """

    # ====================================================================================
    # PSF
    # We support three types of PSF:
    #   1. Geometric PSF (`psf`): incoherent intensity ray tracing
    #   2. Exit-pupil PSF (`psf_pupil_prop` / `psf_coherent`): coherent ray tracing to exit pupil, then free-space propagation with ASM
    #   3. Huygens PSF (`psf_huygens`): coherent ray tracing to exit pupil, then Huygens-Fresnel integration
    # ====================================================================================
    def psf(self, points, wvln=None, ks=PSF_KS, **kwargs):
        """Compute the Point Spread Function (PSF) for given point sources.

        Dispatches to one of three PSF models:
            - geometric: incoherent intensity ray tracing (fast, differentiable).
            - coherent: coherent tracing to exit pupil + ASM propagation (accurate, differentiable, single point).
            - huygens: Huygens-Fresnel integration (accurate, not differentiable, single point).

        Args:
            points (torch.Tensor): Normalized point source positions. Shape [N, 3]
                with x, y in [-1, 1] and z in [-Inf, 0]. The coherent and huygens
                models accept only a single point ([3] or [1, 3]).
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            **kwargs: Model-specific options:
                spp (int): Rays sampled per source. If None, uses the
                model-specific default (SPP_PSF / SPP_COHERENT).
                recenter (bool): If True (default), center the PSF on the chief
                ray; otherwise on the pinhole projection.
                model (str): One of 'geometric' (default), 'coherent', 'huygens'.
                return_field (bool): For the coherent and Huygens models, return
                    the energy-normalized complex sensor-plane field instead of
                    intensity. Defaults to False.

        Returns:
            torch.Tensor: Intensity PSF normalized to sum to 1, or an
                energy-normalized complex field when ``return_field=True``.
                Shape [ks, ks] for a single point, or [N, ks, ks] for the
                geometric model with N points.

        Raises:
            ValueError: If `model` is not one of the supported names.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        spp = kwargs.get("spp", None)
        recenter = kwargs.get("recenter", True)
        return_field = kwargs.get("return_field", False)
        model = kwargs.get("model", "geometric")
        if model == "geometric":
            spp = SPP_PSF if spp is None else spp
            return self.psf_geometric(points, ks, wvln, spp, recenter)
        elif model == "coherent":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_coherent(points, ks, wvln, spp, recenter, return_field)
        elif model == "huygens":
            spp = SPP_COHERENT if spp is None else spp
            return self.psf_huygens(points, ks, wvln, spp, recenter, return_field)
        else:
            raise ValueError(f"Unknown PSF model: {model}")

    def psf_geometric(self, points, ks=PSF_KS, wvln=None, spp=SPP_PSF, recenter=True):
        """Compute the single-wavelength geometric PSF by incoherent ray binning.

        Samples rays from each object point, traces them incoherently to the
        sensor, and bins the hit positions into a `ks × ks` intensity kernel.
        This model is fast and differentiable.

        Args:
            points (torch.Tensor): Normalized point source positions. Shape [N, 3]
                with x, y in [-1, 1] and z in [-Inf, 0].
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            spp (int, optional): Rays sampled per source. Defaults to SPP_PSF.
            recenter (bool, optional): If True (default), center on the chief ray;
                otherwise on the pinhole projection.

        Returns:
            psf (torch.Tensor): PSF normalized to sum to 1. Shape [ks, ks] for a
                single point, or [N, ks, ks] for N points.

        Reference:
            [1] https://optics.ansys.com/hc/en-us/articles/42661723066515-What-is-a-Point-Spread-Function
        """
        wvln = self.primary_wvln if wvln is None else wvln
        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device

        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.as_tensor(points, device=device, dtype=self.dtype)
        else:
            points = points.to(device=device, dtype=self.dtype)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        else:
            single_point = False

        # Sample rays. Ray position in the object space by perspective projection
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # Trace rays to sensor plane (incoherent)
        ray.is_coherent = False
        ray = self.trace2sensor(ray)

        # Calculate PSF center, shape [N, 2]. At the primary wavelength the
        # bundle just traced is reused for the chief ray instead of tracing a
        # second one. Other wavelengths still trace at the primary wavelength so
        # every channel shares one reference center and lateral color stays in
        # the kernels. Sparse bundles are not reused: chief-ray accuracy scales
        # as 1/sqrt(spp).
        if recenter:
            reuse_ray = ray if wvln == self.primary_wvln and spp >= SPP_CALC else None
            pointc = self.psf_center(point_obj, method="chief_ray", ray=reuse_ray)
        else:
            pointc = self.psf_center(point_obj, method="pinhole")

        # Monte Carlo integration
        psf = forward_integral(ray.flip_xy(), ps=pixel_size, ks=ks, pointc=pointc)

        # Intensity normalization
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_coherent(
        self,
        points,
        ks=PSF_KS,
        wvln=None,
        spp=SPP_COHERENT,
        recenter=True,
        return_field=False,
    ):
        """Compute the coherent exit-pupil PSF (alias for `psf_pupil_prop`).

        Traces coherent rays to the exit pupil and propagates the wavefront to
        the sensor with the Angular Spectrum Method (ASM). See `psf_pupil_prop`
        for full argument and return documentation.

        Args:
            points (torch.Tensor): Single normalized point source [3] or [1, 3]
                with x, y in [-1, 1] and z in [-Inf, 0].
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            spp (int, optional): Rays sampled. Defaults to SPP_COHERENT.
            recenter (bool, optional): If True (default), center on the chief ray.
            return_field (bool, optional): Return the energy-normalized complex
                sensor-plane field instead of intensity. Defaults to False.

        Returns:
            torch.Tensor: Intensity PSF normalized to sum to 1, or an
                energy-normalized complex field when ``return_field=True``.
                Shape [ks, ks].

        Note:
            The complex field has an arbitrary global phase because optical path
            length is measured relative to a reference; relative phase is meaningful.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        return self.psf_pupil_prop(
            points,
            ks=ks,
            wvln=wvln,
            spp=spp,
            recenter=recenter,
            return_field=return_field,
        )

    def psf_pupil_prop(
        self,
        points,
        ks=PSF_KS,
        wvln=None,
        spp=SPP_COHERENT,
        recenter=True,
        return_field=False,
    ):
        """Compute the single-point monochromatic PSF via the exit-pupil diffraction model.

        Steps:
            1. Compute the complex wavefront at the exit-pupil plane by coherent ray tracing.
            2. Propagate to the sensor plane with the Angular Spectrum Method (ASM)
               and return either its field or intensity. This function is differentiable.

        Args:
            points (torch.Tensor or list): Single normalized point source [3] or
                [1, 3] with x, y in [-1, 1] and z in [-Inf, 0].
            ks (int, optional): Size of the output PSF patch in pixels. If None,
                the full propagated intensity field is returned uncropped.
                Defaults to PSF_KS.
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            spp (int, optional): Number of rays to sample. Defaults to SPP_COHERENT.
            recenter (bool, optional): If True (default), center on the chief ray;
                otherwise on the pinhole projection.
            return_field (bool, optional): Return the energy-normalized complex
                sensor-plane field instead of intensity. Defaults to False.

        Returns:
            torch.Tensor: Intensity PSF normalized to sum to 1, or an
                energy-normalized complex field when ``return_field=True``.
                Shape [ks, ks] when `ks` is given, or [1, 1, 2H, 2H] when
                `ks` is None (twice the exit-pupil grid after zero-padding).

        Reference:
            [1] "End-to-End Hybrid Refractive-Diffractive Lens Design with Differentiable Ray-Wave Model", SIGGRAPH Asia 2024.

        Note:
            Similar to the ZEMAX FFT PSF, but free-space propagation uses the
            Angular Spectrum Method (ASM) instead of a single FFT. ASM is more
            accurate because the FFT approach assumes a far-field condition
            (e.g., chief ray perpendicular to the image plane).
            The complex field has an arbitrary global phase because optical path
            length is measured relative to a reference; relative phase is meaningful.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        # Pupil field by coherent ray tracing
        wavefront, psfc = self.pupil_field(
            points=points, wvln=wvln, spp=spp, recenter=recenter
        )

        # Propagate to sensor plane and get intensity
        pupilz, pupilr = self.get_exit_pupil()
        h, w = wavefront.shape
        # Manually pad wave field
        wavefront = F.pad(
            wavefront.unsqueeze(0).unsqueeze(0),
            [h // 2, h // 2, w // 2, w // 2],
            mode="constant",
            value=0,
        )
        # Free-space propagation using Angular Spectrum Method (ASM)
        sensor_field = AngularSpectrumMethod(
            wavefront,
            z=self.d_sensor - pupilz,
            wvln=wvln,
            ps=self.pixel_size,
            padding=False,
        )
        # Get intensity
        psf_inten = sensor_field.abs() ** 2

        # Calculate PSF center
        h, w = psf_inten.shape[-2:]
        # consider both interplation and padding
        psfc_idx_i = ((2 - psfc[1]) * h / 4).round().long()
        psfc_idx_j = ((2 + psfc[0]) * w / 4).round().long()

        if return_field:
            field = sensor_field
            if ks is not None:
                field = (
                    F.pad(
                        field,
                        [ks // 2, ks // 2, ks // 2, ks // 2],
                        mode="constant",
                        value=0,
                    )
                    .squeeze(0)
                    .squeeze(0)
                )
                field = field[
                    psfc_idx_i : psfc_idx_i + ks,
                    psfc_idx_j : psfc_idx_j + ks,
                ]
            energy = field.abs().square().sum(dim=(-2, -1), keepdim=True)
            return field / torch.sqrt(energy + EPSILON)

        # Crop valid PSF region and normalize
        if ks is not None:
            psf_inten_pad = (
                F.pad(
                    psf_inten,
                    [ks // 2, ks // 2, ks // 2, ks // 2],
                    mode="constant",
                    value=0,
                )
                .squeeze(0)
                .squeeze(0)
            )
            psf = psf_inten_pad[
                psfc_idx_i : psfc_idx_i + ks, psfc_idx_j : psfc_idx_j + ks
            ]
        else:
            psf = psf_inten

        # Intensity normalization, shape of [ks, ks] or [h, w]
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        return diff_float(psf)

    def pupil_field(self, points, wvln=None, spp=SPP_COHERENT, recenter=True):
        """Compute the complex wavefront at the exit-pupil plane by coherent ray tracing.

        The wavefront is xy-flipped for subsequent PSF calculation and binned at
        the sensor pixel size onto a square `[H, H]` grid (H = sensor height in
        pixels). This function is differentiable.

        Args:
            points (torch.Tensor or list): Single normalized point source [3] or
                [1, 3] with x, y in [-1, 1] and z in [-Inf, 0].
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            spp (int, optional): Number of rays to sample. Must be at least
                1,000,000 for accurate coherent simulation. Defaults to SPP_COHERENT.
            recenter (bool, optional): If True (default), center on the chief ray;
                otherwise on the pinhole projection.

        Returns:
            wavefront (torch.Tensor): Complex wavefront at the exit pupil, binned
                at pixel size. Shape [H, H].
            psf_center (list): Normalized PSF center [x, y] on the sensor in [-1, 1].

        Note:
            Default dtype must be torch.float64 for accurate phase calculation.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        assert spp >= 1_000_000, (
            f"Ray sampling {spp} is too small for coherent ray tracing, which may lead to inaccurate simulation."
        )
        if self.dtype != torch.float64:
            raise ValueError(
                "Coherent pupil propagation requires a float64 lens; call lens.astype(torch.float64)."
            )

        sensor_w, sensor_h = self.sensor_size
        device = self.device

        if isinstance(points, list):
            points = torch.as_tensor(points, device=device, dtype=self.dtype).unsqueeze(
                0
            )
        elif torch.is_tensor(points) and len(points.shape) == 1:
            points = points.unsqueeze(0).to(device=device, dtype=self.dtype)
        elif torch.is_tensor(points) and len(points.shape) == 2:
            points = points.to(device=device, dtype=self.dtype)
            assert points.shape[0] == 1, (
                f"pupil_field only supports single point input, got shape {points.shape}"
            )
        else:
            raise ValueError(f"Unsupported point type {points.type()}.")

        assert points.shape[0] == 1, (
            "Only one point is supported for pupil field calculation."
        )

        # Ray origin in the object space
        scale = self.calc_scale(points[:, 2].item())
        point_obj_x = points[:, 0] * scale * sensor_w / 2
        point_obj_y = points[:, 1] * scale * sensor_h / 2
        points_obj = torch.stack([point_obj_x, point_obj_y, points[:, 2]], dim=-1)

        # Ray center determined by chief ray
        # Shape of [N, 2], un-normalized physical coordinates
        if recenter:
            pointc = self.psf_center(points_obj, method="chief_ray")
        else:
            pointc = self.psf_center(points_obj, method="pinhole")

        # Ray-tracing to exit_pupil
        ray = self.sample_from_points(points=points_obj, num_rays=spp, wvln=wvln)
        ray.is_coherent = True
        ray = self.trace2exit_pupil(ray)

        # Calculate complex field (same physical size and resolution as the sensor)
        # Complex field is flipped here for further PSF calculation
        pointc_ref = torch.zeros_like(points[:, :2])  # [N, 2]
        wavefront = forward_integral(
            ray.flip_xy(),
            ps=self.pixel_size,
            ks=self.sensor_res[1],
            pointc=pointc_ref,
        )
        wavefront = wavefront.squeeze(0)  # [H, H]

        # PSF center (on the sensor plane).
        pointc = pointc[0, :]
        psf_center = [
            pointc[0] / sensor_w * 2,
            pointc[1] / sensor_h * 2,
        ]

        return wavefront, psf_center

    def psf_huygens(
        self,
        points,
        ks=PSF_KS,
        wvln=None,
        spp=SPP_COHERENT,
        recenter=True,
        return_field=False,
    ):
        """Compute the single-wavelength Huygens PSF by spherical-wave integration.

        Not differentiable, due to the heavy computational cost.

        Steps:
            1. Trace coherent rays to the exit-pupil plane.
            2. Treat every ray as a secondary point source emitting a spherical
               wave, and coherently sum these waves over the PSF pixel grid. Each
               contribution uses the Huygens-Fresnel obliquity factor
               $0.5 (1 + \\cos\\theta)$ and $1/r$ spherical-wave amplitude decay.

        Args:
            points (torch.Tensor): Single normalized point source [3] or [1, 3]
                with x, y in [-1, 1] and z in [-Inf, 0].
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            spp (int, optional): Rays sampled. Defaults to SPP_COHERENT.
            recenter (bool, optional): If True (default), center on the chief ray;
                otherwise on the pinhole projection.
            return_field (bool, optional): Return the energy-normalized complex
                sensor-plane field instead of intensity. Defaults to False.

        Returns:
            torch.Tensor: Intensity PSF normalized to sum to 1, or an
                energy-normalized complex field when ``return_field=True``.
                Shape [ks, ks].

        Reference:
            [1] "Optical Aberrations Correction in Postprocessing Using Imaging Simulation", TOG 2021.

        Note:
            Different from the ZEMAX Huygens PSF, which traces rays to the image
            plane and performs plane-wave integration.
            The complex field has an arbitrary global phase because optical path
            length is measured relative to a reference; relative phase is meaningful.
        """
        wvln = self.primary_wvln if wvln is None else wvln
        if self.dtype != torch.float64:
            raise ValueError(
                "Huygens propagation requires a float64 lens; call lens.astype(torch.float64)."
            )

        sensor_w, sensor_h = self.sensor_size
        pixel_size = self.pixel_size
        device = self.device
        wvln_mm = wvln * 1e-3  # Convert wavelength to mm

        # Points shape of [N, 3]
        if not torch.is_tensor(points):
            points = torch.as_tensor(points, device=device, dtype=self.dtype)
        else:
            points = points.to(device=device, dtype=self.dtype)

        if len(points.shape) == 1:
            single_point = True
            points = points.unsqueeze(0)
        elif len(points.shape) == 2 and points.shape[0] == 1:
            single_point = True
        else:
            raise ValueError(
                f"Points must be of shape [3] or [1, 3], got {points.shape}."
            )

        # Sample rays from object point
        depth = points[:, 2]
        scale = self.calc_scale(depth)
        point_obj_x = points[..., 0] * scale * sensor_w / 2
        point_obj_y = points[..., 1] * scale * sensor_h / 2
        point_obj = torch.stack([point_obj_x, point_obj_y, points[..., 2]], dim=-1)
        ray = self.sample_from_points(points=point_obj, num_rays=spp, wvln=wvln)

        # Trace rays coherently through the lens to exit pupil
        ray.is_coherent = True
        ray = self.trace2exit_pupil(ray)

        # Calculate PSF center (not flipped here)
        if recenter:
            pointc = -self.psf_center(point_obj, method="chief_ray")
        else:
            pointc = -self.psf_center(point_obj, method="pinhole")

        # Build PSF pixel coordinates (sensor plane at z = d_sensor)
        sensor_z = self.d_sensor.item()
        psf_half_size = (ks / 2) * pixel_size  # Physical half-size of PSF region
        x_coords = torch.linspace(
            -psf_half_size + pixel_size / 2,
            psf_half_size - pixel_size / 2,
            ks,
            device=device,
            dtype=self.dtype,
        )
        y_coords = torch.linspace(
            psf_half_size - pixel_size / 2,
            -psf_half_size + pixel_size / 2,
            ks,
            device=device,
            dtype=self.dtype,
        )
        psf_x, psf_y = torch.meshgrid(
            pointc[0, 0] + x_coords, pointc[0, 1] + y_coords, indexing="xy"
        )  # [ks, ks] each

        # Get valid rays only
        valid_mask = ray.is_valid > 0
        valid_pos = ray.o[valid_mask]  # [num_valid, 3]
        valid_dir = ray.d[valid_mask]  # [num_valid, 3]
        valid_opl = ray.opl[valid_mask]  # [num_valid]
        num_valid = valid_pos.shape[0]

        # Huygens integration: sum spherical waves from each secondary source
        psf_complex = torch.zeros(ks, ks, dtype=torch.complex128, device=device)
        opl_min = valid_opl.min()

        # Compute distance from each secondary source to each pixel
        batch_size = min(num_valid, 10_000)  # Process rays in batches
        for batch_start in range(0, num_valid, batch_size):
            batch_end = min(batch_start + batch_size, num_valid)

            # Batch ray data
            batch_pos = valid_pos[batch_start:batch_end]  # [batch, 3]
            batch_dir = valid_dir[batch_start:batch_end]  # [batch, 3]
            batch_opl = valid_opl[batch_start:batch_end].squeeze(-1)  # [batch]

            # Distance from each secondary source to each pixel
            # batch_pos: [batch, 3], psf_x: [ks, ks]
            dx = psf_x.unsqueeze(-1) - batch_pos[:, 0]  # [ks, ks, batch]
            dy = psf_y.unsqueeze(-1) - batch_pos[:, 1]  # [ks, ks, batch]
            dz = sensor_z - batch_pos[:, 2]  # [batch]

            # Distance r from secondary source to pixel
            r = torch.sqrt(dx**2 + dy**2 + dz**2)  # [ks, ks, batch]

            # Obliquity factor: cos(theta) where theta is angle from normal
            # Using ray direction at exit pupil (dz component)
            obliq = torch.abs(batch_dir[:, 2])  # [batch]
            amp = 0.5 * (1.0 + obliq)  # Huygens–Fresnel obliquity factor

            # Total optical path = OPL through lens + distance to pixel
            total_opl = batch_opl + r  # [ks, ks, batch]

            # Phase relative to reference
            phase = torch.fmod((total_opl - opl_min) / wvln_mm, 1.0) * (
                2 * torch.pi
            )  # [ks, ks, batch]

            # Complex amplitude: A * exp(i * phase) / r (spherical wave decay)
            # We use 1/r for spherical wave amplitude decay
            complex_amp = (amp / r) * torch.exp(1j * phase)  # [ks, ks, batch]

            # Sum contributions from this batch
            psf_complex += complex_amp.sum(dim=-1)  # [ks, ks]

        if return_field:
            field = torch.flip(psf_complex, [-2, -1])
            energy = field.abs().square().sum(dim=(-2, -1), keepdim=True)
            return field / torch.sqrt(energy + EPSILON)

        # Convert complex field to intensity
        psf = psf_complex.abs() ** 2

        # Intensity normalization
        psf = psf / (torch.sum(psf, dim=(-2, -1), keepdim=True) + EPSILON)

        # Flip PSF
        psf = torch.flip(psf, [-2, -1])

        if single_point:
            psf = psf.squeeze(0)

        return diff_float(psf)

    def psf_map(
        self,
        depth=None,
        grid=(7, 7),
        ks=PSF_KS,
        spp=SPP_PSF,
        wvln=None,
        recenter=True,
    ):
        """Compute the geometric PSF map across the field of view at a given depth.

        Overrides the base `Lens` method to improve efficiency by tracing all
        field points in parallel.

        Args:
            depth (float, optional): Object plane depth [mm]. When None (default),
                falls back to `self.obj_depth`.
            grid (int or tuple, optional): Grid size (grid_w, grid_h); an int is
                broadcast to a square grid. Defaults to (7, 7).
            ks (int, optional): Output kernel size in pixels. Defaults to PSF_KS.
            spp (int, optional): Rays sampled per source. Defaults to SPP_PSF.
            wvln (float, optional): Wavelength in µm. When None (default), falls
                back to `self.primary_wvln`.
            recenter (bool, optional): If True (default), center on the chief ray.

        Returns:
            psf_map (torch.Tensor): PSF map. Shape [grid_h, grid_w, 1, ks, ks].
        """
        wvln = self.primary_wvln if wvln is None else wvln
        depth = self.obj_depth if depth is None else depth
        if isinstance(grid, int):
            grid = (grid, grid)
        points = self.point_source_grid(depth=depth, grid=grid)
        points = points.reshape(-1, 3)
        psfs = self.psf(
            points=points, ks=ks, recenter=recenter, spp=spp, wvln=wvln
        ).unsqueeze(1)  # [grid_h * grid_w, 1, ks, ks]

        psf_map = psfs.reshape(grid[1], grid[0], 1, ks, ks)
        return psf_map

    @torch.no_grad()
    def psf_center(self, points_obj, method="chief_ray", ray=None):
        """Compute the reference PSF center on the sensor for a given point source.

        With method "chief_ray" it returns the negated sensor intercept of the
        sampled real ray closest to the physical aperture-stop centre. Invalid
        chief rays fall back to the pinhole model independently per field, as
        does the whole computation when the lens has no aperture stop. With
        "pinhole" it uses an ideal perspective projection (no distortion).

        Args:
            points_obj (torch.Tensor): Un-normalized object-plane point(s), shape
                [..., 3] [mm], spanning [-Inf, Inf] x [-Inf, Inf] x [-Inf, 0].
            method (str, optional): "chief_ray" or "pinhole". Defaults to "chief_ray".
            ray (Ray, optional): Bundle already traced to the sensor from
                `points_obj` at the primary wavelength, shape [..., num_rays, 3].
                When given, the chief ray is extracted from it instead of
                tracing a new bundle. Ignored for "pinhole".

        Returns:
            psf_center (torch.Tensor): Un-normalized PSF center on the sensor
                plane [mm], shape [..., 2].

        Raises:
            ValueError: If `method` is neither "chief_ray" nor "pinhole".
        """
        if method == "chief_ray":
            if self.aper_idx is None:
                logger.warning(
                    "Lens has no aperture stop; using the pinhole model for "
                    "PSF centers."
                )
                return self.psf_center(points_obj, method="pinhole")

            if ray is None:
                chief_ray = self.calc_chief_ray(points_obj, num_rays=SPP_PSF)
            else:
                chief_ray = self.calc_chief_ray(ray=ray)
            psf_center = -chief_ray.o[..., 0, :2]

            # Info level: a vignetted chief ray is routine mid-optimization, and
            # psf()/psf_map() run every iteration, so a warning would flood logs.
            valid = chief_ray.is_valid[..., 0].bool()
            if not valid.all():
                failed = int((~valid).sum().item())
                logger.info(
                    "%d of %d chief rays are invalid; using the pinhole model "
                    "for those PSF centers.",
                    failed,
                    valid.numel(),
                )
                pinhole_center = self.psf_center(points_obj, method="pinhole")
                psf_center = torch.where(
                    valid.unsqueeze(-1), psf_center, pinhole_center
                )

        elif method == "pinhole":
            # Pinhole camera perspective projection, distortion not considered
            if points_obj[..., 2].min().abs() < 100:
                print(
                    "Point source is too close, pinhole model may be inaccurate for PSF center calculation."
                )
            tan_point_fov_x = -points_obj[..., 0] / points_obj[..., 2]
            tan_point_fov_y = -points_obj[..., 1] / points_obj[..., 2]
            psf_center_x = self.foclen * tan_point_fov_x
            psf_center_y = self.foclen * tan_point_fov_y
            psf_center = torch.stack([psf_center_x, psf_center_y], dim=-1).to(
                self.device
            )

        else:
            raise ValueError(
                f"Unsupported method for PSF center calculation: {method}."
            )

        return psf_center
