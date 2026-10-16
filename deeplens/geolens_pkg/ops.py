# Copyright 2026 KAUST Computational Imaging Group, Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/vccimaging/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Lens operations mixin for GeoLens.

Provides methods that operate on a lens in place:
    - Surface pruning (clear aperture sizing)
    - Lens shape correction
    - Glass-catalog material matching
"""

import logging

import torch

from ..geometric_surface import Aperture


class GeoLensOps:
    """Mixin providing in-place lens operations for GeoLens.

    Bundles methods that modify a lens during design optimization: sizing
    clear apertures by ray tracing (pruning) and correcting lens geometry.
    Intended to be mixed into the `GeoLens` class, so all methods access lens
    state (`self.surfaces`, `self.d_sensor`, `self.rfov`, etc.) on the host.

    Key methods:
        prune_surf: Size clear apertures by ray tracing.
        correct_shape: Fix lens geometry during optimization.
    """

    # ====================================================================================
    # Surface pruning and shape correction
    # ====================================================================================
    @torch.no_grad()
    def prune_surf(self, mounting_margin=None):
        """Prune surface radii so all valid rays pass through, then enforce manufacturability.

        Traces 16 meridional fields from 0 to the full FoV to find the maximum
        ray height [mm] on each surface, expands it by a mounting margin, then
        caps the proposed radii to satisfy an edge-sag limit and edge-clearance
        (air-gap / edge-thickness) constraints with neighbouring surfaces.
        Aperture surfaces are not resized. The capped radii are committed via
        each surface's `update_r`.

        Args:
            mounting_margin (float or None, optional): Absolute mounting margin
                [mm] added to the ray-traced clear-aperture radius. If `None`,
                the margin is auto-selected per surface: 5% of the ray-traced
                radius when that radius is below 5 mm, otherwise 1 mm. Defaults
                to None.
        """
        surface_range = self.find_diff_surf()
        num_surfs = len(self.surfaces)

        # ------------------------------------------------------------------
        # 1. Temporarily remove radius limits so the trace is unclipped
        # ------------------------------------------------------------------
        saved_radii = [self.surfaces[i].r for i in range(num_surfs)]
        for i in surface_range:
            self.surfaces[i].r = self.surfaces[i].max_height()

        # ------------------------------------------------------------------
        # 2. Trace rays at full FoV to find maximum ray height per surface
        # ------------------------------------------------------------------
        assert self.rfov is not None, "prune_surf() requires self.rfov."
        fov_deg = self.rfov * 180 / torch.pi
        num_fov_samples = 16
        fov_y = torch.linspace(0.0, fov_deg, num_fov_samples, device=self.device)
        ray = self.sample_from_fov(fov_x=[0.0], fov_y=fov_y)
        _, ray_o_record = self.trace2sensor(ray=ray, record=True)

        # Ray record, shape [num_rays, num_surfaces + 2, 3]
        ray_o_record = torch.stack(ray_o_record, dim=-2)
        ray_o_record = torch.nan_to_num(ray_o_record, nan=0.0, posinf=0.0, neginf=0.0)
        ray_o_record = ray_o_record.reshape(-1, ray_o_record.shape[-2], 3)

        # Compute the maximum ray height for each surface
        ray_r_record = (ray_o_record[..., :2] ** 2).sum(-1).sqrt()
        surf_r_max = ray_r_record.max(dim=0)[0][1:-1]

        # ------------------------------------------------------------------
        # 3. Propose new radii (not yet committed to surfaces).
        # ------------------------------------------------------------------
        proposed_r = [float(self.surfaces[i].r) for i in range(num_surfs)]
        for i in surface_range:
            # Surface radius required by ray tracing
            if surf_r_max[i] > 0:
                base = float(surf_r_max[i].item())
            else:
                base = float(self.surfaces[i].r)

            # Expand the ray-traced radius by a mounting margin
            if mounting_margin is None:
                r_expand = 0.05 * base if base < 5.0 else 1.0
            else:
                r_expand = float(mounting_margin)

            # Propose the new radius, capped at the surface's physical maximum height
            proposed_r[i] = min(base + r_expand, float(self.surfaces[i].max_height()))

        # ------------------------------------------------------------------
        # 3b. Sag cap: edge sag must not exceed sag_factor * proposed radius.
        # Grid-search for the largest r in [r_min, proposed_r] where the
        # constraint holds. The grid is dense enough for typical aspheric sag
        # profiles; non-monotonic extremes are handled conservatively.
        # ------------------------------------------------------------------
        sag_factor = 0.4
        for i in surface_range:
            if not isinstance(self.surfaces[i], Aperture):
                r_prop = proposed_r[i]
                r_cands = torch.linspace(r_prop / 64, r_prop, 64, device=self.device)
                z0 = self.surfaces[i].surface_with_offset(
                    torch.tensor(0.0, device=self.device), 0.0, valid_check=False
                )
                z_cands = self.surfaces[i].surface_with_offset(
                    r_cands, torch.zeros_like(r_cands), valid_check=False
                )
                sag_valid = (z_cands - z0).abs() <= sag_factor * r_cands
                if sag_valid.any():
                    proposed_r[i] = min(r_prop, float(r_cands[sag_valid].max().item()))
                else:
                    proposed_r[i] = float(r_cands[0].item())

        # ------------------------------------------------------------------
        # 4. Edge-clearance pass — proactively cap adjacent pairs so the
        #    committed radii never produce self-intersection at the edge.
        #    Thresholds match loss_bound. The cap uses the common
        #    clear-aperture overlap between adjacent surfaces so one surface is
        #    not pruned against regions where the neighbour has already been
        #    apertured away. Aperture surfaces are skipped; the stop size is an
        #    optical specification and should not be changed by pruning. The cap
        #    is computed via a single vectorized grid search rather than a
        #    serial binary loop.
        #
        #    Each pruned surface is checked against both neighbours. The
        #    previous implementation only capped surface i against i + 1,
        #    which allowed surface i to expand into i - 1 and later crash
        #    tracing/optimization.
        # ------------------------------------------------------------------
        min_radius_floor = 0.1  # mm — guard against update_r(0) killing a surface
        n_cand = 64
        n_edge = 64
        r_frac = torch.linspace(0.5, 1.0, n_edge, device=self.device)
        cand_frac = torch.linspace(1.0 / n_cand, 1.0, n_cand, device=self.device)

        def cap_radius_against_pair(cap_idx, prev_idx, next_idx):
            prev_surf = self.surfaces[prev_idx]
            next_surf = self.surfaces[next_idx]
            if isinstance(prev_surf, Aperture) or isinstance(next_surf, Aperture):
                return
            if isinstance(self.surfaces[cap_idx], Aperture):
                return

            edge_min = 0.1  # mm
            r_check = proposed_r[cap_idx]

            other_idx = next_idx if cap_idx == prev_idx else prev_idx
            other_r = proposed_r[other_idx]
            z_off = {prev_idx: 0.0, next_idx: float(prev_surf.d_next)}

            required_r = max(
                float(surf_r_max[cap_idx].item()),
                min_radius_floor,
            )

            # Vectorized cap: evaluate gap for 64 candidate radii in one pass.
            cand_r = cand_frac * r_check
            cand_overlap_r = torch.minimum(
                cand_r, torch.tensor(other_r, device=self.device)
            )
            r_grid = cand_overlap_r.unsqueeze(1) * r_frac.unsqueeze(0)
            z_prev_grid = prev_surf.surface_with_offset(
                r_grid.reshape(-1),
                0.0,
                valid_check=False,
                d=z_off[prev_idx],
            ).reshape(n_cand, n_edge)
            z_next_grid = next_surf.surface_with_offset(
                r_grid.reshape(-1),
                0.0,
                valid_check=False,
                d=z_off[next_idx],
            ).reshape(n_cand, n_edge)
            per_cand_gap = (z_next_grid - z_prev_grid).min(dim=-1).values
            overlap_ok = per_cand_gap >= edge_min

            # Sag-bracket: the cap surface's edge z (at candidate r) must not
            # axially cross the other surface's edge z. Catches the case
            # where high-order aspheric terms blow up beyond the surface's
            # design r and drag its edge past the neighbour, while the
            # in-overlap gap above is still fine.
            cap_surf = self.surfaces[cap_idx]
            other_surf = self.surfaces[other_idx]
            z_other_edge = other_surf.surface_with_offset(
                torch.tensor(other_r, device=self.device),
                torch.tensor(0.0, device=self.device),
                valid_check=False,
                d=z_off[other_idx],
            )
            z_cap_at_cand = cap_surf.surface_with_offset(
                cand_r,
                torch.zeros_like(cand_r),
                valid_check=False,
                d=z_off[cap_idx],
            )
            if cap_idx > other_idx:
                # cap is later in light path — must stay axially after other
                bracket_ok = z_cap_at_cand > z_other_edge + edge_min
            else:
                # cap is earlier — must stay axially before other
                bracket_ok = z_cap_at_cand < z_other_edge - edge_min

            valid_mask = overlap_ok & bracket_ok
            if not bool(valid_mask.any()):
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): no candidate "
                    f"radius satisfies edge_min {edge_min:.3f} mm at "
                    f"r_check {r_check:.3f} mm (possible sag crossing near "
                    f"axis). Reducing surface {cap_idx} to the ray-required radius "
                    f"{required_r:.3f} mm, but edge clearance may remain "
                    f"violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = float((cand_frac[valid_mask].max() * r_check).item())
            if r_safe < required_r:
                logging.warning(
                    f"Surf {prev_idx}-{next_idx} "
                    f"({prev_surf.mat2.name}): ray-required "
                    f"radius {required_r:.3f} mm exceeds edge-clearance-safe "
                    f"radius {r_safe:.3f} mm for edge_min {edge_min:.3f} mm. "
                    f"Reducing surface {cap_idx} to the ray-required radius; edge "
                    f"clearance may remain violated."
                )
                proposed_r[cap_idx] = min(proposed_r[cap_idx], required_r)
                return

            r_safe = max(r_safe, min_radius_floor)
            if proposed_r[cap_idx] > r_safe:
                proposed_r[cap_idx] = r_safe

        for i in surface_range:
            if i > 0:
                cap_radius_against_pair(i, i - 1, i)
            if i < num_surfs - 1:
                cap_radius_against_pair(i, i, i + 1)

        # ------------------------------------------------------------------
        # 4b. Commit the capped proposed radii to the surfaces.
        # ------------------------------------------------------------------
        for i in surface_range:
            if proposed_r[i] > 0:
                self.surfaces[i].update_r(proposed_r[i])

    @torch.no_grad()
    def correct_shape(self, mounting_margin=None):
        """Correct invalid lens shape during lens design optimization.

        The first surface is always at z=0 under sequential geometry, so shape
        correction only needs to prune surfaces to let valid rays pass.

        Args:
            mounting_margin (float or None, optional): Absolute mounting margin
                [mm] for surface pruning, passed through to `prune_surf`.
                Defaults to None.
        """
        self.prune_surf(mounting_margin=mounting_margin)

    @torch.no_grad()
    def match_materials(self, mat_table="CDGM"):
        """Match each surface's material to the nearest entry in a glass catalog.

        Replaces every surface's `mat2` glass with the closest real catalog
        glass in-place, making an idealised design manufacturable.

        Args:
            mat_table (str, optional): Glass catalog name. Supported values are
                'CDGM' (default catalog) and 'PLASTIC'. Defaults to 'CDGM'.

        Raises:
            NotImplementedError: If `mat_table` is an unrecognised catalog name.
        """
        for surf in self.surfaces:
            surf.mat2.match_material(mat_table=mat_table)
