# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Tolerance analysis for geometric lens design.

References:
    [1] Jun Dai, Liqun Chen, Xinge Yang, Yuyao Hu, Jinwei Gu, Tianfan Xue, "Tolerance-Aware Deep Optics," arXiv preprint arXiv:2502.04719, 2025.

Functions:
    Tolerance Setup:
        - init_tolerance(): Initialize tolerance parameters for the lens
        - sample_tolerance(): Sample a random manufacturing error for the lens
        - zero_tolerance(): Clear manufacturing error for the lens

    Tolerance Analysis Methods:
        - tolerancing_sensitivity(): Use sensitivity analysis (1st order gradient) to compute the tolerance score
        - tolerancing_monte_carlo(): Use Monte Carlo simulation to compute the tolerance
        - tolerancing_wavefront(): Use wavefront differential method to compute the tolerance
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from deeplens.basics import DEPTH


class GeoLensTolerance:
    """Tolerance analysis for geometric lens."""

    def init_tolerance(self, tolerance_params=None):
        """Initialize tolerance parameters for the lens."""
        if tolerance_params is None:
            tolerance_params = {}

        for i in range(len(self.surfaces)):
            self.surfaces[i].init_tolerance(tolerance_params=tolerance_params)

    @torch.no_grad()
    def sample_tolerance(self):
        """Sample a random manufacturing error for the lens."""
        # Randomly perturb all surfaces
        for i in range(len(self.surfaces)):
            self.surfaces[i].sample_tolerance()

        # Refocus the lens
        self.refocus()

    @torch.no_grad()
    def zero_tolerance(self):
        """Clear manufacturing error for the lens."""
        for i in range(len(self.surfaces)):
            self.surfaces[i].zero_tolerance()

        # Refocus the lens
        self.refocus()

    # ================================================
    # Three tolerancing analysis methods
    # 1. Sensitivity analysis (1st order gradient)
    # 2. Monte Carlo method
    # 3. Wavefront differential method
    # ================================================

    def tolerancing_sensitivity(self, tolerance_params=None):
        """Use sensitivity analysis (1st order gradient) to compute the tolerance score.

        References:
            [1] Page 10 from: https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/08/8-Tolerancing-1.pdf
            [2] Fast sensitivity control method with differentiable optics. Optics Express 2025.
            [3] Optical Design Tolerancing. CODE V.
        """
        # Initialize tolerance
        self.init_tolerance(tolerance_params=tolerance_params)

        # AutoDiff to compute the gradient/sensitivity
        self.get_optimizer_params()
        loss = self.loss_rms()
        loss.backward()

        # Calculate sensitivity results
        sensitivity_results = {}
        for i in range(len(self.surfaces)):
            sensitivity_results.update(self.surfaces[i].sensitivity_score())

        # Toleranced RSS (Root Sum Square) loss
        tolerancing_score = sum(
            v for k, v in sensitivity_results.items() if k.endswith("_score")
        )
        loss_rss = torch.sqrt(loss**2 + tolerancing_score).item()
        sensitivity_results["loss_nominal"] = round(loss.item(), 6)
        sensitivity_results["loss_rss"] = round(loss_rss, 6)
        return sensitivity_results

    @torch.no_grad()
    def tolerancing_monte_carlo(self, trials=1000, tolerance_params=None):
        """Use Monte Carlo simulation to compute the tolerance.

        Note: we can multiplex sampled rays to improve the speed.

        Args:
            trials (int): Number of Monte Carlo trials
            tolerance_params (dict): Tolerance parameters

        Returns:
            dict: Monte Carlo tolerance analysis results

        References:
            [1] https://optics.ansys.com/hc/en-us/articles/43071088477587-How-to-analyze-your-tolerance-results
            [2] Optical Design Tolerancing. CODE V.
        """

        def merit_func(lens, fov=0.0, depth=DEPTH):
            # Calculate MTF at a specific field of view
            point = [0, -fov / lens.rfov, depth]
            psf = lens.psf(points=point, recenter=True)
            freq, mtf_tan, mtf_sag = lens.psf2mtf(psf, pixel_size=lens.pixel_size)

            # Evaluate MTF at a specific frequency
            nyquist_freq = 0.5 / lens.pixel_size
            eval_freq = 0.25 * nyquist_freq
            idx = torch.argmin(torch.abs(torch.tensor(freq) - eval_freq))
            score = (mtf_tan[idx] + mtf_sag[idx]) / 2
            return score.item()

        # Initialize tolerance
        self.init_tolerance(tolerance_params=tolerance_params)

        # Monte Carlo simulation
        merit_ls = []
        with torch.no_grad():
            for i in tqdm(range(trials)):
                # Sample a random perturbation
                self.sample_tolerance()

                # Evaluate perturbed performance
                perturbed_merit = merit_func(lens=self, fov=0.0, depth=DEPTH)
                merit_ls.append(perturbed_merit)

                # Clear perturbation
                self.zero_tolerance()

        merit_ls = np.array(merit_ls)

        # Baseline merit
        self.refocus()
        baseline_merit = merit_func(lens=self, fov=0.0, depth=DEPTH)
        # merit_ls /= baseline_merit

        # Results plot
        sorted_merit = np.sort(merit_ls)
        cumulative_prob = (1 - np.arange(len(sorted_merit)) / len(sorted_merit)) * 100
        plt.figure(figsize=(8, 6))
        plt.xlabel("Merit Score", fontsize=12)
        plt.ylabel("Cumulative Probability (%)", fontsize=12)
        plt.title("Cumulative Probability beyond Merit Score", fontsize=14)
        plt.plot(sorted_merit, cumulative_prob, linewidth=2)
        plt.gca().invert_xaxis()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("Monte_Carlo_Cumulative_Prob.png", dpi=150, bbox_inches="tight")
        plt.close()

        # Results dict
        results = {
            "method": "monte_carlo",
            "trials": trials,
            "baseline_merit": round(baseline_merit, 6),
            "merit_std": round(float(np.std(merit_ls)), 6),
            "merit_mean": round(float(np.mean(merit_ls)), 6),
            "merit_yield": {
                "99% > ": round(float(np.percentile(merit_ls, 1)), 4),
                "95% > ": round(float(np.percentile(merit_ls, 5)), 4),
                "90% > ": round(float(np.percentile(merit_ls, 10)), 4),
                "80% > ": round(float(np.percentile(merit_ls, 20)), 4),
                "70% > ": round(float(np.percentile(merit_ls, 30)), 4),
                "60% > ": round(float(np.percentile(merit_ls, 60)), 4),
                "50% > ": round(float(np.percentile(merit_ls, 50)), 4),
            },
            "merit_percentile": {
                "99% < ": round(float(np.percentile(merit_ls, 99)), 4),
                "95% < ": round(float(np.percentile(merit_ls, 95)), 4),
                "90% < ": round(float(np.percentile(merit_ls, 90)), 4),
                "80% < ": round(float(np.percentile(merit_ls, 80)), 4),
                "70% < ": round(float(np.percentile(merit_ls, 70)), 4),
                "60% < ": round(float(np.percentile(merit_ls, 60)), 4),
                "50% < ": round(float(np.percentile(merit_ls, 50)), 4),
            },
        }
        return results

    def tolerancing_wavefront(self, tolerance_params=None):
        """Use wavefront differential method to compute the tolerance.

        Wavefront differential method is proposed in [1], while the detailed implementation remains unknown. I (Xinge Yang) assume a symbolic differentiation is used to compute the gradient/Jacobian of the wavefront error. With AutoDiff, we can easily calculate Jacobian with gradient backpropagation, therefore I leave the implementation of this method as future work.

        Args:
            tolerance_params (dict): Tolerance parameters

        Returns:
            dict: Wavefront tolerance analysis results

        References:
            [1] Optical Design Tolerancing. CODE V.
        """
        pass
