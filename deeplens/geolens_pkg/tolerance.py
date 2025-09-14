# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

import torch
import numpy as np
from tqdm import tqdm


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

    def tolerancing_sensitivity(self, tolerance_params=None):
        """Use sensitivity analysis (1st order gradient) to compute the tolerance score.
        
        References:
            [1] Page 10 from: https://wp.optics.arizona.edu/optomech/wp-content/uploads/sites/53/2016/08/8-Tolerancing-1.pdf
        """
        # Initialize tolerance
        self.init_tolerance(tolerance_params=tolerance_params)
        
        # AutoDiff to compute the gradient
        self.get_optimizer_params()
        loss = self.loss_rms()
        loss.backward()

        # Calculate tolerancing score
        tolerancing_score = 0.0
        for i in range(len(self.surfaces)):
            tolerancing_score += self.surfaces[i].sensitivity_score()

        # Toleranced loss
        loss_rss = torch.sqrt(loss**2 + tolerancing_score).item()
        return loss_rss

    def tolerancing_monte_carlo(self, trials=1000, tolerance_params=None):
        """Use Monte Carlo simulation to compute the tolerance.

        TODO: we can multiplex sampled rays to improve the speed.

        Args:
            trials (int): Number of Monte Carlo trials
            tolerance_params (dict): Tolerance parameters

        Returns:
            dict: Monte Carlo tolerance analysis results

        References:
            [1] https://optics.ansys.com/hc/en-us/articles/43071088477587-How-to-analyze-your-tolerance-results
        """
        self.init_tolerance(tolerance_params=tolerance_params)
        # merit_func = self.loss_rms
        merit_func = self.loss_infocus
        
        # Baseline merit
        self.refocus()
        baseline_merit = merit_func(target=0.0).item()

        # Monte Carlo simulation
        merit_ls = []
        with torch.no_grad():
            for i in tqdm(range(trials)):
                # Sample a random perturbation
                self.sample_tolerance()

                # Evaluate perturbed performance
                perturbed_merit = merit_func(target=0.0)
                merit_ls.append(perturbed_merit.item())

                # Clear perturbation
                self.zero_tolerance()

        # Analyze results
        merit_ls = np.array(merit_ls)
        merit_mean = np.mean(merit_ls)
        merit_variations = np.abs(merit_ls - baseline_merit)
        merit_std = np.std(merit_variations)

        # Calculate yield rates (assuming the smaller the merit, the better the performance)
        performance_drops = (merit_ls - baseline_merit) / baseline_merit
        yield_95 = np.sum(performance_drops <= 0.05) / trials
        yield_90 = np.sum(performance_drops <= 0.10) / trials
        yield_80 = np.sum(performance_drops <= 0.20) / trials

        # Return results
        results = {
            "method": "monte_carlo",
            "trials": len(merit_variations),
            "baseline_merit": round(baseline_merit, 6),
            "merit_std": round(float(merit_std), 6),
            "merit_mean": round(float(merit_mean), 6),
            "merit_percentiles": {
                "95%": round(float(np.percentile(merit_variations, 95)), 4),
                "99%": round(float(np.percentile(merit_variations, 99)), 4),
                "99.9%": round(float(np.percentile(merit_variations, 99.9)), 4),
            },
            "yield_rates": {
                ">95%": f"{round(float(yield_95 * 100.0), 4)}%",
                ">90%": f"{round(float(yield_90 * 100.0), 4)}%", 
                ">80%": f"{round(float(yield_80 * 100.0), 4)}%",
            },
        }
        print(results)
        return results
