# Copyright 2025 Xinge Yang and DeepLens contributors.
# This file is part of DeepLens (https://github.com/singer-yang/DeepLens).
#
# Licensed under the Apache License, Version 2.0.
# See LICENSE file in the project root for full license information.

"""Grating DOE parameterization.

This module implements a linear grating diffractive optical element (DOE).
A grating introduces a linear phase gradient across the surface, which
diffracts light into multiple diffraction orders.
"""

import torch
from deeplens.optics.diffractive_surface.diffractive import DiffractiveSurface


class Grating(DiffractiveSurface):
    """Grating diffractive optical element.
    
    A grating introduces a linear phase gradient defined by:
        phi(x, y) = alpha * (x * sin(theta) + y * cos(theta)) / norm_radii
    
    where:
        - theta: angle from y-axis to grating vector
        - alpha: slope of the grating (phase gradient strength)
        - norm_radii: normalization radius
    """

    def __init__(
        self,
        d,
        res=(2000, 2000),
        mat="fused_silica",
        wvln0=0.55,
        fab_ps=0.001,
        fab_step=16,
        theta=0.0,
        alpha=0.0,
        device="cpu",
    ):
        """Initialize Grating DOE.
        
        Args:
            d (float): Distance of the DOE surface. [mm]
            res (tuple or int): Resolution of the DOE, [w, h]. [pixel]
            mat (str): Material of the DOE.
            wvln0 (float): Design wavelength. [um]
            fab_ps (float): Fabrication pixel size. [mm]
            fab_step (int): Fabrication step.
            theta (float): Angle from y-axis to grating vector. [rad]
            alpha (float): Slope of the grating (phase gradient strength).
            device (str): Device to run the DOE.
        """
        super().__init__(
            d=d, res=res, mat=mat, wvln0=wvln0, fab_ps=fab_ps, fab_step=fab_step, device=device
        )

        # Grating parameters
        self.theta = torch.tensor(theta)  # angle from y-axis to grating vector
        self.alpha = torch.tensor(alpha)  # slope of the grating

        # Normalization radius (use half of the width)
        self.norm_radii = self.w / 2

        self.to(device)

    @classmethod
    def init_from_dict(cls, doe_dict):
        """Initialize Grating DOE from a dict.
        
        Args:
            doe_dict (dict): Dictionary containing DOE parameters.
            
        Returns:
            Grating: Initialized Grating DOE object.
        """
        return cls(
            d=doe_dict["d"],
            res=doe_dict["res"],
            mat=doe_dict.get("mat", "fused_silica"),
            wvln0=doe_dict.get("wvln0", 0.55),
            fab_ps=doe_dict.get("fab_ps", 0.001),
            fab_step=doe_dict.get("fab_step", 16),
            theta=doe_dict.get("theta", 0.0),
            alpha=doe_dict.get("alpha", 0.0),
        )

    def phase_func(self):
        """Get the phase map at design wavelength.
        
        The grating phase is a linear function of position:
            phi(x, y) = alpha * (x * sin(theta) + y * cos(theta)) / norm_radii
        
        Returns:
            phase (tensor): Phase map at design wavelength.
        """
        # Normalize coordinates
        x_norm = self.x / self.norm_radii
        y_norm = self.y / self.norm_radii

        # Calculate linear phase gradient
        phase = self.alpha * (
            x_norm * torch.sin(self.theta) + y_norm * torch.cos(self.theta)
        )

        return phase

    # =======================================
    # Optimization
    # =======================================
    def get_optimizer_params(self, lr=0.001):
        """Get parameters for optimization.

        Args:
            lr (float): Learning rate for grating parameters.
            
        Returns:
            list: List of parameter groups for optimizer.
        """
        self.theta.requires_grad = True
        self.alpha.requires_grad = True

        optimizer_params = [
            {"params": [self.theta], "lr": lr},
            {"params": [self.alpha], "lr": lr * 10},
        ]

        return optimizer_params

    # =======================================
    # IO
    # =======================================
    def surf_dict(self):
        """Return a dict of surface parameters.
        
        Returns:
            dict: Dictionary containing surface parameters.
        """
        surf_dict = super().surf_dict()
        surf_dict["theta"] = round(self.theta.item(), 6)
        surf_dict["alpha"] = round(self.alpha.item(), 6)
        surf_dict["norm_radii"] = round(self.norm_radii, 6)
        return surf_dict

    def save_ckpt(self, save_path="./grating_doe.pth"):
        """Save grating DOE parameters.
        
        Args:
            save_path (str): Path to save the checkpoint.
        """
        torch.save(
            {
                "param_model": "grating",
                "theta": self.theta.clone().detach().cpu(),
                "alpha": self.alpha.clone().detach().cpu(),
            },
            save_path,
        )

    def load_ckpt(self, load_path="./grating_doe.pth"):
        """Load grating DOE parameters.
        
        Args:
            load_path (str): Path to load the checkpoint from.
        """
        ckpt = torch.load(load_path)
        self.theta = ckpt["theta"].to(self.device)
        self.alpha = ckpt["alpha"].to(self.device)
