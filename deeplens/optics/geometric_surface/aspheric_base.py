"""Base class for aspheric surfaces with shared functionality."""

import numpy as np
from deeplens.optics.geometric_surface.base import Surface


class AsphericBase(Surface):
    """Base class for aspheric surfaces providing shared initialization methods."""

    @classmethod
    def init_from_dict(cls, surf_dict):
        """Initialize aspheric surface from dictionary.
        
        Args:
            surf_dict (dict): Dictionary containing surface parameters
                - r: radius of the surface
                - d: distance from the origin to the surface
                - c or roc: curvature or radius of curvature
                - k: conic constant
                - ai: aspheric coefficients
                - mat2: material of the second medium
        
        Returns:
            AsphericBase: Initialized aspheric surface instance
        """
        if "roc" in surf_dict:
            c = 1 / surf_dict["roc"]
        else:
            c = surf_dict["c"]

        return cls(
            r=surf_dict["r"],
            d=surf_dict["d"],
            c=c,
            k=surf_dict["k"],
            ai=surf_dict["ai"],
            mat2=surf_dict["mat2"],
        )

    # =======================================
    # Tolerancing
    # =======================================
    def sample_tolerance(self):
        """Randomly perturb surface parameters to simulate manufacturing errors."""
        super().sample_tolerance()
        self.c_error = float(np.random.randn() * self.c_tole)
        self.k_error = float(np.random.randn() * self.k_tole)

    def zero_tolerance(self):
        """Clear perturbation."""
        super().zero_tolerance()
        self.c_error = 0.0
        self.k_error = 0.0

    def sensitivity_score(self):
        """Calculate tolerance sensitivity score."""
        score_dict = super().sensitivity_score()
        score_dict.update(
            {
                "c_grad": round(self.c.grad.item(), 6),
                "c_score": round((self.c_tole**2 * self.c.grad**2).item(), 6),
            }
        )
        score_dict.update(
            {
                "k_grad": round(self.k.grad.item(), 6),
                "k_score": round((self.k_tole**2 * self.k.grad**2).item(), 6),
            }
        )
        return score_dict

