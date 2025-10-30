"""Base class for aspheric surfaces with shared functionality."""

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
