from deeplens.geolens import GeoLens
from deeplens.optics.geometric_surface import Prism

# A thin lens
lens = GeoLens(filename="./thinlens.json")

# Add a prism to the lenss
prism = Prism(r=7.5, d=20.0, mirror_angle=45.0, mat2="bk7", device=lens.device)
lens.surfaces.append(prism)

# Ray tracing (after the thinlens, the input ray should be parallel to +z entering the prism)
ray = lens.sample_from_points(points=[[0.0, 0.0, -10.0]], num_rays=1024)
ray, _ = lens.trace(ray)

# Ray direction should point up
print(ray.d)