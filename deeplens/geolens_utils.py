"""Utils for GeoLens class."""

import os
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from .optics.geometric_surface import Aperture, Aspheric, Spheric
from .optics.materials import SELLMEIER_TABLE
from .optics.basics import WAVE_RGB


# ====================================================================================
# ZEMAX file IO
# ====================================================================================
def read_zmx(filename="./test.zmx"):
    """Load the lens from .zmx file."""
    # Initialize a GeoLens
    from .geolens import GeoLens

    geolens = GeoLens()

    # Read .zmx file
    try:
        with open(filename, "r", encoding="utf-8") as file:
            lines = file.readlines()
    except UnicodeDecodeError:
        with open(filename, "r", encoding="utf-16") as file:
            lines = file.readlines()

    # Iterate through the lines and extract SURF dict
    surfs_dict = {}
    current_surf = None
    for line in lines:
        if line.startswith("SURF"):
            current_surf = int(line.split()[1])
            surfs_dict[current_surf] = {}
        elif current_surf is not None and line.strip() != "":
            if len(line.strip().split(maxsplit=1)) == 1:
                continue
            else:
                key, value = line.strip().split(maxsplit=1)
                if key == "PARM":
                    new_key = "PARM" + value.split()[0]
                    new_value = value.split()[1]
                    surfs_dict[current_surf][new_key] = new_value
                else:
                    surfs_dict[current_surf][key] = value

    # Read the extracted data from each SURF
    geolens.surfaces = []
    d = 0.0
    for surf_idx, surf_dict in surfs_dict.items():
        if surf_idx > 0 and surf_idx < current_surf:
            mat2 = (
                f"{surf_dict['GLAS'].split()[3]}/{surf_dict['GLAS'].split()[4]}"
                if "GLAS" in surf_dict
                else "air"
            )
            surf_r = float(surf_dict["DIAM"].split()[0]) if "DIAM" in surf_dict else 1.0
            surf_c = float(surf_dict["CURV"].split()[0]) if "CURV" in surf_dict else 0.0
            surf_d_next = (
                float(surf_dict["DISZ"].split()[0]) if "DISZ" in surf_dict else 0.0
            )

            if surf_dict["TYPE"] == "STANDARD":
                # Aperture
                if surf_c == 0.0 and mat2 == "air":
                    s = Aperture(r=surf_r, d=d)

                # Spherical surface
                else:
                    s = Spheric(c=surf_c, r=surf_r, d=d, mat2=mat2)

            # Aspherical surface
            elif surf_dict["TYPE"] == "EVENASPH":
                raise NotImplementedError()
                s = Aspheric()

            else:
                print(f"Surface type {surf_dict['TYPE']} not implemented.")
                continue

            geolens.surfaces.append(s)
            d += surf_d_next

        elif surf_idx == current_surf:
            # Image sensor
            geolens.r_sensor = float(surf_dict["DIAM"].split()[0])

        else:
            pass

    geolens.d_sensor = torch.tensor(d)
    return geolens


def write_zmx(geolens, filename="./test.zmx"):
    """Write the lens into .zmx file."""
    lens_zmx_str = ""
    ENPD = geolens.calc_entrance_pupil()[1] * 2
    # Head string
    head_str = f"""VERS 190513 80 123457 L123457
MODE SEQ
NAME 
PFIL 0 0 0
LANG 0
UNIT MM X W X CM MR CPMM
ENPD {ENPD}
ENVD 2.0E+1 1 0
GFAC 0 0
GCAT OSAKAGASCHEMICAL MISC
XFLN 0. 0. 0.
YFLN 0.0 {0.707 * geolens.hfov * 57.3} {0.99 * geolens.hfov * 57.3}
WAVL 0.4861327 0.5875618 0.6562725
RAIM 0 0 1 1 0 0 0 0 0
PUSH 0 0 0 0 0 0
SDMA 0 1 0
FTYP 0 0 3 3 0 0 0
ROPD 2
PICB 1
PWAV 2
POLS 1 0 1 0 0 1 0
GLRS 1 0
GSTD 0 100.000 100.000 100.000 100.000 100.000 100.000 0 1 1 0 0 1 1 1 1 1 1
NSCD 100 500 0 1.0E-3 5 1.0E-6 0 0 0 0 0 0 1000000 0 2
COFN QF "COATING.DAT" "SCATTER_PROFILE.DAT" "ABG_DATA.DAT" "PROFILE.GRD"
COFN COATING.DAT SCATTER_PROFILE.DAT ABG_DATA.DAT PROFILE.GRD
SURF 0
TYPE STANDARD
CURV 0.0
DISZ INFINITY
"""
    lens_zmx_str += head_str

    # Surface string
    for i, s in enumerate(geolens.surfaces):
        d_next = (
            geolens.surfaces[i + 1].d - geolens.surfaces[i].d
            if i < len(geolens.surfaces) - 1
            else geolens.d_sensor - geolens.surfaces[i].d
        )
        surf_str = s.zmx_str(surf_idx=i + 1, d_next=d_next)
        lens_zmx_str += surf_str

    # Sensor string
    sensor_str = f"""SURF {i + 2}
TYPE STANDARD
CURV 0.
DISZ 0.0
DIAM {geolens.r_sensor}
"""
    lens_zmx_str += sensor_str

    # Write lens zmx string into file
    with open(filename, "w") as f:
        f.writelines(lens_zmx_str)
        f.close()


# ====================================================================================
# Create lens
# ====================================================================================
def create_lens(
    foclen,
    fov,
    fnum,
    flange,
    thickness=None,
    lens_type=[["Spheric", "Spheric"], ["Aperture"], ["Spheric", "Aspheric"]],
    save_dir="./",
):
    """Create a flat starting point for camera lens design.

    Contributor: Rayengineer

    Args:
        foclen: Focal length in mm.
        fov: Diagonal field of view in degrees.
        fnum: Maximum f number.
        flange: Distance from last surface to sensor.
        thickness: Total thickness if specified.
        lens_type: List of surface types defining each lens element and aperture.
    """
    from .geolens import GeoLens

    # Compute lens parameters
    aper_r = foclen / fnum / 2
    imgh = 2 * foclen * float(np.tan(fov / 2 / 57.3))
    if thickness is None:
        thickness = foclen + flange
    d_opt = thickness - flange

    # Materials
    mat_names = list(SELLMEIER_TABLE.keys())
    remove_materials = ["air", "vacuum", "occluder"]
    for mat in remove_materials:
        if mat in mat_names:
            mat_names.remove(mat)

    # Create lens
    lens = GeoLens()
    surfaces = lens.surfaces

    d_total = 0.0
    for elem_type in lens_type:
        if elem_type == "Aperture":
            d_next = (torch.rand(1) + 0.5).item()
            surfaces.append(Aperture(r=aper_r, d=d_total))
            d_total += d_next

        elif isinstance(elem_type, list):
            if len(elem_type) == 1 and elem_type[0] == "Aperture":
                d_next = (torch.rand(1) + 0.5).item()
                surfaces.append(Aperture(r=aper_r, d=d_total))
                d_total += d_next

            elif len(elem_type) in [2, 3]:
                for i, surface_type in enumerate(elem_type):
                    if i == len(elem_type) - 1:
                        mat = "air"
                        d_next = (torch.rand(1) + 0.5).item()
                    else:
                        mat = random.choice(mat_names)
                        d_next = (torch.rand(1) + 1.0).item()

                    surfaces.append(
                        create_surface(surface_type, d_total, aper_r, imgh, mat)
                    )
                    d_total += d_next
            else:
                raise Exception("Lens element type not supported yet.")
        else:
            raise Exception("Lens type format not correct.")

    # Normalize optical part total thickness
    d_opt_actual = d_total - d_next
    for s in surfaces:
        s.d = s.d / d_opt_actual * d_opt

    # Lens calculation
    lens = lens.to(lens.device)
    lens.d_sensor = torch.tensor(thickness, dtype=torch.float32).to(lens.device)
    lens.r_sensor = imgh / 2
    lens.set_sensor(sensor_res=lens.sensor_res)
    lens.post_computation()

    # Save lens
    filename = f"starting_point_f{foclen}mm_imgh{imgh}_fnum{fnum}.json"
    lens.write_lens_json(os.path.join(save_dir, filename))

    return lens


def create_surface(surface_type, d_total, aper_r, imgh, mat):
    """Create a surface object based on the surface type."""
    if mat == "air":
        c = -float(np.random.rand()) * 0.001
    else:
        c = float(np.random.rand()) * 0.001
    r = max(imgh / 2, aper_r)

    if surface_type == "Spheric":
        return Spheric(r=r, d=d_total, c=c, mat2=mat)
    elif surface_type == "Aspheric":
        ai = np.random.randn(7).astype(np.float32) * 1e-30
        k = float(np.random.rand()) * 0.001
        return Aspheric(r=r, d=d_total, c=c, ai=ai, k=k, mat2=mat)
    else:
        raise Exception("Surface type not supported yet.")


# ====================================================================================
# Draw lens layout
# ====================================================================================
def draw_lens_layout(
    geolens,
    filename,
    depth=float("inf"),
    entrance_pupil=True,
    zmx_format=True,
    multi_plot=False,
    lens_title=None,
):
    """Plot lens layout with ray tracing.

    Args:
        geolens: GeoLens instance
        filename: Output filename
        depth: Depth for ray tracing
        entrance_pupil: Whether to use entrance pupil
        zmx_format: Whether to use ZMX format
        multi_plot: Whether to create multiple plots
        lens_title: Title for the lens plot
    """
    num_rays = 11
    num_views = 3

    # Lens title
    if lens_title is None:
        if geolens.aper_idx is not None:
            fnum = geolens.foclen / geolens.calc_entrance_pupil()[1] / 2
            lens_title = f"FoV{round(2 * geolens.hfov * 57.3, 1)}({int(geolens.calc_eqfl())}mm EQFL)_F/{round(fnum, 2)}_DIAG{round(geolens.r_sensor * 2, 2)}mm_FFL{round(geolens.foclen, 2)}mm"
        else:
            lens_title = f"FoV{round(2 * geolens.hfov * 57.3, 1)}({int(geolens.calc_eqfl())}mm EQFL)_DIAG{round(geolens.r_sensor * 2, 2)}mm_FFL{round(geolens.foclen, 2)}mm"

    # Draw lens layout
    if not multi_plot:
        colors_list = ["#CC0000", "#006600", "#0066CC"]
        views = np.linspace(0, float(np.rad2deg(geolens.hfov) * 0.99), num=num_views)
        ax, fig = draw_setup_2d(geolens, zmx_format=zmx_format)

        for i, view in enumerate(views):
            # Sample rays, shape (num_view, num_rays, 3)
            if depth == float("inf"):
                ray = geolens.sample_parallel_2D(
                    fov=view,
                    wvln=WAVE_RGB[2 - i],
                    num_rays=num_rays,
                    entrance_pupil=entrance_pupil,
                    depth=-1.0,
                )  # shape (num_rays, 3)
            else:
                ray = geolens.sample_point_source_2D(
                    fov=view,
                    depth=depth,
                    num_rays=num_rays,
                    wvln=WAVE_RGB[2 - i],
                    entrance_pupil=entrance_pupil,
                )  # shape (num_rays, 3)

            # Trace rays to sensor and plot ray paths
            _, ray_o_record = geolens.trace2sensor(ray=ray, record=True)
            ax, fig = draw_raytraces_2d(
                ray_o_record, ax=ax, fig=fig, color=colors_list[i]
            )

        ax.axis("off")
        ax.set_title(lens_title, fontsize=10)
        if filename.endswith(".png"):
            fig.savefig(filename, format="png", dpi=600)
        else:
            raise ValueError("Invalid file extension")
        plt.close()

    else:
        views = np.linspace(0, np.rad2deg(geolens.hfov) * 0.99, num=num_views)
        colors_list = ["#CC0000", "#006600", "#0066CC"]
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(lens_title)

        for i, wvln in enumerate(WAVE_RGB):
            ax = axs[i]
            ax, fig = draw_setup_2d(geolens, ax=ax, fig=fig, zmx_format=zmx_format)

            for view in views:
                if depth == float("inf"):
                    ray = geolens.sample_parallel_2D(
                        fov=view,
                        num_rays=num_rays,
                        wvln=wvln,
                        entrance_pupil=entrance_pupil,
                    )  # shape (num_rays, 3)
                else:
                    ray = geolens.sample_point_source_2D(
                        fov=view,
                        depth=depth,
                        num_rays=num_rays,
                        wvln=wvln,
                        entrance_pupil=entrance_pupil,
                    )  # shape (num_rays, 3)

                ray_out, ray_o_record = geolens.trace2sensor(ray=ray, record=True)
                ax, fig = draw_raytraces_2d(
                    ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                )
                ax.axis("off")

        if filename.endswith(".png"):
            fig.savefig(filename, format="png", dpi=300)
        else:
            raise ValueError("Invalid file extension")
        plt.close()


def draw_setup_2d(
    geolens,
    ax=None,
    fig=None,
    color="k",
    linestyle="-",
    zmx_format=False,
    fix_bound=False,
):
    """Draw lens layout in a 2D plot."""

    # If no ax is given, generate a new one.
    if ax is None and fig is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Draw lens surfaces
    for i, s in enumerate(geolens.surfaces):
        s.draw_widget(ax)

    # Connect two surfaces
    for i in range(len(geolens.surfaces)):
        if geolens.surfaces[i].mat2.n > 1.1:
            s_prev = geolens.surfaces[i]
            s = geolens.surfaces[i + 1]

            r_prev = float(s_prev.r)
            r = float(s.r)
            sag_prev = s_prev.surface_with_offset(r_prev, 0.0).item()
            sag = s.surface_with_offset(r, 0.0).item()

            if zmx_format:
                if r > r_prev:
                    z = np.array([sag_prev, sag_prev, sag])
                    x = np.array([r_prev, r, r])
                else:
                    z = np.array([sag_prev, sag, sag])
                    x = np.array([r_prev, r, r])
            else:
                z = np.array([sag_prev, sag])
                x = np.array([r_prev, r])

            ax.plot(z, -x, color, linewidth=0.75)
            ax.plot(z, x, color, linewidth=0.75)
            s_prev = s

    # Draw sensor
    ax.plot(
        [geolens.d_sensor.item(), geolens.d_sensor.item()],
        [-geolens.r_sensor, geolens.r_sensor],
        color,
    )

    # Set figure size
    if fix_bound:
        ax.set_aspect("equal")
        ax.set_xlim(-1, 7)
        ax.set_ylim(-4, 4)
    else:
        ax.set_aspect("equal", adjustable="datalim", anchor="C")
        ax.minorticks_on()
        ax.set_xlim(-0.5, 7.5)
        ax.set_ylim(-4, 4)
        ax.autoscale()

    return ax, fig


def draw_raytraces_2d(ray_o_record, ax, fig, color="b"):
    """Plot ray paths.

    Args:
        ray_o_record (list): list of intersection points.
        ax (matplotlib.axes.Axes): matplotlib axes.
        fig (matplotlib.figure.Figure): matplotlib figure.
    """
    # shape (num_view, num_rays, num_path, 2)
    ray_o_record = torch.stack(ray_o_record, dim=-2).cpu().numpy()
    if ray_o_record.ndim == 3:
        ray_o_record = ray_o_record[None, ...]

    for idx_view in range(ray_o_record.shape[0]):
        for idx_ray in range(ray_o_record.shape[1]):
            ax.plot(
                ray_o_record[idx_view, idx_ray, :, 2],
                ray_o_record[idx_view, idx_ray, :, 0],
                color,
                linewidth=0.8,
            )

            # ax.scatter(
            #     ray_o_record[idx_view, idx_ray, :, 2],
            #     ray_o_record[idx_view, idx_ray, :, 0],
            #     "b",
            #     marker="x",
            # )

    return ax, fig


def draw_layout_3d(geolens, filename=None, figsize=(10, 6), view_angle=30, show=True):
    """Draw 3D layout of the lens system.

    Args:
        geolens: GeoLens instance
        filename (str, optional): Path to save the figure. Defaults to None.
        figsize (tuple): Figure size
        view_angle (int): Viewing angle for the 3D plot
        show (bool): Whether to display the figure

    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Enable depth sorting for proper occlusion
    ax.set_proj_type("persp")  # Use perspective projection for better depth perception

    # Draw each surface
    for i, surf in enumerate(geolens.surfaces):
        surf.draw_widget3D(ax)

        # Connect current surface with previous surface if material is not air
        if i > 0 and geolens.surfaces[i - 1].mat2.get_name() != "air":
            # Get edge points of current and previous surfaces
            theta = np.linspace(0, 2 * np.pi, 256)

            # Current surface edge
            curr_edge_x = surf.r * np.cos(theta)
            curr_edge_y = surf.r * np.sin(theta)
            curr_edge_z = np.array(
                [
                    surf.surface_with_offset(
                        torch.tensor(curr_edge_x[j], device=surf.device),
                        torch.tensor(curr_edge_y[j], device=surf.device),
                    ).item()
                    for j in range(len(theta))
                ]
            )

            # Previous surface edge
            prev_surf = geolens.surfaces[i - 1]
            prev_edge_x = prev_surf.r * np.cos(theta)
            prev_edge_y = prev_surf.r * np.sin(theta)
            prev_edge_z = np.array(
                [
                    prev_surf.surface_with_offset(
                        torch.tensor(prev_edge_x[j], device=prev_surf.device),
                        torch.tensor(prev_edge_y[j], device=prev_surf.device),
                    ).item()
                    for j in range(len(theta))
                ]
            )

            # Create a cylindrical surface connecting the two edges
            theta_mesh, t_mesh = np.meshgrid(theta, np.array([0, 1]))

            # Interpolate between previous and current surface edges
            x_mesh = prev_edge_x[None, :] * (1 - t_mesh) + curr_edge_x[None, :] * t_mesh
            y_mesh = prev_edge_y[None, :] * (1 - t_mesh) + curr_edge_y[None, :] * t_mesh
            z_mesh = prev_edge_z[None, :] * (1 - t_mesh) + curr_edge_z[None, :] * t_mesh

            # Plot the connecting surface with sort_zpos for proper occlusion
            surf = ax.plot_surface(
                z_mesh,
                x_mesh,
                y_mesh,
                color="lightblue",
                alpha=0.3,
                edgecolor="lightblue",
                linewidth=0.5,
                antialiased=True,
            )
            # Set the zorder based on the mean z position for better occlusion
            surf._sort_zpos = np.mean(z_mesh)

    # Draw sensor as a rectangle
    if hasattr(geolens, "sensor_size") and hasattr(geolens, "d_sensor"):
        # Get sensor dimensions
        sensor_width = geolens.sensor_size[0]
        sensor_height = geolens.sensor_size[1]
        sensor_z = geolens.d_sensor.item()

        # Create sensor vertices
        half_width = sensor_width / 2
        half_height = sensor_height / 2

        # Define the corners of the rectangle
        x = np.array([-half_width, half_width, half_width, -half_width, -half_width])
        y = np.array(
            [-half_height, -half_height, half_height, half_height, -half_height]
        )
        z = np.full_like(x, sensor_z)

        # Plot the sensor rectangle
        ax.plot(z, x, y, color="black", linewidth=1.5)

        # Add a semi-transparent surface for the sensor
        sensor_x, sensor_y = np.meshgrid(
            np.linspace(-half_width, half_width, 2),
            np.linspace(-half_height, half_height, 2),
        )
        sensor_z = np.full_like(sensor_x, sensor_z)
        sensor_surf = ax.plot_surface(
            sensor_z,
            sensor_x,
            sensor_y,
            color="gray",
            alpha=0.3,
            edgecolor="black",
            linewidth=0.5,
        )
        # Set the zorder for the sensor
        sensor_surf._sort_zpos = sensor_z.mean()

    # Set axis properties
    ax.set_xlabel("Z")
    ax.set_ylabel("X")
    ax.set_zlabel("Y")
    ax.view_init(elev=20, azim=-view_angle - 90)

    # Make all axes have the same scale (unit step size)
    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect("equal")

    # Enable depth sorting for proper occlusion
    from matplotlib.collections import PathCollection

    for c in ax.collections:
        if isinstance(c, PathCollection):
            c.set_sort_zpos(c.get_offsets()[:, 2].mean())

    plt.tight_layout()

    if filename:
        fig.savefig(f"{filename}.png", format="png", dpi=300)

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax
