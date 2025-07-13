# Copyright (c) 2025 DeepLens Authors. All rights reserved.
#
# This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
#     The license is only for non-commercial use (commercial licenses can be obtained from authors).
#     The material is provided as-is, with no warranties whatsoever.
#     If you publish any code, data, or scientific work based on this, please cite our work.

"""Visualization functions for GeoLens."""

import matplotlib.pyplot as plt
import numpy as np
import torch

from deeplens.optics.basics import DEFAULT_WAVE, DEPTH, WAVE_RGB
from deeplens.optics.ray import Ray


class GeoLensVis:
    # ====================================================================================
    # Ray sampling functions for 2D layout
    # ====================================================================================
    @torch.no_grad()
    def sample_parallel_2D(
        self,
        fov=0.0,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        plane="meridional",
        entrance_pupil=True,
        depth=0.0,
    ):
        """Sample parallel rays (2D) in object space.

        Used for (1) drawing lens setup, (2) 2D geometric optics calculation, for example, refocusing to infinity

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to 0.0.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            plane (str, optional): sampling plane. Defaults to "meridional" (y-z plane).
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to True.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample points on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.calc_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        # Sample ray origins, shape [num_rays, 3]
        if plane == "sagittal":
            ray_o = torch.stack(
                (
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), 0),
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_o = torch.stack(
                (
                    torch.full((num_rays,), 0),
                    torch.linspace(-pupilr, pupilr, num_rays) * 0.99,
                    torch.full((num_rays,), pupilz),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Sample ray directions, shape [num_rays, 3]
        if plane == "sagittal":
            ray_d = torch.stack(
                (
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        elif plane == "meridional":
            ray_d = torch.stack(
                (
                    torch.zeros((num_rays,)),
                    torch.full((num_rays,), float(np.sin(np.deg2rad(fov)))),
                    torch.full((num_rays,), float(np.cos(np.deg2rad(fov)))),
                ),
                axis=-1,
            )
        else:
            raise ValueError(f"Invalid plane: {plane}")

        # Form rays and propagate to the target depth
        rays = Ray(ray_o, ray_d, wvln, device=self.device)
        rays.prop_to(depth)
        return rays

    @torch.no_grad()
    def sample_point_source_2D(
        self,
        fov=0.0,
        depth=DEPTH,
        num_rays=7,
        wvln=DEFAULT_WAVE,
        entrance_pupil=True,
    ):
        """Sample point source rays (2D) in object space.

        Used for (1) drawing lens setup.

        Args:
            fov (float, optional): incident angle (in degree). Defaults to 0.0.
            depth (float, optional): sampling depth. Defaults to DEPTH.
            num_rays (int, optional): ray number. Defaults to 7.
            wvln (float, optional): ray wvln. Defaults to DEFAULT_WAVE.
            entrance_pupil (bool, optional): whether to use entrance pupil. Defaults to False.

        Returns:
            ray (Ray object): Ray object. Shape [num_rays, 3]
        """
        # Sample point on the object plane
        ray_o = torch.tensor([depth * float(np.tan(np.deg2rad(fov))), 0.0, depth])
        ray_o = ray_o.unsqueeze(0).repeat(num_rays, 1)

        # Sample points (second point) on the pupil
        if entrance_pupil:
            pupilz, pupilr = self.calc_entrance_pupil()
        else:
            pupilz, pupilr = 0, self.surfaces[0].r

        x2 = torch.linspace(-pupilr, pupilr, num_rays) * 0.99
        y2 = torch.zeros_like(x2)
        z2 = torch.full_like(x2, pupilz)
        ray_o2 = torch.stack((x2, y2, z2), axis=1)

        # Form the rays
        ray_d = ray_o2 - ray_o
        ray = Ray(ray_o, ray_d, wvln, device=self.device)

        # Propagate rays to the sampling depth
        ray.prop_to(depth)
        return ray

    # ====================================================================================
    # Lens 2D layout
    # ====================================================================================
    def draw_layout(
        self,
        filename,
        depth=float("inf"),
        entrance_pupil=True,
        zmx_format=True,
        multi_plot=False,
        lens_title=None,
    ):
        """Plot 2D lens layout with ray tracing.

        Args:
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
            eff_foclen = self.foclen
            eq_foclen = self.calc_eqfl()
            fov_deg = 2 * self.hfov * 180 / torch.pi

            if self.aper_idx is not None:
                _, pupil_r = self.calc_entrance_pupil()
                fnum = eff_foclen / pupil_r / 2
                lens_title = f"FoV{round(fov_deg, 1)}({int(eq_foclen)}mm EqFocLen) - F/{round(fnum, 2)} - SensorDiag{round(self.r_sensor * 2, 2)}mm - FocLen{round(eff_foclen, 2)}mm"
            else:
                lens_title = f"FoV{round(fov_deg, 1)}({int(eq_foclen)}mm EqFocLen) - SensorDiag{round(self.r_sensor * 2, 2)}mm - FocLen{round(eff_foclen, 2)}mm"

        # Draw lens layout
        if not multi_plot:
            colors_list = ["#CC0000", "#006600", "#0066CC"]
            views = np.linspace(0, float(np.rad2deg(self.hfov) * 0.99), num=num_views)
            ax, fig = self.draw_lens_2d(zmx_format=zmx_format)

            for i, view in enumerate(views):
                # Sample rays, shape (num_view, num_rays, 3)
                if depth == float("inf"):
                    ray = self.sample_parallel_2D(
                        fov=view,
                        wvln=WAVE_RGB[2 - i],
                        num_rays=num_rays,
                        entrance_pupil=entrance_pupil,
                        depth=-1.0,
                        plane="sagittal",
                    )  # shape (num_rays, 3)
                else:
                    ray = self.sample_point_source_2D(
                        fov=view,
                        depth=depth,
                        num_rays=num_rays,
                        wvln=WAVE_RGB[2 - i],
                        entrance_pupil=entrance_pupil,
                    )  # shape (num_rays, 3)
                    ray.prop_to(-1.0)

                # Trace rays to sensor and plot ray paths
                _, ray_o_record = self.trace2sensor(ray=ray, record=True)
                ax, fig = self.draw_ray_2d(
                    ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                )

            ax.axis("off")
            ax.set_title(lens_title, fontsize=8)
            if filename.endswith(".png"):
                fig.savefig(filename, format="png", dpi=600)
            else:
                raise ValueError("Invalid file extension")
            plt.close()

        else:
            views = np.linspace(0, np.rad2deg(self.hfov) * 0.99, num=num_views)
            colors_list = ["#CC0000", "#006600", "#0066CC"]
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle(lens_title)

            for i, wvln in enumerate(WAVE_RGB):
                ax = axs[i]
                ax, fig = self.draw_lens_2d(ax=ax, fig=fig, zmx_format=zmx_format)

                for view in views:
                    if depth == float("inf"):
                        ray = self.sample_parallel_2D(
                            fov=view,
                            num_rays=num_rays,
                            wvln=wvln,
                            entrance_pupil=entrance_pupil,
                            plane="sagittal",
                        )  # shape (num_rays, 3)
                    else:
                        ray = self.sample_point_source_2D(
                            fov=view,
                            depth=depth,
                            num_rays=num_rays,
                            wvln=wvln,
                            entrance_pupil=entrance_pupil,
                        )  # shape (num_rays, 3)

                    ray_out, ray_o_record = self.trace2sensor(ray=ray, record=True)
                    ax, fig = self.draw_ray_2d(
                        ray_o_record, ax=ax, fig=fig, color=colors_list[i]
                    )
                    ax.axis("off")

            if filename.endswith(".png"):
                fig.savefig(filename, format="png", dpi=300)
            else:
                raise ValueError("Invalid file extension")
            plt.close()

    def draw_lens_2d(
        self,
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
        for i, s in enumerate(self.surfaces):
            s.draw_widget(ax)

        # Connect two surfaces
        for i in range(len(self.surfaces)):
            if self.surfaces[i].mat2.n > 1.1:
                s_prev = self.surfaces[i]
                s = self.surfaces[i + 1]

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
            [self.d_sensor.item(), self.d_sensor.item()],
            [-self.r_sensor, self.r_sensor],
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

    def draw_ray_2d(self, ray_o_record, ax, fig, color="b"):
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

    # ====================================================================================
    # Lens 3D layout
    # ====================================================================================
    def draw_layout_3d(self, filename=None, view_angle=30, show=False):
        """Draw 3D layout of the lens system.

        Args:
            filename (str, optional): Path to save the figure. Defaults to None.
            view_angle (int): Viewing angle for the 3D plot
            show (bool): Whether to display the figure

        Returns:
            fig, ax: Matplotlib figure and axis objects
        """
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Enable depth sorting for proper occlusion
        ax.set_proj_type(
            "persp"
        )  # Use perspective projection for better depth perception

        # Draw each surface
        for i, surf in enumerate(self.surfaces):
            surf.draw_widget3D(ax)

            # Connect current surface with previous surface if material is not air
            if i > 0 and self.surfaces[i - 1].mat2.get_name() != "air":
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
                prev_surf = self.surfaces[i - 1]
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
                x_mesh = (
                    prev_edge_x[None, :] * (1 - t_mesh) + curr_edge_x[None, :] * t_mesh
                )
                y_mesh = (
                    prev_edge_y[None, :] * (1 - t_mesh) + curr_edge_y[None, :] * t_mesh
                )
                z_mesh = (
                    prev_edge_z[None, :] * (1 - t_mesh) + curr_edge_z[None, :] * t_mesh
                )

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
        if hasattr(self, "sensor_size") and hasattr(self, "d_sensor"):
            # Get sensor dimensions
            sensor_width = self.sensor_size[0]
            sensor_height = self.sensor_size[1]
            sensor_z = self.d_sensor.item()

            # Create sensor vertices
            half_width = sensor_width / 2
            half_height = sensor_height / 2

            # Define the corners of the rectangle
            x = np.array(
                [-half_width, half_width, half_width, -half_width, -half_width]
            )
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

    # ====================================================================================
    # Lens 3D barrier generation
    # ====================================================================================
    def create_barrier(
        self, filename, barrier_thickness=1.0, ring_height=0.5, ring_size=1.0
    ):
        """Create a 3D barrier for the lens system.

        Args:
            filename: Path to save the figure
            barrier_thickness: Thickness of the barrier
            ring_height: Height of the annular ring
            ring_size: Size of the annular ring
        """
        barriers = []
        rings = []

        # Create barriers
        barrier_z = 0.0
        barrier_r = 0.0
        barrier_length = 0.0
        for i in range(len(self.surfaces)):
            barrier_r = max(self.surfaces[i].r, barrier_r)

            if self.surfaces[i].mat2.get_name() != "air":
                # Update the barrier radius
                # barrier_r = max(geolens.surfaces[i].r, barrier_r)
                pass
            else:
                # Extend the barrier till middle of the air space to the next surface
                max_curr_surf_d = self.surfaces[i].d.item() + max(
                    self.surfaces[i].surface_sag(0.0, self.surfaces[i].r), 0.0
                )
                if i < len(self.surfaces) - 1:
                    min_next_surf_d = self.surfaces[i + 1].d.item() + min(
                        self.surfaces[i + 1].surface_sag(0.0, self.surfaces[i + 1].r),
                        0.0,
                    )
                    extra_space = (min_next_surf_d - max_curr_surf_d) / 2
                else:
                    min_next_surf_d = self.d_sensor.item()
                    extra_space = min_next_surf_d - max_curr_surf_d

                barrier_length = max_curr_surf_d + extra_space - barrier_z

                # Create a barrier
                barrier = {
                    "pos_z": barrier_z,
                    "pos_r": barrier_r,
                    "length": barrier_length,
                    "thickness": barrier_thickness,
                }
                barriers.append(barrier)

                # Reset the barrier parameters
                barrier_z = barrier_length + barrier_z
                barrier_r = 0.0
                barrier_length = 0.0

        # # Create rings
        # for i in range(len(geolens.surfaces)):
        #     if geolens.surfaces[i].mat2.get_name() != "air":
        #         ring = {
        #             "pos_z": geolens.surfaces[i].d.item(),

        # Plot lens layout
        ax, fig = self.draw_layout()

        # Plot barrier
        barrier_z_ls = []
        barrier_r_ls = []
        for b in barriers:
            barrier_z_ls.append(b["pos_z"])
            barrier_z_ls.append(b["pos_z"] + b["length"])
            barrier_r_ls.append(b["pos_r"])
            barrier_r_ls.append(b["pos_r"])
        ax.plot(barrier_z_ls, barrier_r_ls, "green", linewidth=1.0)
        ax.plot(barrier_z_ls, [-i for i in barrier_r_ls], "green", linewidth=1.0)

        # Plot rings

        fig.savefig(filename, format="png", dpi=300)
        plt.close()

        pass
