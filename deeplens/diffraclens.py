""" Pure diffractive lens with all surface represented for wave optics.

    1. Thinlens + DOE + Sensor
    2. Real lens + DOE + Sensor
"""
from .optics.basics import *
from .optics.wave import *
from .optics.waveoptics_utils import *
from .optics.surfaces_diffractive import *
from .utils import *

from torchvision.utils import save_image, make_grid

class DoeThinLens(DeepObj):
    """ DOE with thin lens model.

        Reference: https://www.lighttrans.com/use-cases/application/chromatic-aberration-correction.html
    """
    def __init__(self, thinlens=None, doe=None, sensor=None, aper=None, device=DEVICE):
        super().__init__()

        self.aper = aper
        self.thinlens = thinlens
        self.doe = doe
        self.sensor = sensor

        if aper is None:
            self.surfaces = [self.thinlens, self.doe, self.sensor]
        else:
            self.surfaces = [self.aper, self.thinlens, self.doe, self.sensor]
        
        assert doe.l == sensor.l, 'DOE and sensor should have the same physical size.'
        self.to(device)

    def load_example(self):
        """ Generate an example lens group. The lens is focused to infinity.
        """
        self.thinlens = ThinLens(foclen=50, d=0, r=12.7)
        self.doe = DOE(d=40, l=4, res=1024)
        self.sensor = Sensor(d=50, l=4)

        self.surfaces = [self.thinlens, self.doe, self.sensor]

    def load_example2(self):
        """ Generate an example lens group. The lens is focused to -100mm.
        """
        self.thinlens = ThinLens(foclen=50, d=0, r=12.7)
        self.doe = DOE(d=40, l=4, res=1024)
        self.sensor = Sensor(d=100, l=4)

        self.surfaces = [self.thinlens, self.doe, self.sensor]


    def forward(self, field):
        """ Propagate a wavefront through the lens group.

        Args:
            field (Field): Input wavefront.

        Returns:
            field (torch.tensor): Output energy distribution. Shape of [H_sensor, W_sensor]
        """
        for surf in self.surfaces:
            field = surf(field)

        return field


    # =============================================
    # PSF related functions
    # =============================================   

    def psf(self, point=[0, 0, -10000.], ks=101, wvln=0.589):
        """ Calculate monochromatic point PSF by wave propagation approach.

            For the shifted phase issue, refer to "Modeling off-axis diffraction with the least-sampling angular spectrum method".

        Args:
            point (list, optional): Point source position. Defaults to [0, 0, -10000].
            ks (int, optional): PSF kernel size. Defaults to 256.
            wvln (float, optional): wvln. Defaults to 0.55.
            padding (bool, optional): Whether to pad the PSF. Defaults to True.

        Returns:
            psf_out (tensor): PSF. shape [ks, ks]
        """
        # Get input field
        x, y, z = point
        sensor = self.sensor
        sensor_l = sensor.l
        field_res = self.doe.res
        scale = - z / sensor.d.item()
        x_obj, y_obj = x * scale * sensor_l / 2, y * scale * sensor_l / 2
        
        # We have to sample high resolution to meet Nyquist sampling constraint.
        inp_field = point_source_field(point=[x_obj, y_obj, z], phy_size=[sensor_l, sensor_l], res=field_res, wvln=wvln, fieldz=self.surfaces[0].d.item())

        # Calculate PSF on the sensor. 
        psf_full_res = self.forward(inp_field)[0, 0, :, :]   # shape of [H_sensor, W_sensor]

        # Crop the valid patch of the full-resolution PSF
        coord_c_i = int((1 + y) * sensor.res[0] / 2)
        coord_c_j = int((1 - x) * sensor.res[1] / 2)
        psf_full_res = nnF.pad(psf_full_res, [ks//2, ks//2, ks//2, ks//2], mode='constant', value=0)
        psf_out = psf_full_res[coord_c_i:coord_c_i + ks, coord_c_j:coord_c_j + ks]
        
        # Normalize PSF
        psf_out /= psf_out.sum()
        psf_out = torch.flip(psf_out, [0, 1])

        return psf_out


    def psf_rgb(self, point=[0, 0, -DEPTH], ks=101):
        """ Calculate RGB point PSF of DOEThinLens.

        Args:
            point (list, optional): Point source position. Defaults to [0, 0, -DEPTH].
            ks (int, optional): PSF kernel size. Defaults to 101.

        Returns:
            psf (tensor): RGB PSF. Shape of [3, ks, ks]
        """
        psf = []
        for wvln in WAVE_RGB:
            psf_mono = self.psf(point=point, ks=ks, wvln=wvln)
            psf.append(psf_mono)

        psf = torch.stack(psf, dim=0) # shape [3, ks, ks]
        return psf
    
    def psf_board_band(self, point=[0, 0, -DEPTH], ks=101):
        """ Calculate boardband RGB PSF
        """
        psf_r = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(point=point, ks=ks, wvln=wvln)
            psf_r.append(psf * RED_RESPONSE[i])
        psf_r = torch.stack(psf_r, dim=0).sum(dim=0) / sum(RED_RESPONSE)

        psf_g = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(point=point, ks=ks, wvln=wvln)
            psf_g.append(psf * GREEN_RESPONSE[i])
        psf_g = torch.stack(psf_g, dim=0).sum(dim=0) / sum(GREEN_RESPONSE)

        psf_b = []
        for i, wvln in enumerate(WAVE_BOARD_BAND):
            psf = self.psf(point=point, ks=ks, wvln=wvln)
            psf_b.append(psf * BLUE_RESPONSE[i])
        psf_b = torch.stack(psf_b, dim=0).sum(dim=0) / sum(BLUE_RESPONSE)

        psfs = torch.stack([psf_r, psf_g, psf_b], dim=0) # shape [3, ks, ks]

        return psfs

    def psf_map(self, grid=9, ks=101, wvln=0.589, depth=DEPTH):
        """ Generate PSF map for DoeThinlens.

        Args:
            grid (int, optional): Grid size. Defaults to 9.
            ks (int, optional): PSF kernel size. Defaults to 101.
            wvln (float, optional): wvln. Defaults to 0.589.
            depth (float, optional): Depth of the point source. Defaults to DEPTH.

        Returns:
            psf_map (tensor): PSF map. Shape of [grid*ks, grid*ks]
        """
        points = self.point_source_grid(depth=depth, grid=grid, center=True)
        points = points.reshape(-1, 3)

        psfs = []
        for i in range(points.shape[0]):
            point = points[i, ...]
            psf = self.psf(point=point, ks=ks, wvln=wvln)
            psfs.append(psf)

        psf_map = torch.stack(psfs).unsqueeze(1)
        psf_map = make_grid(psf_map, nrow=grid, padding=0)[0, :, :]

        return psf_map
    

    def point_source_grid(self, depth, grid=9, normalized=True, quater=False, center=False):
        """ This function is a re-implementation of Lensgroup.point_source_grid().
        """
        # ==> Use center of each patch
        if grid == 1:
            x, y = torch.tensor([[0.]]), torch.tensor([[0.]])
            assert not quater, 'Quater should be False when grid is 1.'
        else:
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x, y = torch.meshgrid( 
                    torch.linspace(-1 + half_bin_size, 1 - half_bin_size, grid), 
                    torch.linspace(1 - half_bin_size, -1 + half_bin_size, grid),
                    indexing='xy')
            # ==> Use corner
            else:   
                x, y = torch.meshgrid(
                    torch.linspace(-0.98, 0.98, grid), 
                    torch.linspace(0.98, -0.98, grid),
                    indexing='xy')
        
        z = torch.full((grid, grid), depth)
        point_source = torch.stack([x, y, z], dim=-1)
        
        # ==> Use quater of the sensor plane to save memory
        if quater:
            z = torch.full((grid, grid), depth)
            point_source = torch.stack([x, y, z], dim=-1)
            bound_i = grid // 2 if grid % 2 == 0 else grid // 2 + 1
            bound_j = grid // 2
            point_source = point_source[0:bound_i, bound_j:, :]

        if not normalized:
            scale = self.calc_scale_pinhole(depth)
            point_source[..., 0] *= scale * self.sensor_size[0] / 2
            point_source[..., 1] *= scale * self.sensor_size[1] / 2

        return point_source


    def point_source_radial(self, depth, grid=9, normalized=True, center=False):
        """ This function is a re-implementation of Lensgroup.point_source_radial().
        """
        if grid == 1:
            x = torch.tensor([0.])
        else:
            # Select center of bin to calculate PSF
            if center:
                half_bin_size = 1 / 2 / (grid - 1)
                x = torch.linspace(0, 1 - half_bin_size, grid)
            else:   
                x = torch.linspace(0, 0.98, grid)
        
        z = torch.full_like(x, depth)
        point_source = torch.stack([x, x, z], dim=-1)
        return point_source



    def draw_psf(self, depth=DEPTH, ks=101, save_name='./psf_doethinlens.png'):
        """ Draw RGB on-axis PSF.
        """
        psfs = []
        for wvln in WAVE_RGB:
            psf = self.psf(point=[0,0,depth], ks=ks, wvln=wvln)
            psfs.append(psf)

        psfs = torch.stack(psfs, dim=0) # shape [3, ks, ks]
        save_image(psfs.unsqueeze(0), save_name, normalize=True)


    def draw_psf_map(self, grid=8, depth=DEPTH, ks=101, log_scale=False, save_name='./psf_map_doethinlens.png'):
        """ Draw RGB PSF map of the DOE thin lens.
        """
        # Calculate PSF map
        psf_maps = []
        for wvln in WAVE_RGB:
            psf_map = self.psf_map(grid=grid, depth=depth, ks=ks, wvln=wvln)
            psf_maps.append(psf_map)
        psf_map = torch.stack(psf_maps, dim=0) # shape [3, grid*ks, grid*ks]
        
        # Data processing for visualization
        if log_scale:
            psf_map = torch.log(psf_map + 1e-4)
        psf_map = (psf_map - psf_map.min()) / (psf_map.max() - psf_map.min())
        
        save_image(psf_map.unsqueeze(0), save_name, normalize=True)


    def get_optimizer(self, lr):
        return self.doe.get_optimizer(lr=lr)


    # =============================================
    # Utils
    # =============================================   

    def draw_layout(self, save_name='./doethinlens.png'):
        """ Draw lens setup.
        """
        fig, ax = plt.subplots()

        # Plot thin lens
        d = self.thinlens.d.item()
        r = self.thinlens.r
        roc = r * 2 # A manually set roc for plotting
        r_ls = np.arange(-r, r, 0.01)
        d1_ls = d - (np.sqrt(roc**2 - r_ls**2) - np.sqrt(roc**2 - r**2))
        d2_ls = d + (np.sqrt(roc**2 - r_ls**2) - np.sqrt(roc**2 - r**2))
        ax.plot(d1_ls, r_ls, 'black')
        ax.plot(d2_ls, r_ls, 'black')

        # Plot DOE
        d = self.doe.d.item()
        l = self.doe.l
        ax.plot([d, d], [-l/2, l/2], 'black')

        # Plot sensor
        d = self.sensor.d.item()
        l = self.sensor.l
        ax.plot([d, d], [-l/2, l/2], 'black')

        # ax.set_xlim(-1, 100)
        # ax.set_ylim(-100, 100)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.savefig(save_name, dpi=600, bbox_inches='tight')
        plt.close(fig)

