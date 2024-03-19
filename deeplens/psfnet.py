""" Trainer for incoherent PSF network representation.
"""
import torch.nn.functional as nnF
from torchvision.utils import make_grid
import torchvision.transforms.functional as F

from .optics import *
from .render_psf import *
from .network_surrogate import *

DMIN = 200     # [mm]
DMAX = 20000   # [mm]

class PSFNet(Lensgroup):
    def __init__(self, filename, model_name='mlp', kernel_size=11, sensor_res=(512, 512), device='cuda'):
        super(PSFNet, self).__init__(filename=filename, sensor_res=sensor_res, device=device)
        self.device = device

        # Init implicit PSF network
        self.in_features = 4
        self.kernel_size = kernel_size
        self.model_name = model_name
        self.init_net()

        # Training settings
        self.spp = 4096
        self.patch_size = 64
        self.psf_grid = [sensor_res[0] // self.patch_size, sensor_res[1] // self.patch_size]  # 512*512 sensor - 8*8 grid, 320*480 sensor - 5*8 grid, 480*640 sensor - 8*10 grid, 640*960 sensor - 10*15 grid
        
        # There is a minimum focal distance for each lens. 
        # For example, the Canon EF 50mm lens can only focus to 0.45m and further. 
        self.d_max = - DMAX
        self.d_min = - DMIN
        self.foc_d_arr = np.array([-500,   -600,    -700,    -800,    -900,    
                                   -1000,  -1250,   -1500,   -1750,   -2000, 
                                   -2500,  -3000,   -4000,   -5000,   -6000,   
                                   -8000,  -10000,  -12000,  -15000,  -20000])
        self.foc_z_arr = (self.foc_d_arr - self.d_min) / (self.d_max - self.d_min)  # normalize focal distance [0, 1]


    # ==================================================
    # Training functions
    # ==================================================
    def init_net(self):
        """ Initialize a network. 
        
            Basically there are three kinds of network architectures: (1) MLP, (2) MLP + Conv, (3) Siren. 
            
            We can also choose to represent (1) single-point PSF, (2) PSF map.

            Network input: (x, y, z, foc_dist), shape [N, 4].
            Network output: psf kernel (ks * ks) or psf map (psf_grid * ks * psf_grid * ks).
        """
        ks = self.kernel_size
        model_name = self.model_name

        if model_name == 'mlp':
            self.psfnet = MLP(in_features=4, out_features=ks**2, hidden_features=256, hidden_layers=8)
        
        elif model_name == 'mlpconv':
            self.psfnet = MLPConv(in_features=2)

        elif model_name == 'siren':
            raise NotImplementedError
        
        else:
            raise Exception('Unsupported PSF network architecture.')
        
        self.psfnet.apply(initialize_weights)
        self.psfnet.to(self.device)

    
    def load_net(self, net_path):
        """ Load pretrained network.
        """
        self.psfnet.load_state_dict(torch.load(net_path))


    def train_psfnet(self, iters=10000, bs=128, lr=1e-4, spp=2048, evaluate_every=1000, result_dir='./results/temp'):
        """ Fit the PSF representation network. Training data is generated on the fly.
        """
        # Init network and prepare for training
        ks = self.kernel_size
        psfnet = self.psfnet
        psf_cri = nn.MSELoss()
        optim = torch.optim.AdamW(psfnet.parameters(), lr)
        sche = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=int(iters), eta_min=0)

        # Train the network
        for i in tqdm(range(iters + 1)):

            # Training data
            if self.model_name == 'mlp':
                inp, psf = self.get_training_data(bs=bs, spp=spp)
            else:
                psf_grid = self.psf_grid
                inp, psf = self.get_training_psf_map(bs=bs, psf_grid=psf_grid)
            
            inp, psf = inp.to(self.device), psf.to(self.device)
            
            # Forward-backward optimization
            psf_pred = psfnet(inp)

            optim.zero_grad()
            loss = psf_cri(psf_pred, psf) 
            loss.backward()
            optim.step()
            sche.step()

            # Evaluate
            if (i+1) % evaluate_every == 0:
                if self.model_name == 'mlp':
                    fig, axs = plt.subplots(5, 2)
                    for j in range(5):
                        psf0 = psf[j, :].detach().clone().cpu()
                        psf0 = psf0.view(ks, ks)
                        axs[j, 0].imshow(psf0)

                        psf1 = psf_pred[j, :].detach().clone().cpu()
                        psf1 = psf1.view(ks, ks)
                        axs[j, 1].imshow(psf1)
                    
                    fig.suptitle(f"GT/Pred PSFs at iter {i+1}")
                    plt.savefig(f'{result_dir}/iter{i+1}.png', dpi=300)
                    plt.close()
                else:
                    save_image(psf, f'{result_dir}/iter{i+1}_psf_gt.png')
                    save_image(psf_pred, f'{result_dir}/iter{i+1}_psf_pred.png')

                torch.save(psfnet.state_dict(), f'{result_dir}/iter{i+1}_PSFNet_{self.model_name}.pkl')

        torch.save(psfnet.state_dict(), f'{result_dir}/PSFNet_{self.model_name}.pkl')


    def get_training_data(self, bs=256, spp=4096):
        """ Generate training data for a focus distance (f_d) and a group of spatial points (x, y, z).

            Input (x, y, z, foc_dist) range from [-1, 1] * [-1, 1] * [0, 1]
            Output (psf) normalized to 1D tensor.
        
        Args:
            bs (int): batch size
            spp (int): ray samples per pixel
        """
        # Sample only one f_d and refocus the lens
        foc_z = np.random.choice(self.foc_z_arr)
        foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min
        self.refocus(depth = foc_dist)
        
        # Sample (x, y), uniform distribution
        x = (torch.rand(bs) - 0.5) * 2
        y = (torch.rand(bs) - 0.5) * 2

        # Sample (z), Gaussian distribution (3-sigma interval)
        z_gauss = torch.clamp(torch.randn(bs), min=-3, max=3)
        z = torch.zeros_like(z_gauss)
        z[z_gauss>0] = (1 - foc_z) * z_gauss[z_gauss>0]/3 + foc_z   # sample [foc_z, 1], then scale to [foc_d, dmax]
        z[z_gauss<0] = foc_z * z_gauss[z_gauss<0]/3 + foc_z         # sample [0, foc_z], then scale to [dmin, foc_d]

        # Network input, shape of [N, 4]
        foc_z_tensor = torch.full_like(x, foc_z)
        inp = torch.stack((x, y, z, foc_z_tensor), dim=-1)

        # Ray tracing to compute PSFs, shape of [N, ks, ks]
        depth = self.z2depth(z)
        points = torch.stack((x, y, depth), dim=-1)
        psf = self.psf(points=points, ks=self.kernel_size, spp=spp)
        psf = psf.view(bs, -1)

        return inp, psf
    

    def get_training_psf_map(self, bs=8, psf_grid=(11,11), psf_map_size=(128, 128)):
        """ Generate PSF map for training. This training data is used for MLP_Conv network architecture.

            Reference: "Differentiable Compound Optics and Processing Pipeline Optimization for End-To-end Camera Design."

        Args:
            bs (int): batch size
            psf_grid (tuple): PSF grid size
            psf_map_size (tuple): PSF map size

        Returns:
            inp (tensor): [B, 2] tensor, [z, foc_z]
            psf_map_batch (tensor): [B, 3, psf_map_size, psf_map_size] tensor
        """
        # Refocus
        foc_z = np.random.choice(self.foc_z_arr)
        foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min

        # Different depths
        z_gauss = torch.clamp(torch.randn(bs), min=-3, max=3)
        z = torch.zeros_like(z_gauss)
        z[z_gauss>0] = (1 - foc_z) * z_gauss[z_gauss>0]/3 + foc_z
        z[z_gauss<0] = foc_z * z_gauss[z_gauss<0]/3 + foc_z
        depth = self.z2depth(z)

        # 2D Input (foc_z, z)
        foc_z_tensor = torch.full_like(z, foc_z)
        inp = torch.stack((z, foc_z_tensor), dim=-1)    # [B, 2]

        # Calculate PSF map
        psf_map_batch = []
        for depth_i in depth:
            psf_map = self.calc_psf_map(foc_dist, depth_i, psf_grid=psf_grid)
            psf_map_batch.append(psf_map)
        psf_map_batch = torch.stack(psf_map_batch, dim=0)   # [B, 3, psf_grid*ks, psf_grid*ks]
        
        # Resize to meet the network requirement
        psf_map_batch = F.resize(psf_map_batch, psf_map_size)   # [B, 3, size, size]

        return inp, psf_map_batch


    def calc_psf_map(self, foc_dist, depth, psf_grid=(11, 11)):
        """ Calculate PSF grid by ray tracing. 

            This function is similiar for self.psf() function.
        """
        ks = self.kernel_size
        spp = self.spp

        # Focus to given distance
        self.refocus(depth = foc_dist)

        # Sample grid points
        x, y = torch.meshgrid(
            torch.linspace(-1 + 1/(2*psf_grid[1]), 1 - 1/(2*psf_grid[1]), psf_grid[1]),
            torch.linspace(1 - 1/(2*psf_grid[0]), -1 + 1/(2*psf_grid[0]), psf_grid[0]),
            indexing='xy'
        )
        x, y = x.reshape(-1), y.reshape(-1)
        depth = torch.full_like(x, depth)
        o = torch.stack((x, y, depth), dim=-1)
        
        # Calculate PSf by ray-tracing
        psf = self.psf(points=o, kernel_size=ks, spp=spp, center=True) # [psf_grid^2, ks, ks]

        # Convert to tensor and save image
        psf_map = make_grid(psf.unsqueeze(1), nrow = psf_grid[1], padding=0)    # [3, psf_grid*ks, psf_grid*ks]
        
        return psf_map
    

    # ==================================================
    # Test
    # ==================================================
    @torch.no_grad()
    def evaluate_psf(self, result_dir='./'):
        """ Qualitaticely compare GT, pred, and thinlens PSF. 
        
            Lens focuses to 1.5m, evaluate PSF at 1.2m, 1.5m, 2m.
        """
        # Evalution settings
        ks = self.kernel_size
        ps = self.sensor_size[0] / self.sensor_res[0]
        psfnet = self.psfnet
        psfnet.eval()

        x = torch.Tensor([0, 0.6, 0.98])
        y = torch.Tensor([0, 0.6, 0.98])
        test_foc_dists = torch.Tensor([-1500])
        test_dists = torch.Tensor([-1200, -1500, -2000])
        test_foc_z = self.depth2z(test_foc_dists)
        test_z = self.depth2z(test_dists)

        # Gaussian PSF parameters
        x_gaussi, y_gaussi = torch.meshgrid(
            torch.linspace(-ks/2+1/2, ks/2-1/2, ks),
            torch.linspace(-ks/2+1/2, ks/2-1/2, ks),
            indexing='xy'
        )

        # Evaluation
        for foc_z in test_foc_z:
            foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min
            self.refocus(depth = foc_dist)

            for z in test_z:
                # GT PSF by ray tracing
                depth = z * (self.d_max - self.d_min) + self.d_min
                depth_tensor = torch.full_like(x, depth)
                points = torch.stack((x, y, depth_tensor), dim=-1)
                psf_gt = self.psf(points=points, ks=ks, center=True)
                self.vis_psf_map(psf_gt, filename=f'{result_dir}/foc{-foc_dist}_depth{-depth}_gt.png')

                # Network prediction
                z_tensor = torch.full_like(x, z)
                foc_z_tensor = torch.full_like(x, foc_z)
                inp = torch.stack((x, y, z_tensor, foc_z_tensor), dim=-1).to(self.device)
                psf_pred = psfnet(inp).view(-1,ks,ks)
                self.vis_psf_map(psf_pred, filename=f'{result_dir}/foc{-foc_dist}_depth{-depth}_pred.png')

                # Thin lens Gaussian model
                # "Focus on defocus: bridging the synthetic to real domain gap for depth estimation" Eq.(1)
                coc = torch.abs(depth - foc_dist) * self.foclen**2 / (depth * self.fnum *(foc_dist - self.foclen))
                coc_pixel = torch.clamp(coc / ps, min=0.1)
                coc_pixel_radius = coc_pixel / 2
                psf_thin = torch.exp(- (x_gaussi**2 + y_gaussi**2) / (2 * coc_pixel_radius**2)) # We ignore constant term because PSF will be normalized later
                psf_mask = (x_gaussi**2 + y_gaussi**2 < coc_pixel_radius**2)
                psf_thin = psf_thin * psf_mask # Un-clipped Gaussian PSF
                psf_thin = psf_thin / psf_thin.sum((-1, -2)).unsqueeze(-1).repeat(3,1,1)
                self.vis_psf_map(psf_thin, filename=f'{result_dir}/foc{-foc_dist}_depth{-depth}_thin.png')


    @torch.no_grad()
    def evaluate_psf_score(self, vis=False, evaluate_model=None, result_dir='./'):
        """ Quantitatively evaluate GT, Pred, Thinlens PSF for all settings.

            Focus to all possible distances, evaluate at all possible spatial positions.
        """
        psf_grid = self.psf_grid    # we evaluate PSF grid
        ks = self.kernel_size
        spp = self.spp
        psfnet = self.psfnet
        psfnet.eval()
        evaluate_model = self.evaluate_model if evaluate_model is None else evaluate_model

        # Evaluate at all focus distances and spatial positions (reprented by PSF grid).
        test_foc_z = self.foc_z_arr
        test_z = np.linspace(0, 1, 40, endpoint=True)
        x, y = torch.meshgrid(
            torch.linspace(-1 + 1/(2*psf_grid[1]), 1 - 1/(2*psf_grid[1]), psf_grid[1]),
            torch.linspace(1 - 1/(2*psf_grid[0]), -1 + 1/(2*psf_grid[0]), psf_grid[0]),
            indexing='xy'
        )
        x, y = x.reshape(-1), y.reshape(-1)

        # Evaluate and compute scores.
        l2_error = []
        l1_error = []
        for foc_z in tqdm(test_foc_z):
            foc_dist = foc_z * (self.d_max - self.d_min) + self.d_min
            self.refocus(depth = foc_dist)

            for z in test_z:
                depth = z * (self.d_max - self.d_min) + self.d_min
                
                # ==> Ray optics (PSF GT)
                depth_tensor = torch.full_like(x, depth)
                o = torch.stack((x, y, depth_tensor), dim=-1)
                psf_gt = self.psf(o=o, kernel_size=ks, spp=spp, center=True)    # [psf_grid^2, ks, ks]
                
                if evaluate_model == 'mlp':
                    # MLP network prediction
                    z_tensor = torch.full_like(x, z)
                    foc_z_tensor = torch.full_like(x, foc_z)
                    inp = torch.stack((x, y, z_tensor, foc_z_tensor), dim=-1).to(self.device)
                    psf_pred = psfnet(inp).view(-1,ks,ks)  # [psf_grid^2, ks, ks]

                    # Calculate score
                    l2_error.append(torch.sum((psf_gt - psf_pred)**2) / psf_gt.numel())
                    l1_error.append(torch.sum((psf_gt - psf_pred).abs()) / psf_gt.numel())

                    # Visualization
                    if vis:
                        # [psf_grid^2, 3, ks, ks] to [3, psf_grid*ks, psf_grid*ks]
                        psf_gt = make_grid(psf_gt.unsqueeze(1), nrow = psf_grid[1], padding = 0)    
                        psf_pred = make_grid(psf_pred.unsqueeze(1), nrow = psf_grid[1], padding = 0)
                        
                        scale = 1 / max(psf_gt.max(), psf_pred.max())
                        save_image(psf_gt * scale, f'{result_dir}/psf_foc{-foc_dist}_depth{-depth}_gt.png')
                        save_image(psf_pred * scale, f'{result_dir}/psf_foc{-foc_dist}_depth{-depth}_pred.png')
                
                else:
                    raise Exception('Unimplemented')

        avg_l2_error = sum(l2_error) / len(l2_error)
        avg_l1_error = sum(l1_error) / len(l1_error)
        print(f'avg l1 error: {avg_l1_error}, avg l2 error: {avg_l2_error}.')


    # ==================================================
    # Use network after image simulation
    # ==================================================
    def pred(self, inp):
        """ Predict PSFs using the PSF network.

        Args:
            inp (tensor): [N, 4] tensor, [x, y, z, foc_dist]

        Returns:
            psf (tensor): [N, ks, ks] or [H, W, ks, ks] tensor
        """
        # Network prediction, shape of [N, ks^2]
        psf = self.psfnet(inp)
        
        # Reshape, shape of [N, ks, ks] or [H, W, ks, ks]
        psf = psf.reshape(*psf.shape[:-1], self.kernel_size, self.kernel_size)
        
        return psf
        

    @torch.no_grad()
    def render(self, img, depth, foc_dist):
        """ Render image with aif image and depth map. Receive [N, C, H, W] image.

        Args:
            img (tensor): [N, C, H, W]
            depth (tensor): [N, H, W], depth map, unit in mm, range from [-20000, -200]
            foc_dist (tensor): [N], unit in mm, range from [-20000, -200]
            high_res (bool): whether to use high resolution rendering

        Returns:
            render (tensor): [N, C, H, W]
        """
        if len(img.shape) == 3:
            H, W = depth.shape
            z = self.depth2z(depth)
            x, y = torch.meshgrid(
                torch.linspace(-1,1,W),
                torch.linspace(1,-1,H),
                indexing='xy'
            )
            x, y = x.to(self.device), y.to(self.device)
            foc_dist = torch.full_like(depth, foc_dist)
            foc_z = self.depth2z(foc_dist)
            
            o = torch.stack((x, y, z, foc_z), -1)
            psf = self.pred(o)

            render = local_psf_render(img, psf, self.kernel_size)
            return render
        
        elif len(img.shape) == 4:
            N, C, H, W = img.shape
            z = self.depth2z(depth).squeeze(1)
            x, y = torch.meshgrid(
                torch.linspace(-1,1,W),
                torch.linspace(1,-1,H),
                indexing='xy'
            )
            x, y = x.unsqueeze(0).repeat(N,1,1), y.unsqueeze(0).repeat(N,1,1)
            x, y = x.to(img.device), y.to(img.device)
            foc_dist = foc_dist.unsqueeze(-1).unsqueeze(-1).repeat(1,H,W)
            foc_z = self.depth2z(foc_dist)

            o = torch.stack((x, y, z, foc_z), -1).float()
            psf = self.pred(o)

            render = local_psf_render(img, psf, self.kernel_size)
            return render


    # ==================================================
    # Utils
    # ==================================================
    def depth2z(self, depth):
        z = (depth - self.d_min) / (self.d_max - self.d_min)
        z = torch.clamp(z, min=0, max=1)
        return z

    def z2depth(self, z):
        depth = z * (self.d_max - self.d_min) + self.d_min
        return depth
    
    def vis_psf_map(self, psf, filename=None):
        """ Visualize a [N, N, k, k] or [N, N, k^2] or [N, k, k] PSF kernel.
        """
        if len(psf.shape) == 4:
            N, _, _, _ = psf.shape
            fig, axs = plt.subplots(N, N)
            for i in range(N):
                for j in range(N):
                    psf0 = psf[i, j, :, :].detach().clone().cpu()
                    axs[i, j].imshow(psf0, vmin=0.0, vmax=0.1)

        elif len(psf.shape) == 3:
            N, _, _ = psf.shape
            fig, axs = plt.subplots(1, N)
            for i in range(N):
                psf0 = psf[i, :, :].detach().clone().cpu()
                axs[i].imshow(psf0, vmin=0.0, vmax=0.1)
                axs[i].axis('off')

        # Return fig
        if filename is not None:
            plt.savefig(filename, dpi=300)
        else:
            plt.show()

        plt.cla()


   

# ==================================================================
# Thin lens model (baseline)
# ==================================================================
class ThinLens(DeepObj):
    def __init__(self, foc_len, fnum, kernel_size, sensor_size, sensor_res, device='cpu'):
        super(ThinLens, self).__init__()

        self.d_max = DMAX
        self.d_min = DMIN
        self.kernel_size = kernel_size
        self.foc_len = foc_len
        self.fnum = fnum
        self.sensor_size = sensor_size
        self.sensor_res = sensor_res
        self.ps = self.sensor_size[0] / self.sensor_res[0]


    def coc(self, depth, foc_dist):
        if (depth < 0).any():
            depth = - depth
            foc_dist = - foc_dist

        depth = torch.clamp(depth, self.d_min, self.d_max)
        coc = self.foc_len / self.fnum * torch.abs(depth - foc_dist) / depth * self.foc_len / (foc_dist - self.foc_len)
        coc_pixel = torch.clamp(coc / self.ps, min=0.1)
        return coc_pixel


    def render(self, img, depth, foc_dist):
        """ Render image with aif image and Gaussian PSFs.

        Args:
            img: [N, C, H, W]
            depth: [N, 1, H, W]
            foc_dist: [N]

        Raises:
            Exception: _description_

        Returns:
            _type_: _description_
        """
        ks = self.kernel_size
        device = img.device
        
        if len(img.shape) == 3:
            H, W = depth.shape
            z = self.depth2z(depth)
            x, y = torch.meshgrid(
                torch.linspace(-1,1,W),
                torch.linspace(1,-1,H),
                indexing='xy'
            )
            x, y = x.to(self.device), y.to(self.device)
            foc_dist = torch.full_like(depth, foc_dist)
            foc_z = self.depth2z(foc_dist)
            o = torch.stack((x, y, z, foc_z), -1)    
            
            psf = self.pred(o)
            render = local_psf_render(img, psf, self.kernel_size)
            
            return render
        
        elif len(img.shape) == 4:
            N, C, H, W = img.shape
            foc_dist = foc_dist.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, H, W) # [N] to [N, 1, H, W]
            psf = torch.zeros((N, H, W, ks, ks), device=device)
            x, y = torch.meshgrid(
                torch.linspace(-ks/2+1/2, ks/2-1/2, ks),
                torch.linspace(ks/2-1/2, -ks/2+1/2, ks),
                indexing='xy'
            )
            x, y = x.to(device), y.to(device)

            coc_pixel = self.coc(depth, foc_dist)
            # Shape expands to [N, H, W, ks, ks]
            coc_pixel = coc_pixel.squeeze(1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, ks, ks)
            coc_pixel_radius = coc_pixel / 2
            psf = torch.exp(- (x**2 + y**2) / 2 / coc_pixel_radius**2) / (2 * np.pi * coc_pixel_radius**2)
            psf_mask = (x**2 + y**2 < coc_pixel_radius**2)
            psf = psf * psf_mask
            psf = psf / psf.sum((-1, -2)).unsqueeze(-1).unsqueeze(-1)

            render = local_psf_render(img, psf, self.kernel_size)
            return render