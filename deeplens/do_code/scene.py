""" Mitsuba2 scene functions. Not tested yet.
"""
from ..optics.basics import *
from ..geolens import *
from .shapes import *

class Scene(DeepObj):
    def __init__(self, cameras, screen, lensgroup=None, device=torch.device('cpu')):
        super().__init__()
        self.cameras = cameras
        self.screen = screen
        self.lensgroup = lensgroup
        self.device = device

        self.camera_count = len(self.cameras)

        # rendering options (for now)
        self.pps = 1
        self.wavelength = 500 # [nm]

    def render(self, i=None, with_element=True, mask=None, to_numpy=False):
        im = self._simulate(i, with_element, mask, SimulationMode.render)
        if to_numpy:
            im = [x.cpu().detach().numpy() for x in im]
        return im

    def trace(self, i=None, with_element=True, mask=None, to_numpy=False):
        results = self._simulate(i, with_element, mask, SimulationMode.trace)
        p = [x[0].cpu().detach().numpy() if to_numpy else x[0] for x in results]
        valid = [x[1].cpu().detach().numpy() if to_numpy else x[1] for x in results]
        mask_g = [x[2].cpu().detach().numpy() if to_numpy else x[2] for x in results]
        return p, valid, mask_g

    def plot_setup(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # generate color-ring
        if self.camera_count <= 6:
            colors = ['b','r','g','c','m','y']
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            colors = colors * np.ceil(self.camera_count/len(colors)).astype(int)
            colors = colors[:self.camera_count]

        # draw screen
        seq = [0,2,1] # draw scene in [x,z,y] order
        self.screen.draw_points(ax, 'k', seq)

        # draw metrology part
        if self.lensgroup is not None:
            self.lensgroup.draw_points(ax, 'y.', seq)

        # draw cameras
        for i, camera in enumerate(self.cameras):
            camera.draw_points(ax, colors[i], seq)

        # pretty plot
        labels = 'xyz'
        scales = np.array([1.5,1.5,1])
        ax.set_xlabel(labels[seq[0]] + ' [mm]')
        ax.set_ylabel(labels[seq[1]] + ' [mm]')
        ax.set_zlabel(labels[seq[2]] + ' [mm]')
        ax.set_title('setup')
        set_axes_equal(ax, np.array([scales[seq[i]] for i in range(3)]))
        plt.show()

    def visualize(self):
        pass
    
    def to(self, device=torch.device('cpu')):
        super().to(device)
        
        # set device name (TODO: make it more elegant)
        self.device = device
        self.lensgroup.device = device
        self.screen.device = device
        for i in range(self.camera_count):
            self.cameras[i].device = device
        for i in range(len(self.lensgroup.surfaces)):
            self.lensgroup.surfaces[i].device = device

    def _simulate(self, i=None, with_element=True, mask=None, smode=SimulationMode.render):
        def simulate(i):
            if mask is None: # default: full sensor rendering
                ray = self.cameras[i].sample_ray()
                
                # interaction with metrology part
                if with_element and self.lensgroup is not None:
                    ray, valid_ray, mask_g = self.lensgroup.trace(ray)
                else:
                    mask_g = torch.ones(ray.o.shape[0:2], device=self.device)
                    valid_ray = mask_g.clone().bool()
                    
                # interaction with screen
                # We permute the axes to be compatiable with the image viewer (and the data).
                p, uv, valid_screen = self.screen.intersect(ray)
                valid = valid_screen & valid_ray

                if smode is SimulationMode.render:
                    del p
                    return self.screen.shading(uv, valid).permute(1,0)
                    # return self.screen.shading(uv, valid_screen).permute(1,0)

                elif smode is SimulationMode.trace:
                    del uv
                    return p.permute(1,0,2), valid.permute(1,0), mask_g.permute(1,0)
                    # TODO: if the rays are not valid, set the intersection point to be zeros
            
            else: # get corresponding indices from the mask
                mask_ = mask[i].permute(1,0) # we transpose mask to align with our code
                ix, iy = torch.where(mask_)
                p2 = self.cameras[i].generate_position_sample(mask_)
                ray = self.cameras[i].sample_ray(p2)
                
                # interaction with metrology part
                if with_element and self.lensgroup is not None:
                    ray, valid_ray, mask_g_ = self.lensgroup.trace(ray)
                else:
                    mask_g_ = torch.ones(ray.o.shape[0:2])
                    valid_ray = mask_g_.clone().bool()
                
                # # update via masking
                # ray.o = ray.o[valid_ray, ...]
                # ray.d = ray.d[valid_ray, ...]
                # ix = ix[valid_ray]
                # iy = iy[valid_ray]

                # interaction with screen
                # We permute the axes to be compatiable with the image viewer (and the data).
                p_, uv, valid_screen = self.screen.intersect(ray)
                
                # # update via masking
                # p_ = p_[valid_screen, ...]
                # ix = ix[valid_screen]
                # iy = iy[valid_screen]
                
                if smode is SimulationMode.render:
                    del p_
                    # TODO: unwrap the mask (1D->2D)
                    raise NotImplementedError()

                elif smode is SimulationMode.trace:
                    del uv
                    p = torch.zeros(*self.cameras[i].filmsize, 3, device=self.device)
                    p[ix, iy, ...] = p_
                    valid = torch.zeros(*self.cameras[i].filmsize, device=self.device).bool()
                    valid[ix, iy] = mask_[ix, iy]
                    mask_g = torch.zeros(*self.cameras[i].filmsize, device=self.device)
                    mask_g[ix, iy] = mask_g_
                    return p.permute(1,0,2), valid.permute(1,0), mask_g.permute(1,0)

        return [simulate(j) for j in range(self.camera_count)] if i is None else simulate(i)

# ----------------------------------------------------------------------------------------

def generate_test_scene():
    R = np.eye(3)
    ts = [np.array([0, 0, -300]), np.array([0, 0, -240])]
    cameras = [generate_test_camera(R, t) for t in ts]
    screen = generate_test_screen()
    lensgroup = generate_test_lensgroup()
    return Scene(cameras, screen, lensgroup)

# ----------------------------------------------------------------------------------------


if __name__ == "__main__":
    init()

    scene = generate_test_scene()
    scene.plot_setup()

    # render images
    imgs = scene.render()
    fig, ax = plt.subplots(1,2)
    for i, img in enumerate(imgs):
        ax[i].imshow(img, cmap='gray')
        ax[i].set_title('camera ' + str(i+1))
    plt.show()
