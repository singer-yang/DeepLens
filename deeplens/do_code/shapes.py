""" Old functions, not tested yet.
"""
from ..optics.basics import *

class Endpoint(DeepObj):
    def __init__(self, transformation, device=torch.device('cpu')):
        super(Endpoint, self).__init__()
        self.to_world  = transformation
        self.to_object = transformation.inverse()
        self.to_world.to(device)
        self.to_object.to(device)
        self.device = device

    def intersect(self, ray):
        raise NotImplementedError()

    def sample_ray(self, position_sample=None):
        raise NotImplementedError()

    def draw_points(self, ax, options, seq=range(3)):
        raise NotImplementedError()


class Screen(Endpoint):
    """
    Local frame centers at [-w, w]/2 x [-h, h]/2
    """
    def __init__(self, transformation, size, pixelsize, texture, device=torch.device('cpu')):
        self.size = torch.Tensor(np.float32(size))  # screen dimension [mm]
        self.halfsize  = self.size/2                # screen half-dimension [mm]
        self.pixelsize = torch.Tensor([pixelsize])  # screen pixel size [mm]
        self.texture   = torch.Tensor(texture)      # screen image
        self.texturesize = torch.Tensor(np.array(texture.shape[0:2])) # screen image dimension [pixel]
        self.texturesize_np = self.texturesize.cpu().detach().numpy() # screen image dimension [pixel]
        self.texture_shift = torch.zeros(2)         # screen image shift [mm]
        Endpoint.__init__(self, transformation, device)
        self.to(device)
        
    def intersect(self, ray):
        ray_in = self.to_object.transform_ray(ray)
        t = - ray_in.o[..., 2] / ray_in.d[..., 2] # well-posed for dz (TODO: potential NaN grad)
        local = ray_in(t)

        # Is intersection within ray segment and rectangle?
        valid = (
            (t >= MINT) &
            (t <= MAXT) &
            (torch.abs(local[..., 0] - self.texture_shift[0]) <= self.halfsize[0]) &
            (torch.abs(local[..., 1] - self.texture_shift[1]) <= self.halfsize[1])
        )

        # uv map
        uv = (local[..., 0:2] + self.halfsize - self.texture_shift) / self.size

        # force uv to be valid in [0,1]^2 (just a sanity check: uv should be in [0,1]^2)
        uv = torch.clamp(uv, min=0.0, max=1.0)

        return local, uv, valid

    def shading(self, uv, valid, bmode=BoundaryMode.replicate, lmode=InterpolationMode.linear):
        # p = uv * (self.texturesize[None, None, ...]-1)
        p = uv * (self.texturesize-1)
        p_floor = torch.floor(p).long()

        def tex(x, y): # texture indexing function
            if bmode is BoundaryMode.zero:
                raise NotImplementedError()
            elif bmode is BoundaryMode.replicate:
                x = torch.clamp(x, min=0, max=self.texturesize_np[0]-1)
                y = torch.clamp(y, min=0, max=self.texturesize_np[1]-1)
            elif bmode is BoundaryMode.symmetric:
                raise NotImplementedError()
            elif bmode is BoundaryMode.periodic:
                raise NotImplementedError()
            img = self.texture[x.flatten(), y.flatten()]
            return img.reshape(x.shape)

        if lmode is InterpolationMode.nearest:
            val = tex(p_floor[...,0], p_floor[...,1])
        elif lmode is InterpolationMode.linear:
            x0, y0 = p_floor[...,0], p_floor[...,1]
            s00 = tex(  x0,   y0)
            s01 = tex(  x0, 1+y0)
            s10 = tex(1+x0,   y0)
            s11 = tex(1+x0, 1+y0)
            w1 = p - p_floor
            w0 = 1. - w1
            val = (
                w0[...,0] * (w0[...,1] * s00 + w1[...,1] * s01) + 
                w1[...,0] * (w0[...,1] * s10 + w1[...,1] * s11)
            )
        
        # val = val * valid
        # val[torch.isnan(val)] = 0.0

        # TODO: should be added;
        # but might cause RuntimeError: Function 'MulBackward0' returned nan values in its 0th output.
        val[~valid] = 0.0
        return val

    def draw_points(self, ax, options, seq=range(3)):
        coeffs = np.array([
            [ 1, 1, 1],
            [-1, 1, 1],
            [-1,-1, 1],
            [ 1,-1, 1],
            [ 1, 1, 1]
        ])
        points_local = torch.Tensor(coeffs * np.append(self.halfsize.cpu().detach().numpy(), 0)).to(self.device)
        points_world = self.to_world.transform_point(points_local).T.cpu().detach().numpy()
        ax.plot(points_world[seq[0]], points_world[seq[1]], points_world[seq[2]], options)


class Camera(Endpoint):
    def __init__(self, transformation,
        filmsize, f=np.zeros(2), c=np.zeros(2), k=np.zeros(3), p=np.zeros(2), device=torch.device('cpu')):
        self.filmsize = filmsize
        self.f = torch.Tensor(np.float32(f))  # focal lengths [pixel]
        self.c = torch.Tensor(np.float32(c))  # centers [pixel]
        Endpoint.__init__(self, transformation, device)

        # un-initialized for now:
        self.crop_offset = torch.zeros(2, device=device) # [pixel]

        # no use for now:
        self.k = torch.Tensor(np.float32(k))
        if len(self.k) < 3: self.k = np.append(self.k, 0)
        self.p = torch.Tensor(np.float32(p))

        # configurations
        self.NEWTONS_MAXITER = 5
        self.NEWTONS_TOLERANCE = 50e-6 # in [mm], i.e. 50 [nm] here
        self.use_approximation = False
        self.to(device)

    def generate_position_sample(self, mask=None):
        """Generate position samples (not uniform sampler) from a 2D mask.
        """
        dim = self.filmsize
        X, Y = torch.meshgrid(
            0.5 + dim[0] * torch.linspace(0, 1, 1+dim[0], device=self.device)[:-1],
            0.5 + dim[1] * torch.linspace(0, 1, 1+dim[1], device=self.device)[:-1],
        )
        if mask is not None:
            X, Y = X[mask], Y[mask]
        # X.shape could be 1D (masked) or 2D (no masked)
        return torch.stack((X, Y), axis=len(X.shape))

    def sample_ray(self, position_sample=None, is_sampler=False, wavelength_sample=0.45): #0.277778
        """Sample ray(s) from sensor pixels.
        wavelength_sample: (default: 500 [nm])
        """
        spectrum = Spectrum() # TODO
        wavelength = spectrum.sample_wavelength(
            torch.Tensor(np.asarray(wavelength_sample)).to(self.device)
        ) # 562 nm
        
        if position_sample is None: # default: full-sensor deterministic rendering
            dim = self.filmsize
            position_sample = self.generate_position_sample()
            is_sampler = False
        else:
            dim = position_sample.shape[:-1]

        if is_sampler:
            uv = position_sample * np.float32(dim)
        else:
            uv = position_sample

        # in local
        xy = self._uv2xy(uv)
        dz = torch.ones((*dim, 1), device=self.device)
        d = torch.cat((xy, dz), axis=-1)
        d = normalize(d)
        o = torch.zeros((*dim, 3), device=self.device)

        # in world
        o = self.to_world.transform_point(o)
        d = self.to_world.transform_vector(d)
        ray = Ray(o, d, wavelength, self.device)
        return ray

    def draw_points(self, ax, options, seq=range(3)):
        origin = np.zeros(3)
        scales = np.append(self.filmsize/100, 20)
        coeffs = np.array([
            [ 1, 1, 1],
            [-1, 1, 1],
            [-1,-1, 1],
            [ 1,-1, 1],
            [ 1, 1, 1]
        ])
        sensor_corners = torch.Tensor(coeffs * scales).to(self.device)
        ps = self.to_world.transform_point(sensor_corners).T.cpu().detach().numpy()
        ax.plot(ps[seq[0]], ps[seq[1]], ps[seq[2]], options)

        for i in range(4):
            coeff = coeffs[i] * scales
            line = torch.Tensor(np.array([
                origin,
                coeff
            ])).to(self.device)
            ps = self.to_world.transform_point(line).T.cpu().detach().numpy()
            ax.plot(ps[seq[0]], ps[seq[1]], ps[seq[2]], options)
    
    def _distortion(self, xy):
        x, y = xy[...,0], xy[...,1]
        x2, y2 = x**2, y**2
        r2 = x2 + y2
        tmp = r2 * (self.k[0] + r2 * (self.k[1] + r2 * self.k[2]))
        return torch.stack((
            x * (tmp + 2*self.p[0] * y) + self.p[1] * (r2 + 2*x2),
            y * (tmp + 2*self.p[1] * x) + self.p[0] * (r2 + 2*y2)
        ), axis=-1)

    def _distortion_derivatives(self, xy):
        x, y = xy[...,0], xy[...,1]
        x2, y2 = x**2, y**2
        r2 = x2 + y2
        tmp1 = r2 * (self.k[0] + r2 * (self.k[1] + r2 * self.k[2]))
        tmp2 = self.k[0] + r2 * (2*self.k[1] + r2 * 3*self.k[2])
        return torch.stack((
            tmp1 + 2*self.p[0] * y + 2*x2*tmp2 + 6*self.p[1]*x,
            tmp1 + 2*self.p[1] * x + 2*y2*tmp2 + 6*self.p[0]*y,
        ), axis=-1)

    def _uv2xy(self, uv):
        xy_distorted = (uv + self.crop_offset - self.c) / self.f
        xy = xy_distorted

        # TODO: following code does not work correctly.
        # xy_residual = torch.Tensor([float('inf')]).repeat(uv.shape)
        # iter = 0
        # while ((torch.sum(torch.abs(xy_residual),axis=-1) > self.NEWTONS_TOLERANCE).any()
        #         and (iter < self.NEWTONS_MAXITER)):
        #     iter += 1
        #     gxy = self._distortion(xy)

        #     # compute residual
        #     xy_residual = xy + gxy - xy_distorted

        #     # Newton's update
        #     if self.use_approximation: # approximate version
        #         xy -= xy_residual # Note: could be faster if not computing residuals
        #     else: # accurate version
        #         gxy_prime = self._distortion_derivatives(xy)
        #         xy -= xy_residual / (1 + gxy_prime)
        return xy

# ----------------------------------------------------------------------------------------

def generate_test_camera(R=np.eye(3), t=np.array([0, 0, -400])):
    to_world = Transformation(R, t)

    filmsize = np.array([360, 480])
    f = np.array([1000,1010]) # [pixel]
    c = np.array([150,160])   # [pixel]
    return Camera(to_world, filmsize, f, c)

def generate_test_screen():
    R = np.eye(3)
    t = np.zeros(3)
    to_world = Transformation(R, t)

    screensize = np.array([81., 80.]) # [mm]
    pixelsize = 0.115 # [mm]

    # read texture image
    im = imread('./images/llama.jpg')
    im = np.mean(im, axis=-1) # for now we use grayscale
    im = im[200:500,200:600]
    # plt.figure()
    # plt.imshow(im, cmap='gray')
    # plt.show()

    return Screen(to_world, screensize, pixelsize, im)

def test_camera():
    camera = generate_test_camera()
    # print(camera)

    position_sample = torch.rand((*camera.filmsize, 2))
    ray = camera.sample_ray(position_sample)
    print(ray)

def test_screen():
    screen = generate_test_screen()
    print(screen)

# ----------------------------------------------------------------------------------------


if __name__ == "__main__":
    init()

    test_camera()
    test_screen()
