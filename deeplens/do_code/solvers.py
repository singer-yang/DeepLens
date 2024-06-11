""" In this file we implement several optimizers like Adam, Levenberg-Marquardt.
"""
from ..optics.basics import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import scipy.io
import torch.autograd.functional as F
from numbers import Number
from ..utils import print_memory

# for cv2 compatibility
plt.figure()
plt.close()
import cv2

class Optimization(DeepObj):
    """
    General class for design optimization.
    """
    def __init__(self, lens, diff_parameters_names):
        super(Optimization, self).__init__()
        self.lens = lens
        self.diff_parameters_names = []
        self.diff_parameters = []
        
        # TODO: re-sorting names to make sure strings go first
        # diff_parameters_names = sorted(diff_parameters_names, key=lambda x: (x is not None, '' if isinstance(x, Number) else type(x).__name__, x))
        # diff_parameters_names.reverse()
        
        for name in diff_parameters_names:
            # lens parameter name
            if type(name) is str: 
                self.diff_parameters_names.append(name)
                try:
                    exec('self.lens.{}.requires_grad = True'.format(name))
                except:
                    exec('self.lens.{name} = self.lens.{name}.detach()'.format(name=name))
                    exec('self.lens.{}.requires_grad = True'.format(name))
                exec('self.diff_parameters.append(self.lens.{})'.format(name))

            # actual parameter
            if type(name) is torch.Tensor: 
                name.requires_grad = True
                self.diff_parameters.append(name)

    # def optimize(self, )


class Adam(Optimization):
    def __init__(self, lens, diff_parameters_names, lr, lrs=None, beta=0.99, gamma_rate=None):
        # get gradient
        Optimization.__init__(self, lens, diff_parameters_names)
        self.lr = lr

        # optimizer
        if lrs is None:
            lrs = [1] * len(self.diff_parameters)
        self.optimizer = torch.optim.Adam(
            [{"params": v, "lr": lr*l} for v, l in zip(self.diff_parameters, lrs)],
            betas=(beta,0.999), amsgrad=True
        )

        # scheduler
        if gamma_rate is None:
            gamma_rate = 0.95
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=gamma_rate)


    def get_optimizer(self):
        return self.optimizer


    def optimize(self, y_ref, forward=None, maxit=50, record=True):
        print('optimizing ...')
        last_L = None
        lr = self.lr
        with torch.autograd.set_detect_anomaly(False): #True
            for it in range(maxit):
                # TODO: it is not good to use parameters in a general forward method.
                y = forward()
                L = torch.mean((y - y_ref)**2)
                

                self.optimizer.zero_grad()
                L.backward(retain_graph=True)
 
                print(f'iter = {it}: loss = {L.item()}')

                # descent
                for i, param in enumerate(self.diff_parameters):
                    new_param = param - lr * param.grad.item()
                    self.lens.set_param(self.diff_parameters_names[i], new_param)
                    

                if last_L is not None and L > last_L:
                    lr = lr * 0.9
                last_L = L.detach().item()


    def _change_parameters(self, xs, sign=True):
        diff_parameters = []
        for i, name in enumerate(self.diff_parameters_names):
            if sign:
                exec('self.lens.{name} = self.lens.{name} + xs[{i}]'.format(name=name,i=i))
            else:
                exec('self.lens.{name} = self.lens.{name} - xs[{i}]'.format(name=name,i=i))
            exec('diff_parameters.append(self.lens.{})'.format(name))
        for j in range(i+1, len(self.diff_parameters)):
            diff_parameters.append(self.diff_parameters[j] + 2*(sign - 0.5) * xs[j])
        return diff_parameters


class LM(Optimization): # Levenberg–Marquardt algorithm
    def __init__(self, lens, diff_parameters_names, lamb, mu=None, option='diag'):
        Optimization.__init__(self, lens, diff_parameters_names)
        self.lamb = lamb # damping factor
        self.mu = 2.0 if mu is None else mu # dampling rate (>1)
        self.option = option


    def jacobian(self, func, inputs, create_graph=False, strict=False):
        """ Compute gradient matrix,
            Constructs a M-by-N Jacobian matrix where M >> N.
            TODO: this function calls forward for several times, can we reduce frequency?

        Here, computing the Jacobian only makes sense for a tall Jacobian matrix. In this case,
        column-wise evaluation (forward-mode, or jvp) is more effective to construct the Jacobian.

        This function is modified from torch.autograd.functional.jvp().
        """

        Js = []
        outputs = func()
        M = outputs.shape

        # grad_outputs = (torch.zeros_like(outputs, requires_grad=True),)
        for x in inputs:
            # TODO: memory leak
            N = torch.numel(x)
            v = torch.ones(outputs.shape, device=x.device, requires_grad=True)
            # FIXME: here grad can be None
            # compute \partial outputs / \partial x
            vjp = torch.autograd.grad(outputs, x, grad_outputs=v, create_graph=True)[0].view(-1)

            if N == 1:
                J = torch.autograd.grad(vjp, v, grad_outputs=torch.ones(1, device=x.device))[0][...,None]
            else:
                I = torch.eye(N, device=x.device)
                J = []
                for i in range(N):
                    Ji = torch.autograd.grad(vjp, v, grad_outputs=I[i], retain_graph=True)[0]
                    J.append(Ji)
                J = torch.stack(J, axis=-1)
            Js.append(J)
        return torch.cat(Js, axis=-1)
        

    def optimize(self, func, func_yref_y, maxit=300, record=True):
        """
        Optimization function: (JtJ + lamb R) delta = Jt (y - f(beta))

        Inputs:
        - func: Evaluate `y = f(x)` where `x` is the implicit parameters by `self.diff_parameters` (out of the class)
        - func_yref_y: Compute `y_ref - y`

        Outputs:
        - ls: Loss function.
        """
        print('optimizing ...')
        Ns = [x.numel() for x in self.diff_parameters]
        NS = [[*x.shape] for x in self.diff_parameters]

        ls = [] # loss list
        Is = [] # y list
        lamb = self.lamb
        with torch.autograd.set_detect_anomaly(False):
            for it in range(maxit):
                y = func()
                Is.append(y.cpu().detach().numpy())

                # initial loss
                L = torch.mean(func_yref_y(y)**2).item()
                if L < 1e-16:
                    print('L too small; termiante.')
                    break

                # Jacobian
                J = self.jacobian(func, self.diff_parameters, create_graph=False)
                J = J.view(-1, J.shape[-1])
                JtJ = J.T @ J
                # N = JtJ.shape[0]

                # Regularization matrix
                if self.option == 'I':
                    R = torch.eye(JtJ.shape[0], device=JtJ.device)
                elif self.option == 'diag':
                    R = torch.diag(torch.diag(JtJ).abs())
                else:
                    R = torch.diag(self.option)

                # b = J.T @ (y_ref - y)
                # TODO: inaccurate jvp via pytorch function... Why?
                b = J.T @ func_yref_y(y).flatten()

                # Damping loop
                L_current = L + 1.0
                it_inner = 0
                while L_current >= L:
                    # Solve equation.
                    A = JtJ + lamb * R
                    x_delta = torch.solve(b[...,None], A)[0][...,0]

                    if torch.isnan(x_delta).sum():
                        print('x_delta NaN; Exiting damping loop')
                        break
                    x_delta_s = torch.split(x_delta, Ns)

                    # reshape if x is not 1D array
                    x_delta_s = [*x_delta_s]
                    for xi in range(len(x_delta_s)):
                        x_delta_s[xi] = torch.reshape(x_delta_s[xi],  NS[xi])

                    # update differentiable params `x += x_delta`
                    self.diff_parameters = self._change_parameters(x_delta_s, sign=True)

                    # calculate new loss
                    with torch.no_grad():
                        L_current = torch.mean(func_yref_y(func())**2).item()

                    del A

                    # terminate current iteration
                    if L_current < L:
                        # loss decreases
                        lamb /= self.mu
                        del x_delta_s
                        break
                    else:
                        # increase damping and undo the update
                        lamb *= 2.0*self.mu
                        # undo x, i.e. `x -= x_delta`
                        self.diff_parameters = self._change_parameters(x_delta_s, sign=False)

                    if lamb > 1e16:
                        print('lambda too big; Exiting damping loop.')
                        del x_delta_s
                        break

                    it_inner += 1
                    if it_inner > 20:
                        print('inner loop too many; Exiting damping loop.')
                        break

                # Exit damping loop.
                del JtJ, R, b

                # record
                x_increment = torch.mean(torch.abs(x_delta)).item()
                print('iter = {}: loss = {:.4e}, |x_delta| = {:.4e}'.format(
                    it, L, x_increment
                ))
                # print_memory()
                ls.append(L)
                if it > 0:
                    dls = np.abs(ls[-2] - L)
                    if dls < 1e-8:
                        print("|\Delta loss| = {:.4e} < 1e-8; Exiting LM loop.".format(dls))
                        break

                if x_increment < 1e-8:
                    print("|x_delta| = {:.4e} < 1e-8; Exiting LM loop.".format(x_increment))
                    break

        return {'ls': np.array(ls), 'Is': np.array(Is)}


    def _change_parameters(self, xs, sign=True):
        diff_parameters = []
        for i, name in enumerate(self.diff_parameters_names):
            if sign:
                exec('self.lens.{name} = self.lens.{name} + xs[{i}]'.format(name=name,i=i))
            else:
                exec('self.lens.{name} = self.lens.{name} - xs[{i}]'.format(name=name,i=i))
            exec('diff_parameters.append(self.lens.{})'.format(name))
        for j in range(i+1, len(self.diff_parameters)):
            diff_parameters.append(self.diff_parameters[j] + 2*(sign - 0.5) * xs[j])
        return diff_parameters
