"""Total variation denoising."""

import argparse
import time

import numpy as np
from PIL import Image
from scipy.linalg import blas

EPS = np.finfo(np.float32).eps

import data_util as du
import metric_util as mt

import numpy as np
import scipy.linalg
import tensor_util as tu
import draw_utils as dr
from nilearn import image
from nilearn import plotting


def dot(x, y):
    """Returns the dot product of two arrays with the same shape."""
    return blas.sdot(x.reshape(-1), y.reshape(-1))

dot(np.array(0), np.array(0))


def axpy(a, x, y):
    """Sets y = a*x + y and returns y."""
    shape = x.shape
    x, y = x.reshape(-1), y.reshape(-1)
    return blas.saxpy(x, y, a=a).reshape(shape)

axpy(1, np.array(0), np.array(0))


def tv_norm(x):
    """Computes the total variation norm and its gradient. From jcjohnson/cnn-vis."""
    x_diff = x - np.roll(x, -1, axis=1)
    y_diff = x - np.roll(x, -1, axis=0)
    grad_norm2 = x_diff**2 + y_diff**2 + EPS
    norm = np.sum(np.sqrt(grad_norm2))
    dgrad_norm = 0.5 / np.sqrt(grad_norm2)
    dx_diff = 2 * x_diff * dgrad_norm
    dy_diff = 2 * y_diff * dgrad_norm
    grad = dx_diff + dy_diff
    grad[:, 1:, :] -= dx_diff[:, :-1, :]
    grad[1:, :, :] -= dy_diff[:-1, :, :]
    return norm, grad


def l2_norm(x):
    """Computes 1/2 the square of the L2-norm and its gradient."""
    return np.sum(x**2) / 2, x


class LBFGSOptimizer:
    """Implements the L-BFGS quasi-Newton optimizer."""
    def __init__(self, params, opfunc, step_size=1, n_corr=10, c1=1e-4, c2=0.9, max_ls_fevals=10):
        """Initializes the optimizer."""
        self.params = params
        self.opfunc = opfunc
        self.step_size = step_size
        self.n_corr = n_corr
        self.c1 = c1
        self.c2 = c2
        self.max_ls_fevals = max_ls_fevals
        self.step = 0
        self.fevals = 0
        self.loss = None
        self.grad = None
        self.sk = []
        self.yk = []

    def update(self):
        """Returns a step's parameter update."""
        self.step += 1

        if self.step == 1:
            self.loss, self.grad = self.opfunc(self.params)
            self.fevals += 1

        # Line search.
        step_size, step_min, step_max = 1, 0, np.inf
        ls_fevals = 0
        while True:

            if ls_fevals == self.max_ls_fevals:
                break
            # Compute search direction, step, loss, and gradient
            p = -self.inv_hv(self.grad)
            s = step_size * p
            loss, grad = self.opfunc(self.params + s)
            self.fevals += 1
            y = grad - self.grad
            ls_fevals += 1

            # Test that the weak Wolfe curvature condition holds
            if dot(p, grad) < self.c2 * dot(p, self.grad):
                step_min = step_size
            # Test that the Armijo condition holds
            elif loss > self.loss + self.c1 * step_size * dot(p, self.grad):
                step_max = step_size
                self.store_curvature_pair(s, y)
            # Both hold, accept the step
            else:
                break

            # Compute new step size
            if step_max < np.inf:
                step_size = (step_min + step_max) / 2
            else:
                step_size *= 2

        # Update params
        self.params += s

        # Store curvature pair and gradient
        self.store_curvature_pair(s, y)
        self.loss, self.grad = loss, grad
        return loss, self.params

    def store_curvature_pair(self, s, y):
        """Updates the L-BFGS memory with a new curvature pair."""
        self.sk.append(s)
        self.yk.append(y)
        if len(self.sk) > self.n_corr:
            self.sk, self.yk = self.sk[1:], self.yk[1:]

    def inv_hv(self, p):
        """Computes the product of a vector with an approximation of the inverse Hessian."""
        p = p.copy()
        alphas = []
        for s, y in zip(reversed(self.sk), reversed(self.yk)):
            alphas.append(dot(s, p) / (dot(s, y)) + EPS)
            axpy(-alphas[-1], y, p)

        if len(self.sk) > 0:
            s, y = self.sk[-1], self.yk[-1]
            p *= dot(s, y) / (dot(y, y) + EPS)
        else:
            p /= np.sqrt(dot(p, p) / p.size) + EPS

        for s, y, alpha in zip(self.sk, self.yk, reversed(alphas)):
            beta = dot(y, p) / (dot(s, y) + EPS)
            axpy(alpha - beta, s, p)

        return p


def main():
    """The main function."""
    folder = "/work/project/cmsc655/figures/godec/figures"
    subject_scan_path = du.get_full_path_subject1()
    mri_scan_img = mt.read_image_abs_path(subject_scan_path)
    x_true_data = np.array(mri_scan_img.get_data())
    
    original_tensor_shape = tu.get_tensor_shape(x_true_data)
    target_shape = mt.get_target_shape(x_true_data, 3)
    x_true_2D = mt.reshape_as_nD(x_true_data, 3,target_shape)
    
    norm_ground_x_init = np.linalg.norm(x_true_2D)
    x_init =  x_true_2D * (1./norm_ground_x_init)
    
    img = x_init.copy()
    orig_img = x_init.copy()

    step_size = 1
    lmbda = 0.2

    def opfunc(img):
        tv_loss, tv_grad = tv_norm(img)
        l2_loss, l2_grad = l2_norm(img - orig_img)
        loss = tv_loss + l2_loss/lmbda
        grad = tv_grad + l2_grad/lmbda
        return loss, grad

    last_loss = np.inf
    steps = 0
   
    print('Optimizing using gradient descent.')
    while True:
        steps += 1
        loss, grad = opfunc(img)
        print('step:', steps, 'loss:', loss)
        if loss > last_loss:
            break
        last_loss = loss
        axpy(-step_size, grad, img)
    
    img = x_init.copy()
       
    opt = LBFGSOptimizer(img, opfunc, n_corr=4)
    last_loss = np.inf
    steps = 0

    
    print('\nOptimizing using L-BFGS.')
    while True:
        steps += 1
        loss, img[:] = opt.update()
        print('step:', steps, 'loss:', loss)
        if loss * 1.01 > last_loss:
            break
        last_loss = loss
   
if __name__ == '__main__':
    main()