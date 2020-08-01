import data_util as du
import metric_util as mt

import numpy as np
import scipy.linalg
import tensor_util as tu
import draw_utils as dr
from nilearn import image
from nilearn import plotting


class RandomizedRPCA(object):
    
    def __init__(self, ground_truth_img, snr, lambda_reg, epsilon, logger, meta, rank, noise_type=None):
        
        self.ground_truth_img = ground_truth_img
        self.snr = snr
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        
        # gradient computation
        self.alpha = alpha
        self.regul = 1.0/(1.0 + self.lambda_reg)
        
        self.noise_type = noise_type
        self.max_tt_rank = rank
        
        self.logger = logger
        self.meta = meta
        
        self.init()
        
def _thresh(X, lambda1):
    res = np.abs(X) - lambda1
    return np.sign(X) * ((res > 0) * res)

def svd(X):
            return scipy.linalg.svd(X, full_matrices=False)

def rpca_godec(X, rank, fast=False, lambda1=None,
               power=None, tol=None, maxiter=None):

    # Get shape
    m, n = X.shape

    # Operate on transposed matrix for speed
    transpose = False
    if m < n:
        transpose = True
        X = X.T

    # Get shape
    m, n = X.shape
    
    lambda1 = 1.0 / np.sqrt(n)
    power = 0
    tol = 1e-3

    maxiter = 1e3
    # Check options if None
   
    # Initialize L and E
    L = X
    E = np.zeros(L.shape)

    itr = 0
    while True:
        itr += 1

        # Initialization with bilateral random projections
        Y2 = np.random.randn(n, rank)
        for i in range(power + 1):
            Y2 = np.dot(L.T, np.dot(L, Y2))
        Q, tmp = scipy.linalg.qr(Y2, mode='economic')

        # Estimate the new low-rank and sparse matrices
        Lnew = np.dot(np.dot(L, Q), Q.T)
        A = L - Lnew + E
        L = Lnew
        E = _thresh(A, lambda1)
        A -= E
        L += A

        # Check convergence
        eps = np.linalg.norm(A)
        if (eps < tol):
            print"Converged to %f in %d iterations" % (eps, itr)
            break
        elif (itr >= maxiter):
            print "Maximum iterations reached"
            break

    # Get the remaining Gaussian noise matrix
    G = X - L - E

    # Transpose back
    if transpose:
        L = L.T
        E = E.T
        G = G.T

    # Rescale
    Xhat = L
    Ehat = E
    Ghat = G

    # Do final SVD
    U, S, Vh = svd(Xhat)
    V = Vh.T

    # Chop small singular values which
    # likely arise from numerical noise
    # in the SVD.
    S[rank:] = 0.

    return Xhat, Ehat, Ghat, U, S, V

def compute_mse(x_true, x_hat):
        return (np.linalg.norm(x_true - x_hat)**2)/(np.linalg.norm(x_true)**2)

if __name__ == "__main__":
 
    folder = "/work/project/cmsc655/figures/godec/figures"
    subject_scan_path = du.get_full_path_subject1()
    mri_scan_img = mt.read_image_abs_path(subject_scan_path)
    x_true_data = np.array(mri_scan_img.get_data())
    
    original_tensor_shape = tu.get_tensor_shape(x_true_data)
    target_shape = mt.get_target_shape(x_true_data, 2)
    x_true_2D = mt.reshape_as_nD(x_true_data, 2,target_shape)
    
    print "Target Shape: " + str(x_true_2D.shape)
    x_hat, e_hat, g_hat, u, s, v = rpca_godec(x_true_2D, 20)
    
    mse = compute_mse(x_true_2D, x_hat)
    print "MSE Godec: " + str(mse)
    
    x_hat_img = mt.reconstruct_image_affine_d( mri_scan_img, x_hat, original_tensor_shape)
    e_hat_img = mt.reconstruct_image_affine_d( mri_scan_img, e_hat, original_tensor_shape)
    g_hat_img = mt.reconstruct_image_affine_d( mri_scan_img, g_hat, original_tensor_shape)
    
    fig_id = "x_low_rank_" + "_godec_" + "snr_" + str(0) + "_" + str("no_noise")
    
    # plot original image
    dr.draw_image(image.index_img(mri_scan_img,1), folder, fig_id, title=None)
    
    # plot low-rank image
    dr.draw_image(image.index_img(x_hat_img,1), folder, fig_id, title=None)
    
    # plot sparse-rank image
    dr.draw_image(image.index_img(e_hat_img,1), folder, fig_id, title=None)
    
    # plot gaussian image
    dr.draw_image(image.index_img(g_hat_img,1), folder, fig_id, title=None)
