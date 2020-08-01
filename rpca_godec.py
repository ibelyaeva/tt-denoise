import data_util as du
import metric_util as mt

import numpy as np
import scipy.linalg
import tensor_util as tu
import draw_utils as dr
from nilearn import image
from nilearn import plotting
import noise_util as nu
import pandas as pd
import os
import nibabel as nib

godec_dir = "/work/project/cmsc655/figures/godec/results/solution_go_dec.csv"
godec_dir_images = "/work/project/cmsc655/figures/godec/images"
godec_dir_scans = "/work/project/cmsc655/figures/godec/scans"

#NOT Project Code, but was run to generate results for Matrix Case


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
    
    #lambda1 = 1.0 / np.sqrt(n)
    power = 0
    tol = 1e-3

    maxiter = 1
    # Check options if None
   
    # Initialize L and E
    L = X
    E = np.zeros(L.shape)
    
    low_rank_rse_cost_history = []
    sparse_rank_rse_cost_history = []
    solution_cost = []

    itr = 0
    while True:
        itr += 1

        LOld = L
        # Initialization with bilateral random projections
        Y2 = np.random.randn(n, rank)
        for i in range(power + 1):
            Y2 = np.dot(L.T, np.dot(L, Y2))
        Q, tmp = scipy.linalg.qr(Y2, mode='economic')

        # Estimate the new low-rank and sparse matrices
        Lnew = np.dot(np.dot(L, Q), Q.T)
        A = L - Lnew + E
        L = Lnew
        Eold = E
        Enew = _thresh(A, lambda1)
        A -= Enew
        L += A
        
        low_rank_rel_error = compute_rel_error(L, LOld)
        sparse_rank_rel_error = compute_rel_error(Enew, Eold)
        
        low_rank_rse_cost_history.append(low_rank_rel_error)
        sparse_rank_rse_cost_history.append(sparse_rank_rel_error)
        
        print ("Iteration #: "  + str(itr) + "; Low Rank Rel Error: " + str(low_rank_rel_error) + "; Sparse Rank Relative Error: " + str( sparse_rank_rel_error))
        # Check convergence
        grad_norm = np.linalg.norm(A)
        
        solution_cost.append(grad_norm)
        
        print ("Iteration #: "  + str(itr) + "; " + str(low_rank_rel_error) + "; Residual Convergence: " + str(grad_norm))
        if (grad_norm < tol):
            print"Converged to %f in %d iterations" % (grad_norm, itr)
            break
        elif (itr >= maxiter):
            print "Maximum iterations reached"
            break
        
    save_solution(low_rank_rse_cost_history,sparse_rank_rse_cost_history, solution_cost)
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

def save_solution(l,s,t):
    res = {'low_rank_rel_error':l, 'sol_error': t}
    result = pd.DataFrame(res)
    result.to_csv(godec_dir)
    
    
def compute_mse(x_true, x_hat):
        return (np.linalg.norm(x_true - x_hat)**2)/(np.linalg.norm(x_true)**2)

def compute_rel_error(x_curr, x_prev):
        return (np.linalg.norm(x_curr - x_prev)**2)/(np.linalg.norm(x_prev)**2)

def save_solution_scans(suffix, folder, noise_type, snr, ground_truth_img, x_noisy_img, low_rank_img, x_sparse_img, guass_part_img): 
                    
        if noise_type:
            title = "fMRI Denoising" + " SNR = " + str(snr) + " Noise Type: " + noise_type
        else:
            title = "fMRI Denoising"
            
        x_init_img = ground_truth_img
        x_init_noisy_img = x_noisy_img
        x_true_path = os.path.join(folder,"x_true_img_" + str(suffix))
            
        x_noisy_path = os.path.join(folder,"x_noisy_img_" + str(suffix))
        low_rank_hat_path = os.path.join(folder,"x_low_rank_hat_img_" + str(suffix))
        sparse_hat_path = os.path.join(folder,"x_sparse_hat_img_" + str(suffix))
        x_guass_path = os.path.join(folder,"x_guass_hat_img_" + str(suffix))
            
        nib.save(ground_truth_img, x_true_path)
        nib.save(x_noisy_img, x_noisy_path)
               
        print("x_low_rank_hat_path: " + str(low_rank_hat_path))
        nib.save(low_rank_img, low_rank_hat_path)
            
        print("x_sparse_hat_path: " + str(sparse_hat_path))
        nib.save(x_sparse_img, sparse_hat_path)
            
        print("x_guass_hat_path: " + str(x_guass_path))
        nib.save(guass_part_img, x_guass_path)

               
        print("Images Folder: " + godec_dir_images)
        
        fig_id = "noise_free_decomposition_" + "snr_" + str(snr) + "_" + str(noise_type)
        
        # plot true image
        
        title =  "Noise free fMRI Images"
        dr.draw_x_true(image.index_img(ground_truth_img,1), godec_dir_images, fig_id, title=None)
        
        
        # all
        fig_id = "decomposition_" + "snr_" + str(snr) + "_" + str(noise_type)
        title =  "Low-Rank and Sparse Decomposition. SNR = " + str(snr) +  " Noise Type: " + noise_type
        dr.draw_decomposition_results(image.index_img(x_init_img,1), image.index_img(x_init_noisy_img,1), image.index_img(low_rank_img,1),
                                                  image.index_img(x_sparse_img,1), image.index_img(guass_part_img,1),godec_dir_images, fig_id, title)
        
        fig_id = "decomposition_" + "snr_" + str(snr) + "_" + str(noise_type)+ "_no_title"
        dr.draw_decomposition_results(image.index_img(x_init_img,1), image.index_img(x_init_noisy_img,1), image.index_img(low_rank_img,1),
                                                  image.index_img(x_sparse_img,1), image.index_img(x_sparse_img,1), godec_dir_images, fig_id, title=None)
        
        fig_id = "decomposition_" + "snr_" + str(snr) + "_" + str(noise_type)+ "_original"
        dr.draw_decomposition_results_run(image.index_img(x_init_img,1), image.index_img(x_init_noisy_img,1), image.index_img(low_rank_img,1),
                                                  image.index_img(x_sparse_img,1), image.index_img(guass_part_img,1), godec_dir_images, fig_id, title=None)
        
        
        fig_id = "x_true_" + "snr_" + str(snr) + "_" + str(noise_type)
        
        # plot true image
        dr.draw_image(image.index_img(x_init_img,1), godec_dir_images, fig_id, title)
        
        fig_id = "x_noisy_" + "snr_" + str(snr) + "_" + str(noise_type)
        # plot noisy image
        dr.draw_image(image.index_img(x_init_noisy_img,1),godec_dir_images, fig_id, title)
        
        fig_id = "x_low_rank_" + "snr_" + str(snr) + "_" + str(noise_type)
        # plot low-rank image
        dr.draw_image(image.index_img(low_rank_img,1), godec_dir_images, fig_id, title)
          
        fig_id = "x_sparse_" + "snr_" + str(snr) + "_" + str(noise_type)
        # plot sparse image
        dr.draw_image(image.index_img(x_sparse_img,1), godec_dir_images, fig_id, title)
              
        fig_id = "x_guass_" + "snr_" + str(snr) + "_" + str(noise_type) 
        # plot guass noise
        dr.draw_image(image.index_img(x_sparse_img,1), godec_dir_images, fig_id, title)
    
        
        
if __name__ == "__main__":
 
    folder = "/work/project/cmsc655/figures/godec/figures"
    subject_scan_path = du.get_full_path_subject1()
    mri_scan_img = mt.read_image_abs_path(subject_scan_path)
    x_true_data = np.array(mri_scan_img.get_data())
    
    original_tensor_shape = tu.get_tensor_shape(x_true_data)
    target_shape = mt.get_target_shape(x_true_data, 2)
    x_true_2D = mt.reshape_as_nD(x_true_data, 2,target_shape)
    
   
    norm_ground_x_init = np.linalg.norm(x_true_2D)
    x_init =  x_true_2D * (1./norm_ground_x_init)
    x_init_img = mt.reconstruct_image_affine_d( mri_scan_img, x_init, 2, original_tensor_shape)
    
    signal_w_noise = nu.add_richian_noise(x_init_img, x_init, 25)
    signal_w_noise_img = mt.reconstruct_image_affine_d( mri_scan_img, signal_w_noise, 2, original_tensor_shape)
            
    print "Target Shape: " + str(x_true_2D.shape)
    
    x_true_img = x_init_img
    x_noisy_img = signal_w_noise_img
    
    fig_id = "noise_free_decomposition_" + "snr_" + str(25) + "_" + str("richian")
    dr.draw_x_true(image.index_img(x_init_img,1), godec_dir_images, fig_id, title=None)