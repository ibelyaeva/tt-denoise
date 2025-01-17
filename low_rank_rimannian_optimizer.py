import texfig

import tensorflow as tf
import numpy as np
import t3f
from nitime.analysis import snr
tf.set_random_seed(0)
np.random.seed(0)
import matplotlib.pyplot as plt
import metric_util as mt
import data_util as du
from t3f import shapes
from nilearn import image
from tensorflow.python.util import nest
import copy
from nilearn import plotting
from t3f import ops
import draw_utils as mrd
from t3f import initializers
from t3f import approximate
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from collections import OrderedDict
import pandas as pd
from scipy import stats
from nilearn.image import math_img
import cost_computation as cst
import tensor_util as tu
import nibabel as nib
import os
import os.path as op
import metadata as mdt
import noise_util as nu
import cost_utils as ct
import draw_utils as dr
import math
from nipype.algorithms import confounds as nac
import spike_detection as sp

class LowRankRiemannianGradientDescentOptimizer(object):
    
    def __init__(self, ground_truth_img, snr, lambda_reg, epsilon, alpha, logger, meta, max_tt_rank, noise_type=None):
        
        self.ground_truth_img = ground_truth_img
        self.snr = snr
        self.lambda_reg = lambda_reg
        self.epsilon = epsilon
        
        # gradient computation
        self.alpha = alpha
        self.regul = 1.0 / (1.0 + self.lambda_reg)
        
        self.noise_type = noise_type
        self.max_tt_rank = max_tt_rank
        
        self.logger = logger
        self.meta = meta
        
        self.init()
        
    def init(self):
        self.ground_truth = np.array(self.ground_truth_img.get_data()).astype('float32')
        self.low_rank_rse_cost_history = []
        self.sparse_rank_rse_cost_history = []
        self.solution_cost = []
        self.grad_low_rank_norm = []
        self.rel_solution_cost = []
        self.solution_grad_cost = []
        
        self.scan_mr_folder = self.meta.create_scan_mr_folder(self.snr)
        self.scan_mr_iteration_folder = self.meta.create_scan_mr_folder_iteration(self.snr)
        self.images_mr_folder_iteration = self.meta.create_images_mr_folder_iteration(self.snr)
        self.suffix = self.meta.get_suffix(self.snr)
        
        self.logger.info(self.scan_mr_iteration_folder)
        self.logger.info(self.suffix)
        
        self.solution_snr = None
        
        self.threshold = ct.compute_threshold(self.ground_truth)
        self.generate_file_names()
        self.init_noise_components()
        
        
    def init_noise_components(self):
        self.x_init = copy.deepcopy(self.ground_truth)
        self.norm_ground_x_init = np.linalg.norm(self.x_init)
        self.x_init = self.x_init * (1. / self.norm_ground_x_init)
        self.x_init_img = mt.reconstruct_image_affine(self.ground_truth_img, self.x_init)
             
        self.mask_img = compute_epi_mask(self.x_init_img)
        nib.save(self.mask_img, self.mask_path)
        
        if self.noise_type == 'rician':
            self.signal_w_noise = nu.add_richian_noise(self.x_init_img, self.x_init, self.snr)
        elif self.noise_type == 'gaussian':
            self.signal_w_noise = nu.add_gaussian_noise(self.x_init_img, self.x_init, self.snr)
        elif self.noise_type == 'rayleigh':
            self.signal_w_noise = nu.add_rayleigh_noise(self.x_init_img, self.x_init, self.snr)
        else:
            self.signal_w_noise = self.x_init
            
        self.norm_ground_noise = np.linalg.norm(self.signal_w_noise)
        self.signal_w_noise = self.signal_w_noise * (1. / self.norm_ground_noise)
        self.signal_w_noise_img = mt.reconstruct_image_affine(self.ground_truth_img, self.signal_w_noise)
        
        self.x_init_noisy = self.signal_w_noise.astype('float32')
        self.x_init_noisy_img = mt.reconstruct_image_affine(self.ground_truth_img, self.signal_w_noise)
        self.initial_snr = nu.SNRDb(self.x_init_noisy_img, self.x_init_noisy)
        self.logger.info("Initial SNR: " + str(self.initial_snr))
        
        self.corruption_error = self.compute_corruption_error(self.x_init, self.x_init_noisy)
        self.logger.info("Noise Corruption: " + str(self.corruption_error))    
    
    def init_algorithm(self):
        self.init_variables()
        
    def init_variables(self):
        
        tf.reset_default_graph()
        
        self.title = str(self.snr) + " SNR Low-Sparse Decomposition"
        
        # init tensor variables
        
        # create TT-Tensor by TT-SVD 
        self.ground_truth_tf = t3f.to_tt_tensor(self.x_init_noisy, max_tt_rank=self.max_tt_rank)
        
        # noisy ground truth
        self.X = t3f.get_variable('X', initializer=self.ground_truth_tf, trainable=False)
        
        # estimated low-rank component
        self.L = t3f.get_variable('L', initializer=self.ground_truth_tf)
        self.Lold = t3f.get_variable('Lold', initializer=self.ground_truth_tf)
        
        s_init = nu.init_random(self.x_init)
        s_init.fill(0)
        s_init_tf = t3f.to_tt_tensor(s_init, max_tt_rank=self.max_tt_rank)
        
        # estimated sparse component
        self.S = t3f.get_variable('S', initializer=s_init_tf)
        self.Sold = t3f.get_variable('Sold', initializer=s_init_tf)
        
        self.low_rank = np.zeros_like(self.x_init_noisy)
        self.sparse_part = np.zeros_like(self.x_init_noisy)
        self.G = np.zeros_like(self.x_init_noisy)
        
        self.logger.info("SNR : " + str(self.snr))
        
    def init_gradient_computation(self):
        pass
        # self.cost = t3f.get_variable('cost', initializer=0.0) 
    
    def define_train_operations(self):
        
        # projection onto normal space L via soft wavelet thresholding
        self.normal_space_projection = ct.cost_with_treshold(self.X - self.L, self.threshold)
               
        # Euclidean Gradient - Low Rank Component
        self.gradL = self.regul * (self.X - self.normal_space_projection)
        
        # Riemannian Gradient
        # project onto tangent space of L
        self.rimGradL = t3f.riemannian.project(self.gradL, self.L)
        self.low_rank_grad_norm = t3f.frobenius_norm(self.rimGradL)     
        
        # low rank update operation
        # gradient descent with step alpha along the tangent space
        # retract back to manifold
        self.low_rank_update_op = t3f.assign(self.L, t3f.round(self.L + self.alpha * self.rimGradL, max_tt_rank=self.max_tt_rank))
        
        # update sparse part
        self.sparse_part_update_op = t3f.assign(self.S, t3f.round(self.X - self.low_rank_update_op, max_tt_rank=self.max_tt_rank)) 
        
        # Update Cost
        self.train_step = self.compute_cost(self.low_rank_update_op, self.sparse_part_update_op)
        # self.cost_update_op = t3f.assign(self.cost, self.cost_op)
        
        # Update Relative Errors
        self.low_rank_rel_error = self.compute_rel_error(self.low_rank_update_op, self.Lold)
        
        self.sparse_rank_rel_error = self.compute_rel_error(self.sparse_part_update_op, self.Sold)  
        
        self.solution_rel_error = self.compute_solution_rel_error(self.low_rank_update_op, self.sparse_part_update_op)  
        
        self.solution_grad = t3f.frobenius_norm((self.low_rank_update_op - self.Lold) + (self.sparse_part_update_op - self.Sold))
        
        # save old values of low rank and sparse part
        # Lold
        self.low_rank_save_op = t3f.assign(self.Lold, self.low_rank_update_op)
        
        # SOld
        self.sparse_rank_save_op = t3f.assign(self.Sold, self.sparse_part_update_op)
        
        
    def compute_rel_error(self, x_curr, x_prev):
        return t3f.frobenius_norm_squared(x_curr - x_prev) / t3f.frobenius_norm_squared(x_prev)
    
    def compute_cost(self, low_rank, sparse_part):
        result = 0.5 * t3f.frobenius_norm_squared(low_rank + sparse_part - self.X) + 0.5 * self.lambda_reg * t3f.frobenius_norm_squared(low_rank)
        return result
    
    def compute_solution_rel_error(self, low_rank, sparse_part):
        result = t3f.frobenius_norm_squared(low_rank + sparse_part - self.X) / t3f.frobenius_norm_squared(self.X)
        return result
    
    def compute_corruption_error(self, x_true, x_hat):
        return (np.linalg.norm(x_true - x_hat)) / (np.linalg.norm(x_true))
    
    def optimize(self):
        self.logger.info("Starting Low Rank Decomposition.  Tensor Shape: " 
                         + str(self.ground_truth.shape) + "; Max Rank: " + str(self.max_tt_rank) + "; SNR: " + str(self.snr))
        
        
        self.init_algorithm()
        
        self.images_folder = self.meta.images_folder
        self.save_solution_scans(self.suffix, self.scan_mr_folder)
        self.generate_file_names()
        self.compute_initial_stat()
        
        self.init_gradient_computation()
        self.define_train_operations()
            
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 2
        config.inter_op_parallelism_threads = 2
        tf.Session(config=config)
        
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
        
              
        i = 0
        
        self.solution_rel_error_val = 1
        self.logger.info("Epsilon: " + str(self.epsilon))
        
        while True:
       
            
            cost_val, low_rank_rel_error, sparse_rank_rel_error, norm_grad_low_rank_val, solution_rel_error_val, solution_grad_val, _, _ = self.sess.run([self.train_step,
                                                                                                             self.low_rank_rel_error,
                                                                                                             self.sparse_rank_rel_error, self.low_rank_grad_norm,
                                                                                                             self.solution_rel_error, self.solution_grad,
                                                                                                             self.low_rank_save_op.op, self.sparse_rank_save_op.op])
            
            self.logger.info("Iteration #: " + str(i) + "; Cost: " + str(cost_val) + "; Low Rank Relative Error: " + str(low_rank_rel_error) + 
                        "; " + "; Sparse Rank Relative Error: " + str(sparse_rank_rel_error) + "; Grad Norm (low rank)" + str(norm_grad_low_rank_val)) 
                        
            self.logger.info("Solution cost: " + str(cost_val) + "; Solution Relative Error: " + str(solution_rel_error_val) + "; Solution Grad: " + str(solution_grad_val))
            
            # save history
            self.low_rank_rse_cost_history.append(low_rank_rel_error)
            self.sparse_rank_rse_cost_history.append(sparse_rank_rel_error)
            self.solution_cost.append(cost_val)
            self.grad_low_rank_norm.append(norm_grad_low_rank_val)
            self.rel_solution_cost.append(solution_rel_error_val)
            self.solution_grad_cost.append(solution_grad_val)
            
            self.save_cost_history()
            
            if i > 40:
                diff_train = (self.low_rank_rse_cost_history[i] - self.low_rank_rse_cost_history[i - 1]) / (self.low_rank_rse_cost_history[i - 1])
                
                if diff_train < self.epsilon:
                    self.logger.info("Breaking after " + str(i) + "; Iterations. Relative Cost less than solution tolerance")
                    break
                
                cost_new = self.low_rank_rse_cost_history[i]
                cost_old = self.low_rank_rse_cost_history[i - 1]
                
                if cost_new > cost_old:
                    self.logger.info("Breaking after " + str(i) + "; Iterations. Cost Increase.")
                    break
                
            i = i + 1
            
            
     
        # compute final results after algorithm completion
        self.low_rank, self.sparse_part = self.sess.run([t3f.full(self.L), t3f.full(self.S)])
        self.G = self.x_init_noisy - self.low_rank - self.sparse_part
        
        # compute final SNR after denoising 
    
        self.images_folder = self.meta.images_folder
        self.save_solution_scans(self.suffix, self.scan_mr_folder)
        self.save_cost_history()
        
        self.compute_final_stat()
        
        self.logger.info("Optimization completed after Iterations. " + str(i) + " Done.")
        print("Optimization completed after Iterations. " + str(i) + " Done.")
            
            
    def save_solution_scans(self, suffix, folder): 
        self.logger.info("SNR: " + str(self.snr) + "Noise Type: " + str(self.noise_type))
            
        if self.noise_type:
            title = "fMRI Denoising" + " SNR = " + str(self.snr) + " Noise Type: " + self.noise_type
        else:
            title = "fMRI Denoising"
            
        self.x_true_path = os.path.join(folder, "x_true_img_" + str(suffix))
            
        self.x_noisy_path = os.path.join(folder, "x_noisy_img_" + str(suffix))
        self.low_rank_hat_path = os.path.join(folder, "x_low_rank_hat_img_" + str(suffix))
        self.sparse_hat_path = os.path.join(folder, "x_sparse_hat_img_" + str(suffix))
        self.x_guass_path = os.path.join(folder, "x_guass_hat_img_" + str(suffix))
            
        self.logger.info("x_true_path: " + str(self.x_true_path))
        nib.save(self.x_init_img, self.x_true_path)
            
        self.logger.info("x_noisy_path: " + str(self.x_noisy_path))
        nib.save(self.x_init_noisy_img, self.x_noisy_path)
               
        self.logger.info("x_low_rank_hat_path: " + str(self.low_rank_hat_path))
        self.low_rank_img = mt.reconstruct_image_affine(self.ground_truth_img, self.low_rank)
        nib.save(self.low_rank_img, self.low_rank_hat_path)
            
        self.logger.info("x_sparse_hat_path: " + str(self.sparse_hat_path))
        self.sparse_part_img = mt.reconstruct_image_affine(self.ground_truth_img, self.sparse_part)
        nib.save(self.sparse_part_img, self.sparse_hat_path)
            
        self.logger.info("x_guass_hat_path: " + str(self.x_guass_path))
        self.guass_part_img = mt.reconstruct_image_affine(self.ground_truth_img, self.G)
        nib.save(self.guass_part_img, self.x_guass_path)
               
        self.logger.info("Images Folder: " + self.meta.images_folder)
        
        # plot true image
        
        title = "Noise free fMRI Images"
        fig_id = "noise_free_decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45"
        dr.draw_x_true(image.index_img(self.x_init_img, 45), self.meta.images_folder, fig_id, title=None)
        
        fig_id = "noise_free_decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        dr.draw_x_true(image.index_img(self.x_init_img, 80), self.meta.images_folder, fig_id, title=None)
        
        fig_id = "noise_free_decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_100"
        dr.draw_x_true(image.index_img(self.x_init_img, 100), self.meta.images_folder, fig_id, title=None)
        
        
        # all 45
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45"
        title = "Low-Rank and Sparse Decomposition. SNR = " + str(self.snr) + " Noise Type: " + self.noise_type
        dr.draw_decomposition_results(image.index_img(self.x_init_img, 45), image.index_img(self.x_init_noisy_img, 45), image.index_img(self.low_rank_img, 45),
                                                  image.index_img(self.sparse_part_img, 45), image.index_img(self.guass_part_img, 45), self.meta.images_folder, fig_id, title)
        
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) + "_no_title"+"_45"
        dr.draw_decomposition_results(image.index_img(self.x_init_img, 45), image.index_img(self.x_init_noisy_img, 45), image.index_img(self.low_rank_img, 45),
                                                  image.index_img(self.sparse_part_img, 45), image.index_img(self.guass_part_img, 1), self.meta.images_folder, fig_id, title=None)
        
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) + "_original"+"_45"
        dr.draw_decomposition_results_run(image.index_img(self.x_init_img, 45), image.index_img(self.x_init_noisy_img, 45), image.index_img(self.low_rank_img, 45),
                                                  image.index_img(self.sparse_part_img, 45), image.index_img(self.guass_part_img, 1), self.meta.images_folder, fig_id, title=None)
        
        dr.draw_x_true(image.index_img(self.x_init_img, 45), self.meta.images_folder, fig_id, title=None)
        
        
        # all 80
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        title = "Low-Rank and Sparse Decomposition. SNR = " + str(self.snr) + " Noise Type: " + self.noise_type
        dr.draw_decomposition_results(image.index_img(self.x_init_img, 80), image.index_img(self.x_init_noisy_img, 80), image.index_img(self.low_rank_img, 80),
                                                  image.index_img(self.sparse_part_img, 80), image.index_img(self.guass_part_img, 80), self.meta.images_folder, fig_id, title)
        
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) + "_no_title"+"_80"
        dr.draw_decomposition_results(image.index_img(self.x_init_img, 80), image.index_img(self.x_init_noisy_img, 80), image.index_img(self.low_rank_img, 80),
                                                  image.index_img(self.sparse_part_img, 80), image.index_img(self.guass_part_img, 1), self.meta.images_folder, fig_id, title=None)
        
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) + "_original"+"_80"
        dr.draw_decomposition_results_run(image.index_img(self.x_init_img, 80), image.index_img(self.x_init_noisy_img, 80), image.index_img(self.low_rank_img, 80),
                                                  image.index_img(self.sparse_part_img, 80), image.index_img(self.guass_part_img, 1), self.meta.images_folder, fig_id, title=None)
        
        # 100 
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        title = "Low-Rank and Sparse Decomposition. SNR = " + str(self.snr) + " Noise Type: " + self.noise_type
        dr.draw_decomposition_results(image.index_img(self.x_init_img, 100), image.index_img(self.x_init_noisy_img, 80), image.index_img(self.low_rank_img, 100),
                                                  image.index_img(self.sparse_part_img, 100), image.index_img(self.guass_part_img, 80), self.meta.images_folder, fig_id, title)
        
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) + "_no_title"+"_80"
        dr.draw_decomposition_results(image.index_img(self.x_init_img, 100), image.index_img(self.x_init_noisy_img, 100), image.index_img(self.low_rank_img, 100),
                                                  image.index_img(self.sparse_part_img, 100), image.index_img(self.guass_part_img, 1), self.meta.images_folder, fig_id, title=None)
        
        fig_id = "decomposition_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) + "_original"+"_80"
        dr.draw_decomposition_results_run(image.index_img(self.x_init_img, 100), image.index_img(self.x_init_noisy_img, 100), image.index_img(self.low_rank_img, 100),
                                                  image.index_img(self.sparse_part_img, 100), image.index_img(self.guass_part_img, 1), self.meta.images_folder, fig_id, title=None)
        
        fig_id = "x_true_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45"
        
        # 45
        # plot true image
        dr.draw_image(image.index_img(self.x_init_img, 45), self.meta.images_folder, fig_id, title)
        
        fig_id = "x_noisy_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45"
        # plot noisy image
        dr.draw_image(image.index_img(self.x_init_noisy_img, 45), self.meta.images_folder, fig_id, title)
        
        fig_id = "x_low_rank_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45"
        # plot low-rank image
        dr.draw_image(image.index_img(self.low_rank_img, 45), self.meta.images_folder, fig_id, title)
          
        fig_id = "x_sparse_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45"
        # plot sparse image
        dr.draw_image(image.index_img(self.sparse_part_img, 45), self.meta.images_folder, fig_id, title)
              
        fig_id = "x_guass_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_45" 
        # plot guass noise
        dr.draw_image(image.index_img(self.guass_part_img, 45), self.meta.images_folder, fig_id, title)
    
        self.solution_snr = nu.SNRDb(self.low_rank_img, self.low_rank)
        
        # 80
        # plot true image
        fig_id = "x_true_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        dr.draw_image(image.index_img(self.x_init_img, 80), self.meta.images_folder, fig_id, title)
        
        fig_id = "x_noisy_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        # plot noisy image
        dr.draw_image(image.index_img(self.x_init_noisy_img, 80), self.meta.images_folder, fig_id, title)
        
        fig_id = "x_low_rank_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        # plot low-rank image
        dr.draw_image(image.index_img(self.low_rank_img, 80), self.meta.images_folder, fig_id, title)
          
        fig_id = "x_sparse_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_80"
        # plot sparse image
        dr.draw_image(image.index_img(self.sparse_part_img, 80), self.meta.images_folder, fig_id, title)
              
        fig_id = "x_guass_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) +"_80"
        # plot guass noise
        dr.draw_image(image.index_img(self.guass_part_img, 80), self.meta.images_folder, fig_id, title)
    
       # 100
       # plot true image
        fig_id = "x_true_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_100"
        dr.draw_image(image.index_img(self.x_init_img, 100), self.meta.images_folder, fig_id, title)
        
        fig_id = "x_noisy_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_100"
        # plot noisy image
        dr.draw_image(image.index_img(self.x_init_noisy_img, 100), self.meta.images_folder, fig_id, title)
        
        fig_id = "x_low_rank_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_100"
        # plot low-rank image
        dr.draw_image(image.index_img(self.low_rank_img, 100), self.meta.images_folder, fig_id, title)
          
        fig_id = "x_sparse_" + "snr_" + str(self.snr) + "_" + str(self.noise_type)+"_100"
        # plot sparse image
        dr.draw_image(image.index_img(self.sparse_part_img, 100), self.meta.images_folder, fig_id, title)
              
        fig_id = "x_guass_" + "snr_" + str(self.snr) + "_" + str(self.noise_type) +"_100"
        # plot guass noise
        dr.draw_image(image.index_img(self.guass_part_img, 100), self.meta.images_folder, fig_id, title)
            
            
    def save_cost_history(self):
        
        low_rank_cost_out = OrderedDict()
        low_rank_cost_arr = []
        
        sparse_rank_rse_cost_arr = []
        solution_cost_arr = []
        grad_low_rank_norm_arr = []
        rel_solution_cost_arr = []
        
        indices = []
        snr_arr = []
        threshold_arr = []
        noise_type_arr = []
        initial_snr_arr = []
        solution_snr_arr = []
        corruption_error_arr = []
        solution_grad_arr = []
        
              
        counter = 0
        for item in  self.low_rank_rse_cost_history:
            low_rank_cost_arr.append(item)
            sparse_rank_rse_cost_arr.append(min(self.sparse_rank_rse_cost_history[counter], 1))
            solution_cost_arr.append(self.solution_cost[counter])
            grad_low_rank_norm_arr.append(self.grad_low_rank_norm[counter])
            rel_solution_cost_arr.append(self.rel_solution_cost[counter])
            corruption_error_arr.append(self.corruption_error)
            solution_grad_arr.append(self.solution_grad_cost[counter])
                       
            indices.append(counter)
            snr_arr.append(self.snr)
            threshold_arr.append(self.threshold)
            noise_type_arr.append(self.noise_type)
            initial_snr_arr.append(self.initial_snr)
            
            if self.solution_snr:
                solution_snr_arr.append(self.solution_snr)
            
            counter = counter + 1
                   
        low_rank_cost_out['k'] = indices
        low_rank_cost_out['snr'] = snr_arr
        low_rank_cost_out['noise_type'] = noise_type_arr
        low_rank_cost_out['low_rank_rse'] = low_rank_cost_arr
        low_rank_cost_out['sparse_rank_rse'] = sparse_rank_rse_cost_arr
        low_rank_cost_out['grad_low_rank_norm'] = grad_low_rank_norm_arr
        low_rank_cost_out['rel_solution_cost'] = rel_solution_cost_arr
        low_rank_cost_out['solution_cost'] = solution_cost_arr
        low_rank_cost_out['initial_snr'] = initial_snr_arr
        low_rank_cost_out['corruption_error'] = corruption_error_arr
        low_rank_cost_out['solution_grad'] = solution_grad_arr
                
        if self.solution_snr:
            low_rank_cost_out['solution_snr'] = solution_snr_arr
           
        output_df = pd.DataFrame(low_rank_cost_out, index=indices)
        results_folder = self.meta.results_folder
        
        fig_id = 'solution_cost' + '_' + self.suffix
        dr.save_csv_by_path(output_df, results_folder, fig_id)
    
    def compute_initial_stat(self):
        
        dvar, tsnr = self.compute_stat(self.x_noisy_path, self.init_tsnr_path, 
                          self.init_mean_path, self.init_stdev_path)
        
        col_names = ['avg_std', 'avg_nstd', 'avg_vxstd', 'avg_tsnr']
     
        dvar_file = dvar._results["out_all"]
        print(dvar_file)
             
        # load all dvar   
        base_file_name=os.path.basename(self.x_noisy_path)       
        fname, in_ext = op.splitext(base_file_name)    
        dvar_file_name = fname + "_dvars.tsv" 
          
        #load dvar file
        df_file = pd.read_csv(dvar_file_name,sep="\t", names=['std_var', 'non_std_var', 'vx_wise_std'])
                  
        fig_id = 'initial_dvar_all'
        dr.save_csv_by_path(df_file, self.meta.results_folder, fig_id) 
        
        # compute tsnr
        tu.compute_tsnr(self.x_noisy_path, self.init_tsnr_path)
        
        tsnr_img = mt.read_image_abs_path(self.init_tsnr_path)
        tsnr_img_data = np.array(tsnr_img.get_data())
        tsnr_mean = np.mean(tsnr_img_data)
        
        col_names_tsnr = ['avg_tsnr']
    
        avg_tsnr_result = pd.DataFrame(col_names_tsnr)
        avg_tsnr_result['avg_tsnr'] = tsnr_mean
        
        fig_id = 'initial_tsnr'
        dr.save_csv_by_path(avg_tsnr_result, self.meta.results_folder, fig_id) 
        
        avg_result = pd.DataFrame(col_names)   
        avg_result['avg_std'] = dvar._results["avg_std"]
        avg_result['avg_nstd'] = dvar._results["avg_nstd"]
        avg_result['avg_vxstd'] = dvar._results["avg_vxstd"]
        avg_result['avg_tsnr'] = tsnr_mean
        
        #avg dvar
        fig_id = 'initial_avg_dvar'
        dr.save_csv_by_path(avg_result, self.meta.results_folder, fig_id) 
        print(avg_result)
        
        
    def compute_final_stat(self):
        
        self.generate_file_names()
        
        folder = self.scan_mr_folder
        suffix = self.suffix
        self.low_rank_hat_path = os.path.join(folder, "x_low_rank_hat_img_" + str(suffix)+".nii")
        
        dvar, tsnr = self.compute_stat(self.low_rank_hat_path, self.final_tsnr_path, 
                          self.final_mean_path, self.final_stdev_path)
        
        col_names = ['avg_std', 'avg_nstd', 'avg_vxstd', 'avg_tsnr']
    
        avg_result = pd.DataFrame(col_names)   
        avg_result['avg_std'] = dvar._results["avg_std"]
        avg_result['avg_nstd'] = dvar._results["avg_nstd"]
        avg_result['avg_vxstd'] = dvar._results["avg_vxstd"]
        print(avg_result)
        
        dvar_file = dvar._results["out_all"]
        print(dvar_file)
             
        # load all dvar
        
        base_file_name=os.path.basename(self.low_rank_hat_path)       
        fname, in_ext = op.splitext(base_file_name)
        dvar_file_name = fname + "_dvars.tsv" 
        
        df_file = pd.read_csv(dvar_file_name,sep="\t", names=['std_var', 'non_std_var', 'vx_wise_std'])
        
        fig_id = 'final_dvar_all'
        dr.save_csv_by_path(df_file, self.meta.results_folder, fig_id)
        
        #compute tsnr
        tu.compute_tsnr(self.low_rank_hat_path, self.final_tsnr_path)
        tsnr_img = mt.read_image_abs_path(self.final_tsnr_path)
        tsnr_img_data = np.array(tsnr_img.get_data())
        tsnr_mean = np.mean(tsnr_img_data)
        col_names = ['avg_tsnr']
    
        avg_tsnr_result = pd.DataFrame(col_names)
        avg_tsnr_result['avg_tsnr'] = tsnr_mean
        
        fig_id = 'final_tsnr'
        dr.save_csv_by_path(avg_tsnr_result, self.meta.results_folder, fig_id)
        
        avg_result = pd.DataFrame(col_names)   
        avg_result['avg_std'] = dvar._results["avg_std"]
        avg_result['avg_nstd'] = dvar._results["avg_nstd"]
        avg_result['avg_vxstd'] = dvar._results["avg_vxstd"]
        avg_result['avg_tsnr'] = tsnr_mean
        
        #avg dvar
        fig_id = 'final_avg_dvar'
        dr.save_csv_by_path(avg_result, self.meta.results_folder, fig_id) 
        print(avg_result)
        
        #compute diff tsnr
        self.init_tsnr_img = mt.read_image_abs_path(self.init_tsnr_path)
        self.final_tsnr_img = mt.read_image_abs_path(self.final_tsnr_path)
        
        self.diff_tsnr_img = image.math_img("img1 - img2", img1=self.final_tsnr_img, img2= self.init_tsnr_img)
        nib.save(self.diff_tsnr_img, self.diff_tsnr_path)
        
        self.ratio_tsnr_img = image.math_img("img1/img2", img1=self.final_tsnr_img, img2= self.init_tsnr_img)
        nib.save(self.diff_tsnr_img, self.diff_tsnr_path)       
            
        fig_id = 'final_tsnr.pdf'
        dr.draw_image_z(self.final_tsnr_img, self.meta.images_folder, fig_id, cut_coords=[8])
        
        fig_id = 'init_tsnr.pdf'
        dr.draw_image_z(self.init_tsnr_img, self.meta.images_folder, fig_id, cut_coords=[8])
        
        fig_id = 'diff_tsnr.pdf'
        dr.draw_image_z(self.diff_tsnr_img, self.meta.images_folder, fig_id, cut_coords=[8])
        
        fig_id = 'ratio_tsnr.pdf'
        dr.draw_image_z(self.ratio_tsnr_img, self.meta.images_folder, fig_id, cut_coords=[8])
        
        self.init_stddev_img = mt.read_image_abs_path(self.init_stdev_path)
        self.final_stddev_img = mt.read_image_abs_path(self.final_stdev_path)
        
        fig_id = 'final_stddev.pdf'
        dr.draw_image_z(self.final_stddev_img, self.meta.images_folder, fig_id, cut_coords=[8])
        
        fig_id = 'init_stddev.pdf'
        dr.draw_image_z(self.init_stddev_img, self.meta.images_folder, fig_id, cut_coords=[8])
    
    
    def compute_stat(self, infile_path, tnsr_path, mean_path, stdev_path):
        
        self.generate_file_names()
        
        dvars = nac.ComputeDVARS()
        dvars.inputs.in_file = infile_path
        dvars.inputs.in_mask = self.mask_path 
        dvars.inputs.save_std = True
        dvars.inputs.save_nstd = True
        dvars.inputs.save_vxstd = True
        dvars.inputs.save_all = True
        dvars.inputs.save_plot = True
        dvars.inputs.figformat = 'pdf'
        dvars.run()
    
        tsnr_file = tnsr_path
        mean_file = mean_path
        stddev_file = stdev_path
        tsnr = nac.TSNR()
        tsnr.inputs.in_file = infile_path
        tsnr.inputs.tsnr_file = tsnr_file
        tsnr.inputs.mean_file = mean_file
        tsnr.inputs.stddev_file = stddev_file
        tsnr.run()
        
        return dvars, tsnr
    
    def generate_file_names(self):  
        
        folder = self.scan_mr_folder
        suffix = self.suffix
        self.x_true_path = os.path.join(folder, "x_true_img_" + str(suffix)+".nii")
        self.mask_path = os.path.join(folder, "epi_mask.nii")
            
        self.x_noisy_path = os.path.join(folder, "x_noisy_img_" + str(suffix)+".nii")
        self.low_rank_hat_path = os.path.join(folder, "x_low_rank_hat_img_" + str(suffix)+".nii")
        self.sparse_hat_path = os.path.join(folder, "x_sparse_hat_img_" + str(suffix)+".nii")
        self.x_guass_path = os.path.join(folder, "x_guass_hat_img_" + str(suffix)+".nii")
        
        self.init_tsnr_path = os.path.join(folder, "init" + "_tsnr_" + ".nii.gz")
        self.init_mean_path = os.path.join(folder, "init" + "_mean_" + ".nii.gz")
        self.init_stdev_path = os.path.join(folder, "init" + "_stdev_" + ".nii.gz")
        
        self.final_tsnr_path = os.path.join(folder, "final" + "_tsnr_" + ".nii.gz")
        self.final_mean_path = os.path.join(folder, "final" + "_mean_" + ".nii.gz")
        self.final_stdev_path = os.path.join(folder, "final" + "_stdev_" + ".nii.gz")
        
        self.diff_tsnr_path = os.path.join(folder, "diff" + "_tsnr_" + ".nii.gz")
        self.ratio_tsnr_path = os.path.join(folder, "ratio" + "_tsnr_" + ".nii.gz")
        
        self.diff_var_path = os.path.join(folder, "diff" + "_var_" + ".nii.gz")
        self.diff_std_path = os.path.join(folder, "diff" + "_std_" + ".nii.gz")
        
        self.initial_dvar_path = os.path.join(folder, "x_noisy_dvar.csv")
        self.final_dvar_path = os.path.join(folder, "x_low_rank_dvar.csv")

        
            
            
             
        

    
