import texfig

import os
import data_util as du
import configparser
from os import path
import metadata as mdt
import low_rank_rimannian_optimizer as opt
import metric_util as mt


config_loc = path.join('config')
config_filename = 'low_rank_solution.config'
config_file = os.path.join(config_loc, config_filename)
config = configparser.ConfigParser()
config.read(config_file)
config = config

def decompose_richian_noise():
    
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('richian', 4)
    root_dir = config.get('log', 'resuts.dir.richian')
   
    lamda=0.000001
    epsilon=1e-7
    
    snr_list = [40, 35, 25, 20, 15, 10, 5]
    
    for item in snr_list:   
        solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
        mri_scan_img = mt.read_image_abs_path(subject_scan_path)
        optimization_runner = opt.LowRankRiemannianGradientDescentOptimizer(mri_scan_img, item, lamda, epsilon, 0.01, meta.logger, meta, 63,noise_type='rician')
        optimization_runner.optimize()

def decompose_gaussian_noise():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('gaussian', 4)
    root_dir = config.get('log', 'resuts.dir.gaussian')
   
    lamda=0.000001
    epsilon=1e-7
    
    snr_list = [40, 35, 25, 20, 15, 10, 5]
        
    for item in snr_list:   
        solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
        mri_scan_img = mt.read_image_abs_path(subject_scan_path)
        optimization_runner = opt.LowRankRiemannianGradientDescentOptimizer(mri_scan_img, item, lamda, epsilon, 0.001, meta.logger, meta, 63,noise_type='gaussian')
        optimization_runner.optimize()

def separate_rayleigh_noise():
    subject_scan_path = du.get_full_path_subject1()
    meta = mdt.Metadata('gaussian', 4)
    root_dir = config.get('log', 'resuts.dir.rayleigh')
   
    lamda=0.000001
    epsilon=1e-9
    
    snr_list = [40, 35, 25, 20, 15, 10, 5]
        
    for item in snr_list:   
        solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    
        mri_scan_img = mt.read_image_abs_path(subject_scan_path)
        optimization_runner = opt.LowRankRiemannianGradientDescentOptimizer(mri_scan_img, item, lamda, epsilon, 0.001, meta.logger, meta, 63,noise_type='rayleigh')
        optimization_runner.optimize()

def separate_original_noise():
    #subject_scan_path = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45/x_miss_45.nii"
    #subject_scan_path = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80/x_miss_45_80.nii"
    subject_scan_path = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80-100/x_miss_45_80_100.nii"

    meta = mdt.Metadata('gaussian', 4)
    root_dir = config.get('log', 'resuts.dir.nonoise')
   
    lamda=0.000001
    epsilon=1e-7
    
    snr_list = [50]
    
    solution_dir, movies_folder, images_folder, results_folder, reports_folder, scans_folder, scans_folder_final, scans_folder_iteration = meta.init_meta_data(root_dir)
    mri_scan_img = mt.read_image_abs_path(subject_scan_path)
    optimization_runner = opt.LowRankRiemannianGradientDescentOptimizer(mri_scan_img, 0, lamda, epsilon, 0.001, meta.logger, meta, 63,noise_type='no_noise')
    optimization_runner.optimize()       

if __name__ == "__main__":
    pass

    #decompose_richian_noise()
    #decompose_gaussian_noise()
    #separate_rayleigh_noise()
    separate_original_noise()
    