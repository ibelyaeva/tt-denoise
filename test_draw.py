import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import t3f
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
import mri_draw_utils as mrd
from t3f import initializers
from t3f import approximate
from scipy import optimize 
from nilearn.masking import compute_background_mask
from nilearn.masking import compute_epi_mask
from collections import OrderedDict
import pandas as pd
from scipy import stats
from nilearn.image import math_img
from t3f import initializers
import numpy.testing as nps
from dipy.sims.voxel import add_noise
import nibabel as nib
import noise_util as nu
import draw_utils as dr
import os
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
import math
import fmri_plot as fm
from nilearn import datasets
from nipype.algorithms import confounds as nac


tex_width = 4.78853
default_ratio = (math.sqrt(5.0) - 1.0) / 2.0

def draw_image1(img, folder, fig_id, title):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    fig = plt.figure(frameon = False, figsize=(5,3))
         
    grid = gridspec.GridSpec(1,1, top=0.95, bottom = 0.05, hspace=0, wspace=0)
    
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
        
    main_ax = fig.add_subplot(grid[0, 0])
    y_image = plotting.plot_epi(img, annotate=True, draw_cross=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords = None)     
    
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    
    dr.save_fig_pdf(fig_path)
    
def draw_image2(img, folder, fig_id, title):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    fig = plt.figure(frameon = False, figsize=(5,2.5))
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
        
    y_image = plotting.plot_epi(img, annotate=True, draw_cross=False, bg_img=None,black_bg=True, figure= fig, cmap='jet', cut_coords = None)     
      
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
      
    dr.save_fig_pdf(fig_path)
    
def draw_composition_per_multiple_corruptions(images, fig_id, width, title=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    
    plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=True)
        
    fig, ((ax0, ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8, ax9), (ax10, ax11, ax12, ax13, ax14)) = plt.subplots(ncols=5, nrows = 3, sharey=True, sharex=True)
    
    fig.tight_layout()
    fig.subplots_adjust(top=0.9, wspace = 0.05, hspace=0.05, left=0.05, right = 0.95) 
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
    
    ax_list0 = []
    ax_list0.append((ax0, ax1, ax2, ax3, ax4)) 
    
    ax_list1 = []
    ax_list1.append((ax5, ax6, ax7, ax8, ax9)) 
    
    ax_list2= []
    ax_list2.append((ax10, ax11, ax12, ax13, ax14)) 
    
    axes = []
    
    axes.append(ax_list0)
    axes.append(ax_list1)
    axes.append(ax_list2)
    
    row0 = images[0]
    row1 = images[1]
    row2 = images[2]
     
    img0 = row0[0]
    img1 = row0[1]
    img2 = row0[2]
    img3 = row0[3]
    img4 = row0[4]
    
    img5 = row1[0]
    img6 = row1[1]
    img7 = row1[2]
    img8 = row1[3]
    img9 = row1[4]
    
    img10 = row2[0]
    img11 = row2[1]
    img12 = row2[2]
    img13 = row2[3]
    img14 = row2[4]
     
    ax0.set_yticklabels([])
    ax0.set_xticklabels([])
    
    ax0.xaxis.set_major_locator(plt.NullLocator())
    ax0.yaxis.set_major_locator(plt.NullLocator())
    
    ax0.spines['right'].set_visible(False)
    ax0.spines['top'].set_visible(False)
    ax0.spines['bottom'].set_visible(False)
    ax0.spines['left'].set_visible(False)
    
    ax1.set_yticklabels([])
    ax1.set_xticklabels([])
    
    ax1.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    ax2.set_yticklabels([])
    ax2.set_xticklabels([])
    
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['bottom'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    
    ax3.set_yticklabels([])
    ax3.set_xticklabels([])
    
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['bottom'].set_visible(False)
    ax3.spines['left'].set_visible(False)
    
    
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    
    
    ax4.spines['right'].set_visible(False)
    ax4.spines['top'].set_visible(False)
    ax4.spines['bottom'].set_visible(False)
    ax4.spines['left'].set_visible(False)
    
    ax5.spines['right'].set_visible(False)
    ax5.spines['top'].set_visible(False)
    ax5.spines['bottom'].set_visible(False)
    ax5.spines['left'].set_visible(False)
    
    
    ax6.spines['right'].set_visible(False)
    ax6.spines['top'].set_visible(False)
    ax6.spines['bottom'].set_visible(False)
    ax6.spines['left'].set_visible(False)
    
    ax7.spines['right'].set_visible(False)
    ax7.spines['top'].set_visible(False)
    ax7.spines['bottom'].set_visible(False)
    ax7.spines['left'].set_visible(False)
   
    ax8.spines['right'].set_visible(False)
    ax8.spines['top'].set_visible(False)
    ax8.spines['bottom'].set_visible(False)
    ax8.spines['left'].set_visible(False)
       
    ax9.spines['right'].set_visible(False)
    ax9.spines['top'].set_visible(False)
    ax9.spines['bottom'].set_visible(False)
    ax9.spines['left'].set_visible(False)
        
    ax10.spines['right'].set_visible(False)
    ax10.spines['top'].set_visible(False)
    ax10.spines['bottom'].set_visible(False)
    ax10.spines['left'].set_visible(False)
    
    
    ax11.spines['right'].set_visible(False)
    ax11.spines['top'].set_visible(False)
    ax11.spines['bottom'].set_visible(False)
    ax11.spines['left'].set_visible(False)
    
    ax12.spines['right'].set_visible(False)
    ax12.spines['top'].set_visible(False)
    ax12.spines['bottom'].set_visible(False)
    ax12.spines['left'].set_visible(False)
    
    ax13.spines['right'].set_visible(False)
    ax13.spines['top'].set_visible(False)
    ax13.spines['bottom'].set_visible(False)
    ax13.spines['left'].set_visible(False)
    
    ax14.spines['right'].set_visible(False)
    ax14.spines['top'].set_visible(False)
    ax14.spines['bottom'].set_visible(False)
    ax14.spines['left'].set_visible(False)
   
    ax4.xaxis.set_major_locator(plt.NullLocator())
    ax4.yaxis.set_major_locator(plt.NullLocator())
    
    ax5.xaxis.set_major_locator(plt.NullLocator())
    ax5.yaxis.set_major_locator(plt.NullLocator())
    
    ax6.xaxis.set_major_locator(plt.NullLocator())
    ax6.yaxis.set_major_locator(plt.NullLocator())
    
    ax7.xaxis.set_major_locator(plt.NullLocator())
    ax7.yaxis.set_major_locator(plt.NullLocator())
    
    ax8.xaxis.set_major_locator(plt.NullLocator())
    ax8.yaxis.set_major_locator(plt.NullLocator())
    
    ax9.xaxis.set_major_locator(plt.NullLocator())
    ax9.yaxis.set_major_locator(plt.NullLocator())
    
    ax10.xaxis.set_major_locator(plt.NullLocator())
    ax10.yaxis.set_major_locator(plt.NullLocator())
    
    ax11.xaxis.set_major_locator(plt.NullLocator())
    ax11.yaxis.set_major_locator(plt.NullLocator())
    
    ax12.xaxis.set_major_locator(plt.NullLocator())
    ax12.yaxis.set_major_locator(plt.NullLocator())
    
    ax13.xaxis.set_major_locator(plt.NullLocator())
    ax13.yaxis.set_major_locator(plt.NullLocator())
        
    ax14.xaxis.set_major_locator(plt.NullLocator())
    ax14.yaxis.set_major_locator(plt.NullLocator())
   
    plotting.plot_epi(image.index_img(img0,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax0, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img1,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax1, cmap='jet', cut_coords = [32]) 
     
    plotting.plot_epi(image.index_img(img2,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax2, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img3,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax3, cmap='jet', cut_coords = [32])   
    
    plotting.plot_epi(image.index_img(img4,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax4, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img5,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax5, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img6,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax6, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img7,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax7, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img8,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax8, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img9,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax9, cmap='jet', cut_coords = [32])   
    
    plotting.plot_epi(image.index_img(img10,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax10, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img11,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax11, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img12,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax12, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img13,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax13, cmap='jet', cut_coords = [32]) 
    
    plotting.plot_epi(image.index_img(img14,1), annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes =ax14, cmap='jet', cut_coords = [32]) 

     
    print ("Figure Path: " + fig_id)
      
    dr.save_fig_pdf_white(fig_id)
    
def draw_sepation_results_single_run(org_img, noisy_img, low_rank_img, sparse_img, gaussian_img, folder, fig_id, title=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    
    fig = plt.figure(frameon = False, figsize=(5,2))
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    grid_rows = 1
    grid_cols = 5
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, top=0.85, hspace=0.01, wspace=0.01)
    
    subtitle = 'Original'
    
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")    
   
    main_ax.set_xlabel('(a)', color=fg_color)
        
    org_image = plotting.plot_epi(org_img, annotate=True, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = main_ax, cmap='jet', cut_coords = [32])     
    
    subtitle = 'Noisy'
    noisy_ax = fig.add_subplot(grid[0, 1])
    noisy_ax.set_facecolor("blue")    
   
    noisy_ax.set_xlabel('(a)', color=fg_color)
    noisy_image = plotting.plot_epi(noisy_img, annotate=True, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = noisy_ax, cmap='jet', cut_coords = [32]) 
    
    low_ax = fig.add_subplot(grid[0, 2])
    low_ax.set_facecolor("blue")    
   
    low_ax.set_xlabel('(b)', color=fg_color)
    
    subtitle = 'Low-Rank'
    low_rank_image = plotting.plot_epi(low_rank_img, annotate=True, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = low_ax, cmap='jet', cut_coords = [32])  
    
    subtitle = 'Sparse'
    sparse_ax = fig.add_subplot(grid[0, 3])
    sparse_ax.set_facecolor("blue")    
   
    sparse_ax.set_xlabel('(c)', color=fg_color)
    
    sparse_image = plotting.plot_epi(sparse_img, annotate=True, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = sparse_ax, cmap='jet', cut_coords = [32])  
    
    subtitle = 'Gaussian'
    gaus_ax = fig.add_subplot(grid[0, 4])
    gaus_ax.set_facecolor("blue")    
   
    gaus_ax.set_xlabel('(c)', color=fg_color)
    
    gauss_image = plotting.plot_epi(gaussian_img, annotate=True, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = gaus_ax, cmap='jet', cut_coords = [32])  
   
  
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
      
    dr.save_fig_pdf(fig_path)
    
def run():
    subject_scan_path = du.get_full_path_subject1()
    print "Subject Path: " + str(subject_scan_path)
    x_true_org = mt.read_image_abs_path(subject_scan_path)
    
    folder = "/work/scratch/alternate_minimation/richian/run_2018-12-05_22_38_47/d4/richian/images"
    fig_id = "x_low_rank_" + "snr_" + str(30) + "_" + str("Richian1")
    
    title =  "Noise free fMRI Images"
    #draw_image2(image.index_img(x_true_org,1), folder, fig_id, title)
    
    fig_id = "decomposition_" + "snr_" + str(30) + "_" + str("richian")
    title =  "Low-Rank and Sparse Decomposition. SNR = " + str(30) +  "; Noise Type: " + "Richian1"
    #draw_sepation_results_single_run(image.index_img(x_true_org,1), image.index_img(x_true_org,1), image.index_img(x_true_org,1), image.index_img(x_true_org,1), image.index_img(x_true_org,1), folder, fig_id, title)
    
    fig_id = "decomposition_" + "snr_" + str(30) + "_" + str("richian") + "_no_title"
    
    title = None
    #draw_sepation_results_single_run(image.index_img(x_true_org,1), image.index_img(x_true_org,1), image.index_img(x_true_org,1), image.index_img(x_true_org,1), image.index_img(x_true_org,1), folder, fig_id, title)
    
def get_images_richian():
        
    row0 = []
    
    #folder = "/work/scratch/alternate_minimation/richian/run_2018-12-06_16_27_43/d4/richian/scans/final/mr/4000"
    
    folder = "/work/scratch/alternate_minimation/richian/run_2018-12-06_16_43_28/d4/richian/scans/final/mr/3500"
    
    x_true = "x_true_img_3500.nii"
    x_noisy = "x_noisy_img_3500.nii"
    x_low_rank = "x_low_rank_hat_img_3500.nii"
    x_sparse = "x_sparse_hat_img_3500.nii"
    x_guass = "x_guass_hat_img_3500.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row0.append(x_true_path)
    row0.append(x_noisy_path)
    row0.append(x_low_rank_path)
    row0.append(x_sparse_path)
    row0.append(x_guass_path)
    
    
    row1 = []
    
    folder = "/work/scratch/alternate_minimation/richian/run_2018-12-06_16_59_16/d4/richian/scans/final/mr/2500"
    
    x_true = "x_true_img_2500.nii"
    x_noisy = "x_noisy_img_2500.nii"
    x_low_rank = "x_low_rank_hat_img_2500.nii"
    x_sparse = "x_sparse_hat_img_2500.nii"
    x_guass = "x_guass_hat_img_2500.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row1.append(x_true_path)
    row1.append(x_noisy_path)
    row1.append(x_low_rank_path)
    row1.append(x_sparse_path)
    row1.append(x_guass_path)
   
   
    row2 = []
    
    folder = "/work/scratch/alternate_minimation/richian/run_2018-12-06_17_46_50/d4/richian/scans/final/mr/1000"
    
    x_true = "x_true_img_1000.nii"
    x_noisy = "x_noisy_img_1000.nii"
    x_low_rank = "x_low_rank_hat_img_1000.nii"
    x_sparse = "x_sparse_hat_img_1000.nii"
    x_guass = "x_guass_hat_img_1000.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row2.append(x_true_path)
    row2.append(x_noisy_path)
    row2.append(x_low_rank_path)
    row2.append(x_sparse_path)
    row2.append(x_guass_path)
    
    images = {0:row0, 1:row1, 2:row2}
    
    return images

def draw_richian_multiple():
    
    images = get_images_richian()
    
    folder = "/work/project/cmsc655/figures/"
    fig_id = "richian_per_snr"
    file_path = os.path.join(folder, fig_id)
    
    width = tex_width
    draw_composition_per_multiple_corruptions(images, file_path, width, title=None)
    
def get_images_raleign():
        
    row0 = []
    
    #folder = "/work/scratch/alternate_minimation/richian/run_2018-12-06_16_27_43/d4/richian/scans/final/mr/4000"
    
    folder = "/work/scratch/alternate_minimation/rayleigh1/run_2018-12-07_07_18_11/d4/gaussian/scans/final/mr/3500"
    
    x_true = "x_true_img_3500.nii"
    x_noisy = "x_noisy_img_3500.nii"
    x_low_rank = "x_low_rank_hat_img_3500.nii"
    x_sparse = "x_sparse_hat_img_3500.nii"
    x_guass = "x_guass_hat_img_3500.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row0.append(x_true_path)
    row0.append(x_noisy_path)
    row0.append(x_low_rank_path)
    row0.append(x_sparse_path)
    row0.append(x_guass_path)
    
    
    row1 = []
    
    folder = "/work/scratch/alternate_minimation/rayleigh1/run_2018-12-07_07_35_32/d4/gaussian/scans/final/mr/2500"
    
    x_true = "x_true_img_2500.nii"
    x_noisy = "x_noisy_img_2500.nii"
    x_low_rank = "x_low_rank_hat_img_2500.nii"
    x_sparse = "x_sparse_hat_img_2500.nii"
    x_guass = "x_guass_hat_img_2500.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row1.append(x_true_path)
    row1.append(x_noisy_path)
    row1.append(x_low_rank_path)
    row1.append(x_sparse_path)
    row1.append(x_guass_path)
   
   
    row2 = []
    
    folder = "/work/scratch/alternate_minimation/rayleigh1/run_2018-12-07_08_27_42/d4/gaussian/scans/final/mr/1000"
    
    x_true = "x_true_img_1000.nii"
    x_noisy = "x_noisy_img_1000.nii"
    x_low_rank = "x_low_rank_hat_img_1000.nii"
    x_sparse = "x_sparse_hat_img_1000.nii"
    x_guass = "x_guass_hat_img_1000.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row2.append(x_true_path)
    row2.append(x_noisy_path)
    row2.append(x_low_rank_path)
    row2.append(x_sparse_path)
    row2.append(x_guass_path)
    
    images = {0:row0, 1:row1, 2:row2}
    
    return images

def get_images_gaussian():
        
    row0 = []
    
    #folder = "/work/scratch/alternate_minimation/richian/run_2018-12-06_16_27_43/d4/richian/scans/final/mr/4000"
    
    folder = "/work/scratch/alternate_minimation/gaussian1/run_2018-12-07_05_18_36/d4/gaussian/scans/final/mr/3500"
    
    x_true = "x_true_img_3500.nii"
    x_noisy = "x_noisy_img_3500.nii"
    x_low_rank = "x_low_rank_hat_img_3500.nii"
    x_sparse = "x_sparse_hat_img_3500.nii"
    x_guass = "x_guass_hat_img_3500.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row0.append(x_true_path)
    row0.append(x_noisy_path)
    row0.append(x_low_rank_path)
    row0.append(x_sparse_path)
    row0.append(x_guass_path)
    
    
    row1 = []
    
    folder = "/work/scratch/alternate_minimation/gaussian1/run_2018-12-07_05_35_19/d4/gaussian/scans/final/mr/2500"
    
    x_true = "x_true_img_2500.nii"
    x_noisy = "x_noisy_img_2500.nii"
    x_low_rank = "x_low_rank_hat_img_2500.nii"
    x_sparse = "x_sparse_hat_img_2500.nii"
    x_guass = "x_guass_hat_img_2500.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row1.append(x_true_path)
    row1.append(x_noisy_path)
    row1.append(x_low_rank_path)
    row1.append(x_sparse_path)
    row1.append(x_guass_path)
   
   
    row2 = []
    
    folder = "/work/scratch/alternate_minimation/gaussian1/run_2018-12-07_06_26_05/d4/gaussian/scans/final/mr/1000"
    
    x_true = "x_true_img_1000.nii"
    x_noisy = "x_noisy_img_1000.nii"
    x_low_rank = "x_low_rank_hat_img_1000.nii"
    x_sparse = "x_sparse_hat_img_1000.nii"
    x_guass = "x_guass_hat_img_1000.nii"
    
    x_true_path = os.path.join(folder, x_true)
    x_noisy_path = os.path.join(folder, x_noisy)
    x_low_rank_path = os.path.join(folder, x_low_rank)
    x_sparse_path = os.path.join(folder, x_sparse)
    x_guass_path = os.path.join(folder, x_guass)
    
    row2.append(x_true_path)
    row2.append(x_noisy_path)
    row2.append(x_low_rank_path)
    row2.append(x_sparse_path)
    row2.append(x_guass_path)
    
    images = {0:row0, 1:row1, 2:row2}
    
    return images

def draw_raleign():
    
    images = get_images_raleign()
    
    folder = "/work/project/cmsc655/figures/"
    fig_id = "raleign_per_snr"
    file_path = os.path.join(folder, fig_id)
    
    width = tex_width
    draw_composition_per_multiple_corruptions(images, file_path, width, title=None)
    
def draw_guassian():
    
    images =  get_images_gaussian()
    
    folder = "/work/project/cmsc655/figures/"
    fig_id = "guassian_per_snr"
    file_path = os.path.join(folder, fig_id)
    
    width = tex_width
    draw_composition_per_multiple_corruptions(images, file_path, width, title=None)
    
def draw_fmri_plot():
    
    folder = "/work/project/cmsc655/figures/"
    subject_scan_path = du.get_full_path_subject1()
    ground_truth_img = mt.read_image_abs_path(subject_scan_path)
    ground_truth = np.array(ground_truth_img.get_data()).astype('float32')
    
    norm_ground_x_init = np.linalg.norm(ground_truth)
    x_init = ground_truth * (1./norm_ground_x_init)
    x_init_img = mt.reconstruct_image_affine(ground_truth_img, x_init)
       
    fig_id = "fmri_plot"
    file_path = os.path.join(folder,fig_id)
    
    mask_file="epi_mask.nii"

    dvars = nac.ComputeDVARS()
    dvars.inputs.in_file = subject_scan_path
    dvars.inputs.in_mask = 'epi_mask.nii'
    dvars.inputs.save_all = True
    dvars.inputs.save_plot = True
    dvars.inputs.figformat = 'png'
    dvars.run()
    
    dataframe = pd.DataFrame({
                'DVARS': [np.nan] + np.loadtxt(
                'dvars.tsv', skiprows=1, usecols=[1]).tolist(),
        })
    
    fig = fm.fMRIPlot(
            subject_scan_path,
            mask_file=mask_file,
            spikes_files=['spikes.tsv'],
            seg_file=None,
            tr=125,
            data=dataframe[['DVARS']],
        ).plot()
    
    dr.save_fig_pdf_white(file_path)

    
if __name__ == "__main__":
    #run()
    #draw_richian_multiple()
    #draw_fmri_plot()
    #draw_raleign()
    draw_guassian()