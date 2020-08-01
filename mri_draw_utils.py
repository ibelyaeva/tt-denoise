import nilearn
import texfig

from nilearn import image
from nilearn import plotting
import os
import matplotlib.pyplot as plt
from math import ceil
import matplotlib.gridspec as gridspec
import matplotlib
from nilearn.plotting import find_xyz_cut_coords

import metric_util as mc
import math
import math_format as mf

PROJECT_DIR  = "/work/pl/sch/analysis/scripts"
PROJECT_ROOT_DIR = "."
DATA_DIR = "data"
CSV_DATA = "csv_data"
FIGURES = "figures"

#SMALL_SIZE = 8
#matplotlib.rc('font', size=SMALL_SIZE)
#matplotlib.rc('axes', titlesize=SMALL_SIZE)
#matplotlib.rc('figure', titlesize=SMALL_SIZE)

'''
Saves plots at the desired location 
'''
def save_fig(fig_id, tight_layout=True):
        path = os.path.join(PROJECT_DIR, FIGURES, fig_id + ".png")
        path_eps = os.path.join(PROJECT_DIR, FIGURES, fig_id + ".eps")
        path_tiff = os.path.join(fig_id + ".tiff")
        path_pdf = os.path.join(PROJECT_DIR, FIGURES, fig_id + ".pdf")
        print("Saving figure", path_tiff)
        print("Called from mrd")
        #plt.savefig(path_eps, format='eps', facecolor='k', edgecolor='k', dpi=600)
        plt.savefig(path_tiff, format='tiff', facecolor='k', edgecolor='k', dpi=1000)
        plt.savefig(path_pdf, format='pdf', facecolor='k', edgecolor='k', dpi=1000)
        plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
        
def save_fig_png(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".png")
        print("Saving figure", path)
        print("Called from mrd")
        plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
        
def save_fig_pdf(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".pdf")
        print("Saving figure", path)
        print("Called from mrd")
        plt.savefig(path, format='pdf', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
        
def save_report_fig(fig_id, tight_layout=True):
        path = os.path.join(PROJECT_DIR, FIGURES, fig_id + ".png")
        print("Saving figure", path)
        plt.savefig(path, format='png', dpi=300)

def save_fig_abs_path(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", path)
    plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        
def save_csv(df, dataset_id):
    path = os.path.join(PROJECT_DIR, CSV_DATA, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)

def save_csv_by_path(df, file_path, dataset_id):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)
    
def draw_original_vs_reconstructed(img_ref, x_true, x_hat, x_miss, plot_title, observed_ratio, coord=None):
    grid_rows = 4
    grid_cols = 1
    
    grid = gridspec.GridSpec(grid_rows,grid_cols)

    figure = plt.figure(frameon = False, figsize=(10,10))
    figure.set_size_inches(7, 7)
    ax = plt.Axes(figure, [0., 0., 1., 1.], )
    ax.set_axis_off()
    figure.add_axes(ax)
    
    grid.update(wspace=0.001, hspace=0.15)
    grid_range = grid_rows*grid_cols

    if plot_title:
        figure.suptitle(plot_title, fontsize=10)
        
    col_rows = range(grid_rows)
    
    counter = 0
        
    missing_ratio = (1 - observed_ratio)*100
    percent_error = mc.relative_error(x_hat,x_true)
    reconstructed_image = mc.reconstruct_image_affine(img_ref, x_hat)
    error_formatted = float("{0:.3f}".format(percent_error))
    
    #Original Image
    ax = plt.subplot(grid[counter])
    
                
    true_image = plotting.plot_epi(img_ref, bg_img=None,black_bg=False, axes = ax, cmap='jet', cut_coords=coord)
    ax.set_title('Original Image', fontsize=9)
       
    figure.add_subplot(ax)
    counter = counter +1
                
    #Missing Tensor Image
    ax = plt.subplot(grid[counter])
    masked_missing_image = mc.reconstruct_image_affine(img_ref, x_miss)
    ax.set_title('Missing Tensor Image - ' + " " + str("Missed Data %:") + str(missing_ratio))
    
    mask_mage = plotting.plot_epi(masked_missing_image, bg_img=None,black_bg=False, axes = ax, cmap='jet', cut_coords=coord)
    figure.add_subplot(ax)
    counter = counter +1
        
    ax = plt.subplot(grid[counter])
    ax.set_title('Reconstructed Image - ' + " Score (Relative Error):" + str(error_formatted))

    
    est_image = plotting.plot_epi(reconstructed_image, bg_img=None,black_bg=False, axes = ax, cmap='jet', cut_coords=coord)
    counter = counter +1
    figure.add_subplot(ax)
        
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_aspect('equal')
                   
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
        ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(9)
            
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_id =  "missing_ratio_" + str(missing_ratio) + ".png"
    save_fig(fig_id)

def floored_percentage(val, digits):
        val *= 10 ** (digits + 2)
        return '{1:.{0}f}%'.format(digits, math.floor(val) / 10 ** digits)

def formatted_percentage(value, digits):
    format_str = "{:." + str(digits) + "%}"
    
    return format_str.format(value) 
   
def draw_original_vs_reconstructed2(img_ref, x_true, x_hat, x_miss, plot_title, observed_ratio, coord=None, folder=None):
    
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=11)
    
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('Original fMRI Scan - Slice 0', color=fg_color, fontweight='normal', fontsize=10)  
    main_ax.set_aspect('equal')      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error = mc.relative_error(x_hat,x_true)
    masked_missing_image = mc.reconstruct_image_affine(img_ref, x_miss)
    reconstructed_image = mc.reconstruct_image_affine(img_ref, x_hat)
    relative_error_str = "{0:.5f}".format(relative_error)  
                
    true_image = plotting.plot_epi(img_ref, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_title('Corrupted fMRI Scan - ' + " " + str("Missed Ratio: ") + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=10)
    miss_image = plotting.plot_epi(masked_missing_image, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_title('Completed fMRI Scan - ' + " " + str("Relative Error: ") + str(relative_error_str), color=fg_color, fontweight='normal', fontsize=10)
    
    recovered_image = plotting.plot_epi(reconstructed_image, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
   
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
    save_fig(fig_id)
    
    
def draw_original_vs_reconstructed3(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord=None, folder=None):
        
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    
    #fig = plt.figure(frameon = False, figsize=(10,10))
    #fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=10)
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('Original fMRI brain volume in three projections at first time point.', color=fg_color, fontweight='normal', fontsize=8)  
    main_ax.set_aspect('equal')
      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_xlabel('(b)', color=bg_color)
    
    miss_ax.set_title('Corrupted fMRI brain volume. ' + " " + str("Missing Voxels Ratio: ")
                       + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=8)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_xlabel('(c)', color=bg_color)
    
    recov_ax.set_title('Completed fMRI brain volume. ' + " " + str("RSE: ") + relative_error_str, color=fg_color, fontweight='normal', fontsize=8)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    fig_id = fig_id[:-1]
    save_fig(fig_id)
    
    draw_original_vs_reconstructed3_pub(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord, folder)
    draw_original_vs_reconstructed3_pub_black(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord, folder)
    
def draw_original_vs_reconstructed_rim(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord=None, folder=None):
        
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    
    #fig = plt.figure(frameon = False, figsize=(10,10))
    #fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=10)
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('Original fMRI brain volume in three projections at first time point.', color=fg_color, fontweight='normal', fontsize=8)  
    main_ax.set_aspect('equal')
      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_xlabel('(b)', color=bg_color)
    
    miss_ax.set_title('Corrupted fMRI brain volume. ' + " " + str("Missing Voxels Ratio: ")
                       + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=8)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_xlabel('(c)', color=bg_color)
    
    recov_ax.set_title('Completed fMRI brain volume. ' + " " + str("RSE: ") + relative_error_str, color=fg_color, fontweight='normal', fontsize=8)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    fig_id = fig_id[:-1]
    save_fig_png(fig_id) 
    
def draw_original_vs_reconstructed_rim_z_score(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, tcs, tcs_z_score, z_score, coord=None, folder=None, iteration=-1, time=-1):
        
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    
    #fig = plt.figure(frameon = False, figsize=(10,10))
    #fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=10)
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    
    subtitle = 'Original fMRI brain volume in three projections.'
    
    if time >-1: 
        subtitle = 'Original fMRI brain volume in three projections. Timepoint: ' + str(time + 1)
        
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")    
   
    main_ax.set_title(subtitle , color=fg_color, fontweight='normal', fontsize=8)  
    main_ax.set_aspect('equal')
      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)
    print ("Missing Ratio Str:" + missing_ratio_str)                          
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
    
    tsc_str = mf.format_number(tcs, fmt='%1.2e')
    tsc_z_score_str = mf.format_number(tcs_z_score, fmt='%1.2e')
    z_score_str = str(z_score)
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_xlabel('(b)', color=bg_color)
    
    miss_ax.set_title('Corrupted fMRI brain volume. ' + " " + str("Missing Ratio: ")
                       + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=8)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_xlabel('(c)', color=bg_color)
    
    recov_ax.set_title('Completed. ' + " " + str("TCS: ") + tsc_str + " TCS(Z_Score >" + z_score_str + "): "  + tsc_z_score_str, color=fg_color, fontweight='normal', fontsize=8)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    print ("Iteration: " + str(iteration))
    if iteration >=0:
        fig_id = fig_id[:-1] + '_' + str(iteration)
    else:
        fig_id = fig_id[:-1]
        
    if time >=0:
        fig_id = fig_id + '_timepoint_' + str(time)  

    save_fig_pdf(fig_id) 
    
    
def draw_original_vs_reconstructed_rim_tex(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord=None, folder=None):
        
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    
    #fig = plt.figure(frameon = False, figsize=(10,10))
    #fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=10)
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('Original fMRI brain volume in three projections at first time point.', color=fg_color, fontweight='normal', fontsize=8)  
    main_ax.set_aspect('equal')
      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_xlabel('(b)', color=bg_color)
    
    miss_ax.set_title('Corrupted fMRI brain volume. ' + " " + str("Missing Voxels Ratio: ")
                       + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=8)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_xlabel('(c)', color=bg_color)
    
    recov_ax.set_title('Completed fMRI brain volume. ' + " " + str("RSE: ") + relative_error_str, color=fg_color, fontweight='normal', fontsize=8)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    fig_id = fig_id[:-1]
    texfig.savefig(fig_id, edgecolor='k', dpi=1000) 
    
def draw_original_vs_reconstructed_rim_z(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, z_score, coord=None, folder=None):
        
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    
    #fig = plt.figure(frameon = False, figsize=(10,10))
    #fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if plot_title:
        fig.suptitle(plot_title, color=fg_color, fontweight='normal', fontsize=10)
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('Original fMRI brain volume in three projections at first time point.', color=fg_color, fontweight='normal', fontsize=8)  
    main_ax.set_aspect('equal')
      
   
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_xlabel('(b)', color=bg_color)
    
    miss_ax.set_title('Corrupted fMRI brain volume. ' + " " + "Z-Score: " + str(z_score), color=fg_color, fontweight='normal', fontsize=8)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_xlabel('(c)', color=bg_color)
    
    recov_ax.set_title('Completed fMRI brain volume. ' + " " + str("TCS: ") + relative_error_str, color=fg_color, fontweight='normal', fontsize=8)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
        
    fig_id = fig_id[:-1]
    save_fig_png(fig_id)

    
def draw_original_vs_reconstructed3_pub(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord=None, folder=None):
    
    fig = texfig.figure(frameon = False)
    fig.set_frameon(False)

    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.4, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('$(a)$', color=bg_color, fontweight='normal', fontsize=6)  
    main_ax.set_aspect('equal')
     
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
                
    true_image = plotting.plot_epi(x_true_img, annotate=False,draw_cross=False,  bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_title('$(b)$', color=bg_color, fontweight='normal', fontsize=6)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    #diff_img = image.math_img("img1 - img2",
     #                 img1=x_true_img, img2= x_miss_img)
    
    #diff_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    #diff_image = plotting.plot_epi(diff_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True, figure= fig, axes = diff_ax, cmap='jet', cut_coords=coord)       
    #diff_ax.set_title('$(c)$', color=bg_color, fontweight='normal', fontsize=6)
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    
    recov_ax.set_title('$(c)$', color=bg_color, fontweight='normal', fontsize=6)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
    
    fig_id = fig_id[:-1]
    
    texfig.savefig(fig_id, edgecolor='k', dpi=1000)

def draw_original_vs_reconstructed3_pub_black(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord=None, folder=None):
    
    fig = texfig.figure(frameon = False)
    fig.set_frameon(False)

    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
    
    if not coord:
        coord = find_xyz_cut_coords(x_true_img)
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.4, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('$(a)$', color=fg_color, fontweight='normal', fontsize=6)  
    main_ax.set_aspect('equal')
     
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error) 
    relative_error_str = mf.format_number(relative_error, fmt='%1.2e')
                
    true_image = plotting.plot_epi(x_true_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    miss_ax.set_title('$(b)$', color=fg_color, fontweight='normal', fontsize=6)
    
    miss_image = plotting.plot_epi(x_miss_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    
    recov_ax.set_title('$(c)$', color=fg_color, fontweight='normal', fontsize=6)
    
    recovered_image = plotting.plot_epi(x_hat_img, annotate=False, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
    
    if folder:
        fig_id =  str(folder) + "/" + "missing_ratio_" + str(missing_ratio_str)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
    
    fig_id = fig_id[:-1]
    
    fig_id_black_bg = fig_id + "_black_bg"
    
    texfig.savefig(fig_id_black_bg, facecolor= 'k', edgecolor='k', dpi=1000)
   
    
def draw_original_vs_reconstructed4(x_true_img, x_hat_img, x_miss_img, plot_title, relative_error, observed_ratio, coord=None, folder=None, radius = None):
    
    fig = plt.figure(frameon = False, figsize=(10,10))
    fig.set_size_inches(7, 7)
    grid_rows = 3
    grid_cols = 1
    
    fg_color = 'white'
    bg_color = 'black'
      
    
    missing_ratio = (1.0 - observed_ratio)
    missing_ratio_str = formatted_percentage(missing_ratio, 2)                        
 
    relative_error_str = "{0:.5f}".format(relative_error)
    
    if plot_title:
        plt_title = plot_title + str("Missed Ratio: ") + str(missing_ratio_str)
        fig.suptitle(plt_title, color=fg_color, fontweight='normal', fontsize=11)
    
    grid = gridspec.GridSpec(grid_rows,grid_cols, hspace=0.2, wspace=0.2)
    main_ax = fig.add_subplot(grid[0, 0])
    main_ax.set_facecolor("blue")
    main_ax.set_title('Original fMRI Scan - Slice 1', color=fg_color, fontweight='normal', fontsize=10)  
    main_ax.set_aspect('equal')      
  
    
    if coord:
        x0, y0, z0 = coord
        
    if radius:
        x_r, y_r, z_r = radius
    
    miss_title = None
    if coord and radius:
        miss_title = "3D Ellipsoid Mask" + " Center: " + str(x0) + ", "+ str(y0) + ", "+  str(z0)+ "; Radius: " + str(x_r) + ", "+ str(y_r) + ", "+  str(z_r)
        
    true_image = plotting.plot_epi(x_true_img, bg_img=None,black_bg=True, figure= fig, axes = main_ax, cmap='jet', cut_coords=coord)     
    
    miss_ax = fig.add_subplot(grid[1, 0], sharex=main_ax)
    
    if miss_title:
        miss_ax.set_title(miss_title, color=fg_color, fontweight='normal', fontsize=10)
    else:
        miss_ax.set_title('Corrupted fMRI Scan - ' + " " + str("Missed Ratio: ") + str(missing_ratio_str), color=fg_color, fontweight='normal', fontsize=10)
        
    miss_image = plotting.plot_epi(x_miss_img, bg_img=None,black_bg=True, figure= fig, axes = miss_ax, cmap='jet', cut_coords=coord)  
    miss_image.add_contours(x_miss_img, levels=[0.1, 0.5, 0.7, 0.9], filled=True, colors='b')
    coords = [(x0, y0, z0)]
    miss_image.add_markers(coords, marker_color='b', marker_size=50)
    
    recov_ax = fig.add_subplot(grid[2, 0], sharex=main_ax)
    recov_ax.set_title('Completed fMRI Scan - ' + " " + str("Relative Error: ") + str(relative_error_str), color=fg_color, fontweight='normal', fontsize=10)
    
    recovered_image = plotting.plot_epi(x_hat_img, bg_img=None,black_bg=True, figure= fig, axes = recov_ax, cmap='jet', cut_coords=coord)       
   
    if folder and radius:
        fig_id =  str(folder) + "/" + "_"+ str(x_r) + "_"+ str(y_r) + "_"+  str(z_r)
    else:
        fig_id =  "_missing_ratio_" + str(missing_ratio_str)
    save_fig_abs_path(fig_id)
