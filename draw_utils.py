import nilearn

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

'''
Saves plots at the desired location 
'''
def save_fig(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".png")
        path_tiff = os.path.join(fig_id + ".tiff")
        path_pdf = os.path.join(fig_id + ".pdf")
        print("Saving figure", path_pdf)
        print("Called from draw utils")
        plt.savefig(path_pdf, format='pdf', facecolor='k', edgecolor='k', dpi=1000)
        plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
        
def save_fig_png(fig_id, tight_layout=True):
        path = os.path.join(fig_id + ".png")
        print("Saving figure", path)
        print("Called from draw utils")
        plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        plt.close()
        
def save_fig_pdf(fig_id, tight_layout=True):
        path_pdf = os.path.join(fig_id + ".pdf")
        print("Saving figure", path_pdf)
        print("Called from draw utils")
        plt.savefig(path_pdf, format='pdf', facecolor='k', edgecolor='k', dpi=1000, bbox_inches='tight')
        plt.close()
        
def save_fig_pdf_white(fig_id, tight_layout=True):
        path_pdf = os.path.join(fig_id + ".pdf")
        print("Saving figure", path_pdf)
        print("Called from draw utils")
        plt.savefig(path_pdf, format='pdf', facecolor='w', edgecolor='w', dpi=1000, bbox_inches='tight')
        plt.close()
    

def save_fig_abs_path(fig_id, tight_layout=True):
    path = os.path.join(fig_id + ".png")
    print("Saving figure", path)
    plt.savefig(path, format='png', facecolor='k', edgecolor='k', dpi=300)
        

def save_csv(df, file_path, dataset_id):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)
    
def save_csv_by_path(df, file_path, dataset_id):
    path = os.path.join(file_path, dataset_id + ".csv")
    print("Saving dataset", path)
    df.to_csv(path)

def legend_outside(ncol, extra_text):
    handles, labels = plt.gca().get_legend_handles_labels()
    lgd = plt.legend(handles, labels, loc='upper center', ncol=ncol, bbox_to_anchor=(-0.15, -0.3))
    import matplotlib.offsetbox as offsetbox 
    txt=offsetbox.TextArea(extra_text, textprops=dict(size=lgd.get_texts()[0].get_fontsize()))
    box = lgd._legend_box
    box.get_children().append(txt) 
    box.set_figure(box.figure)
    return lgd    
    
def draw_image(img, folder, fig_id, title=None):
    
    fig = plt.figure(frameon = False, figsize=(5,5))
        
    scan_image = plotting.plot_epi(img, annotate=True, draw_cross=False, bg_img=None,black_bg=True, cmap='jet', cut_coords=None)     
    
    fg_color = 'white'
    bg_color = 'black'
    
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
      
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
        
    save_fig_pdf(fig_path)
    
def draw_x_true(img, folder, fig_id, title=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    fig = plt.figure(frameon = False, figsize=(5,2.5))
            
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=10)
        
    y_image = plotting.plot_epi(img, annotate=True, draw_cross=False, bg_img=None,black_bg=True, figure= fig, cmap='jet', cut_coords = None)     
      
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
    save_fig_pdf(fig_path)

def draw_decomposition_results_run(org_img, noisy_img, low_rank_img, sparse_img, gaussian_img, folder, fig_id, title=None):
    
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
        
    org_image = plotting.plot_epi(org_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = main_ax, cmap='jet', cut_coords = [32])     
    
    subtitle = 'Noisy'
    noisy_ax = fig.add_subplot(grid[0, 1])
    noisy_ax.set_facecolor("blue")    
   
    noisy_ax.set_xlabel('(a)', color=fg_color)
    noisy_image = plotting.plot_epi(noisy_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = noisy_ax, cmap='jet', cut_coords = [32]) 
    
    low_ax = fig.add_subplot(grid[0, 2])
    low_ax.set_facecolor("blue")    
   
    low_ax.set_xlabel('(b)', color=fg_color)
    
    subtitle = 'Low-Rank'
    low_rank_image = plotting.plot_epi(low_rank_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = low_ax, cmap='jet', cut_coords = [32])  
    
    subtitle = 'Sparse'
    sparse_ax = fig.add_subplot(grid[0, 3])
    sparse_ax.set_facecolor("blue")    
   
    sparse_ax.set_xlabel('(c)', color=fg_color)
    
    sparse_image = plotting.plot_epi(sparse_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = sparse_ax, cmap='jet', cut_coords = [32])  
    
    subtitle = 'Gaussian'
    gaus_ax = fig.add_subplot(grid[0, 4])
    gaus_ax.set_facecolor("blue")    
   
    gaus_ax.set_xlabel('(c)', color=fg_color)
    
    gauss_image = plotting.plot_epi(gaussian_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = gaus_ax, cmap='jet', cut_coords = [32])  
   
  
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
      
    save_fig_pdf(fig_path)
        
def draw_decomposition_results(org_img, noisy_img, low_rank_img, sparse_img, gaussian_img, folder, fig_id, title=None):
    
    fg_color = 'white'
    bg_color = 'black'
    
    plt.clf()
    fig = plt.figure(frameon = False, figsize=(5,2))
          
    if title:
        fig.suptitle(title, color=fg_color, fontweight='normal', fontsize=8, y=0.85)
        
    grid_rows = 1
    grid_cols = 4
        
    grid = gridspec.GridSpec(grid_rows,grid_cols, top=0.85, hspace=0.01, wspace=0.01)
     
    
    subtitle = 'Noisy'
    noisy_ax = fig.add_subplot(grid[0, 0])
    noisy_ax.set_facecolor("blue")    
   
    noisy_ax.set_xlabel('(a)', color=fg_color)
    noisy_image = plotting.plot_epi(noisy_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = noisy_ax, cmap='jet', cut_coords = [32]) 
    
    low_ax = fig.add_subplot(grid[0, 1])
    low_ax.set_facecolor("blue")    
   
    low_ax.set_xlabel('(b)', color=fg_color)
    
    subtitle = 'Low-Rank'
    low_rank_image = plotting.plot_epi(low_rank_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = low_ax, cmap='jet', cut_coords = [32])  
    
    subtitle = 'Sparse'
    sparse_ax = fig.add_subplot(grid[0, 2])
    sparse_ax.set_facecolor("blue")    
   
    sparse_ax.set_xlabel('(c)', color=fg_color)
    
    sparse_image = plotting.plot_epi(sparse_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = sparse_ax, cmap='jet', cut_coords = [32])  
    
    subtitle = 'Gaussian'
    gaus_ax = fig.add_subplot(grid[0, 3])
    gaus_ax.set_facecolor("blue")    
   
    gaus_ax.set_xlabel('(c)', color=fg_color)
    
    gauss_image = plotting.plot_epi(gaussian_img, annotate=False, draw_cross=False, bg_img=None,black_bg=True,
                                   figure= fig, display_mode='z', axes = gaus_ax, cmap='jet', cut_coords = [32])  
   
  
    fig_path = os.path.join(folder, fig_id)
    
    print ("Figure Path: " + fig_path)
      
    save_fig_pdf(fig_path)
    