import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import t3f
import metric_util as mt
import data_util as du
from tensorflow.python.util import nest
import pandas as pd
from nilearn.image import math_img
import nibabel as nib
import draw_utils as dr
import os
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
import math
import fmri_plot_decomposition_plot as fm
from nipype.algorithms import confounds as nac
import os.path as op
import spike_detection as sp
from scipy.stats import zscore

def frmi_plot(in_file, low_rank_file, sparse_file, gauss_file, spike_folder, file_path, legend=False, tr=None, stddev=False, tsnr=False, img_slice=None):
    
    x_img = mt.read_image_abs_path(in_file)
    data = np.array(x_img.get_data())
    z_scored_data = np.nan_to_num(zscore( data, axis=-1))
    fname, ext = op.splitext(op.basename(in_file))
    
    
    denoised_fname, ext = op.splitext(op.basename(low_rank_file))
    
    n_spikes, out_spikes, out_fft, spikes_list = sp.slice_wise_fft(in_file, spike_folder, spike_thres=4.)
    dvar_file_name = fname + "_dvars.tsv"
    
    denoised_dvar_file_name = denoised_fname + "_dvars.tsv"
    
    print("Denoised Dvar File Name:" + denoised_dvar_file_name)
    
    mask_file="epi_mask.nii"
    
    if n_spikes == 0:
        spikes_files = None
    else:
        spikes_files=[out_spikes]    
        

    dvars = nac.ComputeDVARS()
    dvars.inputs.in_file = in_file
    dvars.inputs.in_mask = mask_file
    dvars.inputs.save_all = True
    dvars.inputs.save_plot = True
    dvars.inputs.figformat = 'pdf'
    dvars.run()
    
    dvars1 = nac.ComputeDVARS()
    dvars1.inputs.in_file = low_rank_file
    dvars1.inputs.in_mask = mask_file
    dvars1.inputs.save_all = True
    dvars1.inputs.save_plot = True
    dvars1.inputs.figformat = 'pdf'
    dvars1.run()
    
    out_suffix = fname + ".nii.gz"
    tsnr_file = "tsnr_" + out_suffix
    mean_file = "mean_" + out_suffix
    stddev_file = "stddev_" + out_suffix
    tsnr = nac.TSNR()
    tsnr.inputs.in_file = in_file
    tsnr.inputs.tsnr_file = tsnr_file
    tsnr.inputs.mean_file = mean_file
    tsnr.inputs.stddev_file = stddev_file
    tsnr.run()
    
    dataframe = pd.DataFrame({
                'DVARS': [np.nan] + np.loadtxt(
                dvar_file_name, skiprows=1, usecols=[1]).tolist(),
                'Denoised DVARS': [np.nan] + np.loadtxt(
                denoised_dvar_file_name, skiprows=1, usecols=[1]).tolist()
        })
        
    if tr and tsnr and stddev:
        fm_plot = fm.fMRIPlotDecompositionPlot(
                in_file,
                low_rank_file,
                sparse_file,
                gauss_file,
                mask_file=mask_file,
                spikes_files=spikes_files,
                seg_file=None,
                tr=tr,
                data=dataframe[['DVARS']],
                legend=legend,
                img_slice = img_slice,
                tsnr = tsnr_file,
                stddev =  stddev_file
                )
    elif tr:
        fm_plot = fm.fMRIPlotDecompositionPlot(
                in_file,
                low_rank_file,
                sparse_file,
                gauss_file,
                mask_file=mask_file,
                spikes_files=spikes_files,
                seg_file=None,
                tr=tr,
                data=dataframe[['DVARS']],
                legend=legend
                )
    else:
        fm_plot = fm.fMRIPlotDecompositionPlot(
                in_file,
                low_rank_file,
                sparse_file,
                gauss_file,
                mask_file=mask_file,
                spikes_files=spikes_files,
                seg_file=None,
                data=dataframe[['DVARS']],
                legend=legend,
                img_slice = img_slice
                )
    fm_plot.plot()
    
    dr.save_fig_pdf_white(file_path)
    
if __name__ == "__main__":

    in_file = subject_scan_path = du.get_full_path_subject1()
    fname, ext = op.splitext(op.basename(in_file))
    folder = "/work/project/cmsc655/figures/"
    spike_folder = "/work/project/cmsc655/subject_artifacts"
    
    fig_id = "godec_frmi_summary_plot_signal_slice_decomposition_richian_25db" + fname
    fig_path = os.path.join(folder,fig_id)
    
    #image_folder_path = "/work/scratch/alternate_minimation/richian1/run_2018-12-07_03_54_42/d4/richian/scans/final/mr/2000"
    
    image_folder_path = "/work/scratch/alternate_minimation/richian1/run_2018-12-07_03_37_55//d4/richian/scans/final/mr/2500"
   
    image_folder_path = "/work/project/cmsc655/figures/godec/figures"
    
    #x_true_path = os.path.join(image_folder_path, "x_true_img_2000.nii")
    #sparse_path = os.path.join(image_folder_path, "x_sparse_hat_img_2000.nii")
    #guass_path = os.path.join(image_folder_path, "x_guass_hat_img_2000.nii")
    #low_rank_path = os.path.join(image_folder_path, "x_low_rank_hat_img_2000.nii")
    #noisy_path = os.path.join(image_folder_path, "x_noisy_img_2000.nii")
    
    x_true_path = os.path.join(image_folder_path, "x_true_img_2500.nii")
    sparse_path = os.path.join(image_folder_path, "x_sparse_hat_img_2500.nii")
    guass_path = os.path.join(image_folder_path, "x_guass_hat_img_2500.nii")
    low_rank_path = os.path.join(image_folder_path, "x_low_rank_hat_img_2500.nii")
    noisy_path = os.path.join(image_folder_path, "x_noisy_img_2500.nii")
    
    #fig_id = "frmi_summary_plot_signal_slice_nonoise_decomposition_" + fname
    #fig_path = os.path.join(folder,fig_id)
       
    #image_folder_path = "/work/scratch/alternate_minimation/nonoise1/run_2018-12-07_09_02_26/d4/gaussian/scans/final/mr/5000/"
    
    
    #x_true_path = os.path.join(image_folder_path, "x_true_img_5000.nii")
    #sparse_path = os.path.join(image_folder_path, "x_sparse_hat_img_5000.nii")
    #guass_path = os.path.join(image_folder_path, "x_guass_hat_img_5000.nii")
    #low_rank_path = os.path.join(image_folder_path, "x_low_rank_hat_img_5000.nii")
    #noisy_path = os.path.join(image_folder_path, "x_noisy_img_5000.nii")
    #
    
    frmi_plot(x_true_path, low_rank_path, sparse_path,  guass_path, spike_folder, fig_path, legend=True, tr=25, stddev=True, tsnr=True, img_slice = 8)
    