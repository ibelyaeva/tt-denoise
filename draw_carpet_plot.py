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
import os.path as op
import spike_detection as sp
from scipy.stats import zscore

def frmi_plot(in_file, spike_folder, file_path, legend=False, tr=None, stddev=False, tsnr=False, img_slice=None):
    
    x_img = mt.read_image_abs_path(in_file)
    data = np.array(x_img.get_data())
    z_scored_data = np.nan_to_num(zscore( data, axis=-1))
    fname, ext = op.splitext(op.basename(in_file))
    
    n_spikes, out_spikes, out_fft, spikes_list = sp.slice_wise_fft(in_file, spike_folder, spike_thres=4.)
    dvar_file_name = fname + "_dvars.tsv"
    
    mask_file="epi_mask.nii"

    dvars = nac.ComputeDVARS()
    dvars.inputs.in_file = in_file
    dvars.inputs.in_mask = mask_file
    dvars.inputs.save_all = True
    dvars.inputs.save_plot = True
    dvars.inputs.figformat = 'pdf'
    dvars.run()
    
    out_suffix = fname + ".nii.gz"
    tsnr_file = os.path.join(spike_folder, "tsnr_" + out_suffix)
    mean_file = os.path.join(spike_folder, "mean_" + out_suffix)
    stddev_file = os.path.join(spike_folder, "stddev_" + out_suffix)
    tsnr = nac.TSNR()
    tsnr.inputs.in_file = in_file
    tsnr.inputs.tsnr_file = tsnr_file
    tsnr.inputs.mean_file = mean_file
    tsnr.inputs.stddev_file = stddev_file
    tsnr.run()
    
    dataframe = pd.DataFrame({
                'DVARS': [np.nan] + np.loadtxt(
                'dvars.tsv', skiprows=1, usecols=[1]).tolist(),
        })
    
    if tr and tsnr and stddev:
        fm_plot = fm.fMRIPlot(
                in_file,
                mask_file=mask_file,
                spikes_files=[out_spikes],
                seg_file=None,
                tr=tr,
                data=dataframe[['DVARS']],
                legend=legend,
                img_slice = img_slice,
                tsnr = tsnr_file,
                stddev =  stddev_file
                )
    elif tr:
        fm_plot = fm.fMRIPlot(
                in_file,
                mask_file=mask_file,
                spikes_files=[out_spikes],
                seg_file=None,
                tr=tr,
                data=dataframe[['DVARS']],
                legend=legend
                )
    else:
        fm_plot = fm.fMRIPlot(
                in_file,
                mask_file=mask_file,
                spikes_files=[out_spikes],
                seg_file=None,
                data=dataframe[['DVARS']],
                legend=legend,
                img_slice = img_slice
                )
    fm_plot.plot()
    
    dr.save_fig_pdf_white(file_path)
    
if __name__ == "__main__":

    #in_file = du.get_full_path_subject1()
    in_file = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80-100/x_miss_45_80_100.nii"
    fname, ext = op.splitext(op.basename(in_file))
    folder = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80-100/"
    spike_folder = "/work/scratch/tensor_completion/4D/noise/corrupted_subjects/artifacts/45-80-100/"
    
    fig_id = fname
    fig_path = os.path.join(folder,fig_id)
    frmi_plot(in_file, spike_folder, fig_path, legend=True, tr=None, stddev=True, tsnr=True, img_slice = 8)
    
    #fig_id = "frmi_summary_plot_al" + fname
    #frmi_plot(in_file, spike_folder, fig_path, legend=True, tr=None)
    