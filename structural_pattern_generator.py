import nilearn

from nilearn import image
import nibabel as nib
import copy
from nilearn import plotting
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from math import ceil
from nilearn.datasets import MNI152_FILE_PATH
from sklearn.model_selection import train_test_split
from nibabel.affines import apply_affine
from nilearn.image.resampling import coord_transform, get_bounds, get_mask_bounds
from skimage.draw import ellipsoid
from nilearn.image import resample_img
from nilearn.masking import compute_background_mask
import metric_util as mt
import data_util as du
import mri_draw_utils as mrd
import ellipsoid_masker as elpm
import ellipsoid_mask as em
import traceback

def generate_structural_missing_pattern(x0,y0,z0, x_r, y_r, z_r,frames_count, folder):
    subject_scan_path = du.get_full_path_subject1()

    n = 0
    
    masked_img_file_path  = folder + "/" + "size_" + str(x_r) + "_" + str(y_r) + "_" + str(z_r) + "_scan_" + str(n)
    
    corrupted_volumes_list = []
    corrupted_volumes_list_scan_numbers = []
    
    for i in xrange(frames_count):
        target_img = image.index_img(subject_scan_path,i)
        image_masked_by_ellipsoid = elpm.create_ellipsoid_mask(x0, y0, z0, x_r, y_r, z_r, target_img, masked_img_file_path)
        
       # ellipsoid = em.EllipsoidMask(x0, y0, z0, x_r, y_r, z_r, folder)
        #ellipsoid_volume = ellipsoid.volume()
        observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
        
        corrupted_volumes_list.append(image_masked_by_ellipsoid)
        corrupted_volumes_list_scan_numbers.append(i)
        #print ("Ellipsoid Volume: " + str(ellipsoid_volume) + "; Missing Ratio: " + str(observed_ratio))
    
    # now create corrupted 4d where fist n frames has ellipsoid missing across 10 frames
    counter = 0
    
    volumes_list = []
    for img in image.iter_img(subject_scan_path):
        print "Volume Index: " + str(counter)
        if counter in corrupted_volumes_list_scan_numbers:
            print "Adding corrupted volume to the list " + str(counter)
            volumes_list.append(corrupted_volumes_list[counter])
        else:
            print "Adding normal volume to the list " + str(counter)
            volumes_list.append(img)
        counter = counter + 1
        
    # now generate corrupted 4D from the list
    x_corr_img = image.concat_imgs(volumes_list)
    #observed_ratio = mt.compute_observed_ratio(image_masked_by_ellipsoid)
    #print ("Effective Missing Ratio: " + str(observed_ratio))
    return x_corr_img
