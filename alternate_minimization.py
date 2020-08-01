
# coding: utf-8

# In[1]:


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
from skimage.measure import compare_ssim as ssim
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


# In[2]:


subject_scan_path = du.get_full_path_subject1()
print "Subject Path: " + str(subject_scan_path)
x_true_org = mt.read_image_abs_path(subject_scan_path)
ground_truth = np.array(x_true_org.get_data()).astype('float32')
ten_ones = np.ones_like(ground_truth).astype('float32')
ten_zeros = np.zeros_like(ground_truth).astype('float32')


# In[3]:


#Initialize variables


# In[4]:


x_init = copy.deepcopy(ground_truth)
norm_ground_x_init = np.linalg.norm(x_init)
x_init = x_init * (1./norm_ground_x_init)
tf_ten_zeros = t3f.to_tt_tensor(ten_zeros, max_tt_rank=63)
tf_ten_ones = t3f.to_tt_tensor(ten_ones, max_tt_rank=63)


# In[5]:


ground_truth_tf = t3f.to_tt_tensor(x_init, max_tt_rank=63)
tf_zeros = t3f.get_variable('tf_zeros', initializer=tf_ten_zeros, trainable=False)
tf_ones = t3f.get_variable('tf_ones', initializer=tf_ten_ones, trainable=False)
X = t3f.get_variable('X', initializer=ground_truth_tf, trainable=False)
L = t3f.get_variable('L', initializer=ground_truth_tf)
S = t3f.get_variable('S', initializer=tf_zeros)

#Sold = t3f.get_variable('Sold', initializer=tf_ten_zeros)
#G = t3f.get_variable('G', initializer=tf_ten_zeros)


# In[6]:



#Initialize Gradienst


# In[7]:


gradS = X - L
gradL = X - S


# In[8]:


rimGradS = t3f.riemannian.project(gradS,S)
rimGradL = t3f.riemannian.project(gradL,L)


# In[9]:


alpha = 0.1

train_step_tensor_s = t3f.assign(S, t3f.round(S - alpha * rimGradS, max_tt_rank=63))
train_step_tensor_l = t3f.assign(L, t3f.round(L - alpha * rimGradL, max_tt_rank=63))


# In[10]:


loss = 0.5 * t3f.frobenius_norm_squared(X - L - S)


# In[11]:


def relative_convergence_error(x_k, x_xprev):
    np.linalg.norm(x_k - x_prev)/np.linalg.norm(x_prev)


# In[12]:


cost_S = []
cost_L = []


# In[ ]:


sess = tf.Session()
sess.run(tf.global_variables_initializer())


# In[ ]:


for i in range(10):
    sess.run([train_step_tensor_s.op])
    #loss_val,_, = sess.run([train_step_tensor_l.op])
    #print "loss :" + str(loss_val)
    

