import tensorflow as tf
import numpy as np
import t3f
np.random.seed(0)
import copy
import math

def cost_with_treshold(x, thresh = 0.000212633):
    res = tf.abs(t3f.full(x))
    zeros = tf.zeros_like(res)
    masked = tf.greater(res, thresh)
    new_tensor = tf.where(masked, t3f.full(x), zeros)
    return t3f.to_tt_tensor(new_tensor, max_tt_rank=63)


def compute_threshold(x):
    result = 1.0/(math.sqrt(x.size))
    return result
    
def frobenius_norm_tf_squared(x):
    return tf.reduce_sum(x ** 2)

def frobenius_norm_tf(x):
    return tf.reduce_sum(x ** 2) ** 0.5

def compute_rel_error(self, x_curr, x_prev):
        return t3f.frobenius_norm(x_curr - x_prev)/t3f.frobenius_norm(x_prev)