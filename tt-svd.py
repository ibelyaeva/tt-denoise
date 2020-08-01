## TT_SVD on ortogonal matrices

def tt_svd(tens, max_tt_rank=10, epsilon=None):
  tens = tf.convert_to_tensor(tens)
  static_shape = tens.get_shape()
  dynamic_shape = tf.shape(tens)

  d = static_shape.__len__()
  max_tt_rank = np.array(max_tt_rank).astype(np.int32)
 
  if epsilon is not None and epsilon < 0:
    raise ValueError('Epsilon should be non-negative.')
  if max_tt_rank.size == 1:
    max_tt_rank = (max_tt_rank * np.ones(d + 1)).astype(np.int32)
  elif max_tt_rank.size != d + 1:
    raise ValueError('max_tt_rank should be a number or a vector of size (d+1)')
  ranks = [1] * (d + 1)
  tt_cores = []
  are_tt_ranks_defined = True
  for core_idx in range(d - 1):
    curr_mode = static_shape[core_idx].value
    if curr_mode is None:
      curr_mode = dynamic_shape[core_idx]
    rows = ranks[core_idx] * curr_mode
    tens = tf.reshape(tens, [rows, -1])
    columns = tens.get_shape()[1].value
    if columns is None:
      columns = tf.shape(tens)[1]
    s, u, v = tf.svd(tens, full_matrices=False)
    if max_tt_rank[core_idx + 1] == 1:
      ranks[core_idx + 1] = 1
    else:
      try:
        ranks[core_idx + 1] = min(max_tt_rank[core_idx + 1], rows, columns)
      except TypeError:
        # Some of the values are undefined on the compilation stage and thus
        # they are tf.tensors instead of values.
        min_dim = tf.minimum(rows, columns)
        ranks[core_idx + 1] = tf.minimum(max_tt_rank[core_idx + 1], min_dim)
        are_tt_ranks_defined = False
    
    u = u[:, 0:ranks[core_idx + 1]]
    s = s[0:ranks[core_idx + 1]]
    v = v[:, 0:ranks[core_idx + 1]]
    core_shape = (ranks[core_idx], curr_mode, ranks[core_idx + 1])
    print ranks, core_idx, core_shape
    tt_cores.append(tf.reshape(u, core_shape))
    tens = tf.matmul(tf.diag(s), tf.transpose(v))
  last_mode = static_shape[-1].value
  if last_mode is None:
    last_mode = dynamic_shape[-1]
  core_shape = (ranks[d - 1], last_mode, ranks[d])
  tt_cores.append(tf.reshape(tens, core_shape))
  print "ranks: " + str(ranks)
  if not are_tt_ranks_defined:
    ranks = None
  return TensorTrain(tt_cores, static_shape, ranks)
