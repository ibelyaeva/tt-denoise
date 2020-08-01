import nilearn as ni
from nilearn import datasets as ds

data_dir = "/work/pl/sch/analysis/data/cobre"
ds.fetch_cobre(n_subjects=2, data_dir=data_dir, verbose=1)