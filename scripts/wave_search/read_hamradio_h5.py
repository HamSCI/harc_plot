#!/usr/bin/env python3
import pandas as pd

h5_path = 'output/ham_superdarn_tec/raw_data/20171103.data.bz2.h5'
h5_key  = 'df'
df      = pd.read_hdf(h5_path,h5_key,complib='bzip2',complevel=9)
import ipdb; ipdb.set_trace()
