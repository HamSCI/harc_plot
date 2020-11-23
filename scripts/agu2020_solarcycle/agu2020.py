#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import bz2
import pickle

import hashlib

import harc_plot
from harc_plot import visualize_histograms as vh


def hash_rd(run_dict):
    """
    Quick hash of the run dictionary for checking that pickle files are up-to-date.
    """
    rd_ = repr(sorted(run_dict.items())).encode('utf-8')
    m   = hashlib.sha1()
    m.update(rd_)
    return m.hexdigest()


def load_nc_cache(rd):
    rd_hash     = hash_rd(rd)
    sTime_str   = rd['sTime'].strftime('%Y%m%d')
    eTime_str   = rd['eTime'].strftime('%Y%m%d')
    fname       = '{!s}-{!s}.{!s}.p.bz2'.format(sTime_str,eTime_str,rd_hash)

    cache_dir   = os.path.join(rd['baseout_dir'],'cache')
    fpath       = os.path.join(cache_dir,fname)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if os.path.exists(fpath):
        print('Loading cached file: {!s}'.format(fpath))
        with bz2.BZ2File(fpath,'rb') as fl:
            nc_obj = pickle.load(fl)
    else:
        nc_obj              = vh.ncLoader(**rd)
        nc_obj.rd_original  = rd

        print('Saving cached file: {!s}'.format(fpath))
        with bz2.BZ2File(fpath,'wb') as fl:
            pickle.dump(nc_obj,fl)

    return nc_obj

#run_name    = 'Europe'
run_name    = 'World'
data_dir    = os.path.join('data/solarcycle_3hr_250km/histograms',run_name)
plot_dir    = os.path.join('output/galleries/solarcycle_3hr_250km',run_name)
#params      = ['spot_density']
xkeys       = ['ut_hrs','slt_mid']
#xkeys       = ['ut_hrs']
#sTime       = datetime.datetime(2009,1,1)
#eTime       = datetime.datetime(2020,1,1)

sTime       = datetime.datetime(2018,1,1)
eTime       = datetime.datetime(2018,1,31)

region      = run_name
rgc_lim     = (0, 10000)

#geo_env     = harc_plot.GeospaceEnv()
geo_env = None

# Visualization ################################################################
### Visualize Observations
rd = {}
rd['srcs']                  = os.path.join(data_dir,'*.data.nc.bz2')
rd['baseout_dir']           = plot_dir
rd['sTime']                 = sTime
rd['eTime']                 = eTime
rd['plot_region']           = region
rd['geospace_env']          = geo_env
rd['plot_sza']              = False
rd['plot_trend']            = True 
rd['plot_kpsymh']           = False
rd['plot_goes']             = False
rd['log_z']                 = False
rd['band_keys']             = [28, 21, 14, 7, 3, 1]
rd['xkeys']                 = xkeys

nc = load_nc_cache(rd)
nc.plot(**rd)

import ipdb; ipdb.set_trace()
#harc_plot.visualize_histograms.main(rd)
#harc_plot.visualize_histograms.plot_dailies(rd)


