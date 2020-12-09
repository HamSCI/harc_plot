#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import bz2
import pickle

import argparse

import hashlib

import numpy as np
import pandas as pd

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


def load_nc_cache(rd,reprocess=False):
    rd_hash     = hash_rd(rd)
    sTime_str   = rd['sTime'].strftime('%Y%m%d')
    eTime_str   = rd['eTime'].strftime('%Y%m%d')
    fname       = '{!s}-{!s}.{!s}.p.bz2'.format(sTime_str,eTime_str,rd_hash)

    cache_dir   = os.path.join(rd['baseout_dir'],'cache')
    fpath       = os.path.join(cache_dir,fname)

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)


    if os.path.exists(fpath) and (reprocess is True):
        os.remove(fpath)

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


def main(data_source='WSPRNet_RBN',diurnal='day'):

    region          = 'World'
    run_name        = '-'.join([region,data_source])
    data_dir        = os.path.join('data/solarcycle_3hr_250km/histograms',run_name)
    plot_dir        = os.path.join('output/galleries/solarcycle_3hr_250km',run_name)

#    xkeys       = ['slt_mid','ut_hrs']
    xkeys       = ['slt_mid']
    sTime       = datetime.datetime(2009,1,1)
    eTime       = datetime.datetime(2020,1,1)

#    sTime       = datetime.datetime(2015,1,1)
#    eTime       = datetime.datetime(2015,2,1)

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
    rd['plot_f107']             = True
    rd['log_z']                 = False
    rd['band_keys']             = [28, 21, 14, 7, 3, 1]
    rd['xkeys']                 = xkeys

    nc      = load_nc_cache(rd,reprocess=False)

    # Select Day / Night ###########################################################
    rd['no_percentages'] = True
    for prefix in ['time_series','map']:
        ds          = nc.datasets[prefix]['slt_mid']                 # Pull out the dataset of interest.
        attrs       = ds['spot_density'].attrs
        slt_mids    = ds['slt_mid'] % 24                                # Modulo 24 so we see what hour of the day it actually is

        if diurnal == 'night':
            tf      = np.logical_or(slt_mids >=18, slt_mids < 6)            # Find where it is night only.
            ds      = ds.loc[{'slt_mid': ds['slt_mid'][tf]}].copy() # Select the night times in the data array.

            ds      = ds.rolling({'slt_mid':4}).sum()                   # Rolling sum of every 4 points.

            tf      = (ds['slt_mid']%24) == 18                          # Only keep those data points starting at LT == 18 hr.
            ds      = ds.loc[{'slt_mid': ds['slt_mid'][tf]}].copy()
            rd['fname_suffix']  = 'NIGHT'
            rd['title']         = 'Night (18 - 06 SLT)'

        elif diurnal == 'day':
            tf      = np.logical_and(slt_mids >=6, slt_mids < 18)            # Find where it is day only.
            ds      = ds.loc[{'slt_mid': ds['slt_mid'][tf]}].copy()  # Select the night times in the data array.

            ds      = ds.rolling({'slt_mid':4}).sum()                    # Rolling sum of every 4 points.

            tf      = (ds['slt_mid']%24) == 6                            # Only keep those data points starting at LT == 6 hr.
            ds      = ds.loc[{'slt_mid': ds['slt_mid'][tf]}].copy()
            rd['fname_suffix']  = 'DAY'
            rd['title']         = 'Day (06 - 18 SLT)'

        ds['spot_density'].attrs = attrs 
        nc.datasets[prefix]['slt_mid'] = ds                          # Put the data array back into the plotting object.


    fpaths  = nc.plot(**rd)

    print()
    if fpaths is not None:
        for fpath in fpaths:
            print('http://arrow.lan/~w2naf/code/harc_plot/scripts/agu2020_solarcycle/'+fpath)
    print()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_source',default='WSPRNet_RBN',help='[WSPRNet, RBN, WSPRNet_RBN]')
    parser.add_argument('--diurnals', default=['day','night',None])
    args = parser.parse_args()

    diurnals = args.diurnals

    for diurnal in diurnals:
        rd = {}
        rd['diurnal']           = diurnal
        rd['data_source']       = args.data_source
        print(rd)
        main(**rd)
