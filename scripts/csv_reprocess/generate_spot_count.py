#!/usr/bin/env python3
"""
Script to reprocess CSV data into something more useable.
"""
import os
import datetime
import tqdm

import numpy as np
import pandas as pd

import harc_plot
from harc_plot.timeutils import daterange,strip_time
from harc_plot import gen_lib as gl

def main(run_dct):
    """
    data_sources: list, i.e. [1,2]
        0: dxcluster
        1: WSPRNet
        2: RBN

    loc_sources: list, i.e. ['P','Q']
        P: user Provided
        Q: QRZ.com or HAMCALL
        E: Estimated using prefix
    """


    # Get Variables from run_dct
    sDate               = run_dct['sDate']
    eDate               = run_dct['eDate']
    rgc_lim             = run_dct.get('rgc_lim')
    filter_regions      = run_dct['filter_regions']
    filter_region_kind  = run_dct['filter_region_kind']
    loc_sources         = run_dct.get('loc_sources')
    data_sources        = run_dct.get('data_sources',[1,2])
    data_out            = run_dct['data_out']
    bands               = run_dct.get('bands',[1,3,7,14,21,28])
    bin_size            = run_dct.get('bin_size',datetime.timedelta(minutes=1))

    # Generate bins for each day.
    bin_size_hr         = bin_size.total_seconds()/3600.
    bins                = np.arange(0,24+bin_size_hr,bin_size_hr)

    sources = {0:'DXCluster',1:'WSPR',2:'RBN'}
    sources_included = []
    for src_inx,source in sources.items():
        if src_inx in data_sources:
            sources_included.append(source)

    # Loop through dates
    dates       = list(daterange(sDate, eDate))
    if strip_time(sDate) != strip_time(eDate):
        dates   = dates[:-1]

    dt0_str     = dates[0].strftime('%Y%m%d')
    dt1_str     = dates[-1].strftime('%Y%m%d')
    fname       = '{!s}-{!s}-{!s}_spot_counts.csv'.format(dt0_str,dt1_str,'_'.join(sources_included))
    fpath       = os.path.join(data_out,fname)
    bz2_fpath   = fpath + '.bz2'
    if os.path.exists(bz2_fpath):
        print('File already processed: {!s}'.format(bz2_fpath))
        return

    spot_counts = None
    for dt in dates:
        ld_str  = dt.strftime("%Y-%m-%d") 
        print('Processing {!s}'.format(ld_str))
        bins_dt = [dt + datetime.timedelta(hours=x) for x in bins]

        columns = {}
        for filter_region in filter_regions:
            df = gl.load_spots_csv(ld_str,data_sources=data_sources,
                            rgc_lim=rgc_lim,loc_sources=loc_sources,
                            filter_region=filter_region,filter_region_kind=filter_region_kind)

            if df is None:
                print('No data for {!s}'.format(ld_str))
                continue

            # Generate MHz Column
            df['MHz']   = np.floor(df['freq']/1000.)
            
            df['ut_hr'] = df['occurred'].apply(lambda x:x.hour + x.minute/60. + x.second/3600.)

            for band in bands:
                col_key             = '{!s}MHz-{!s}'.format(band,filter_region)

                tf  = df['MHz'] == band
                dfb = df[tf].copy()
                hist, bins_out      = np.histogram(dfb['ut_hr'],bins) 
                columns[col_key]    = hist

        spc = pd.DataFrame(columns,index=bins_dt[:-1])
        spc.index.rename('datetime_ut',inplace=True)
        
        if spot_counts is None:
            spot_counts = spc
        else:
            spot_counts = spot_counts.append(spc)

    print('Saving file: {!s}'.format(fpath))
    with open(fpath,'w') as fl:
        txt = '# Amateur (Ham) Radio Spot Count File\n'
        fl.write(txt)
        fl.write('#\n')

        for key,item in run_dct.items():
            if key == 'data_out': continue
            txt = '# {!s}: {!s}\n'.format(key,item)
            fl.write(txt)

        txt = '# Sources: {!s}\n'.format(sources_included)
        fl.write(txt)

        fl.write('#\n')

    spot_counts.to_csv(fpath,mode='a')
    
if __name__ == '__main__':
    data_out    = os.path.join('spot_counts')
    sTime       = datetime.datetime(2013,5,1)
    eTime       = datetime.datetime(2013,6,1)
    regions     = ['US','Europe']
    rgc_lim     = None

    gl.prep_output({0:data_out})

    # Create histogram NetCDF Files ################################################
    rd  = {}
    rd['sDate']                 = sTime
    rd['eDate']                 = eTime
    rd['rgc_lim']               = rgc_lim
    rd['filter_regions']        = regions
    rd['filter_region_kind']    = 'mids'
    rd['bands']                 = [7,14]
    rd['data_out']              = data_out
    rd['bin_size']              = datetime.timedelta(minutes=1)

    main(rd)
