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
    filter_region       = run_dct['filter_region']
    filter_region_kind  = run_dct['filter_region_kind']
    loc_sources         = run_dct.get('loc_sources')
    data_sources        = run_dct.get('data_sources',[1,2])
    data_out            = run_dct['data_out']
    bands               = run_dct.get('bands',[1,3,7,14,21,28])

    sources = {0:'DXCluster',1:'WSPR',2:'RBN'}
    sources_included = []
    for src_inx,source in sources.items():
        if src_inx in data_sources:
            sources_included.append(source)

    # Loop through dates
    dates       = list(daterange(sDate, eDate))
    if strip_time(sDate) != strip_time(eDate):
        dates   = dates[:-1]

    for dt in dates:
        dt_str  = dt.strftime('%Y%m%d')
        fname   = '{!s}-{!s}-{!s}.csv'.format(dt_str,'_'.join(sources_included),filter_region)
        fpath   = os.path.join(data_out,fname)

        bz2_fpath   = fpath + '.bz2'
        if os.path.exists(bz2_fpath):
            print('File already processed: {!s}'.format(bz2_fpath))
            continue

        ld_str  = dt.strftime("%Y-%m-%d") 
        print('Processing {!s}'.format(ld_str))
        df = gl.load_spots_csv(ld_str,data_sources=data_sources,
                        rgc_lim=rgc_lim,loc_sources=loc_sources,
                        filter_region=filter_region,filter_region_kind=filter_region_kind)

        if df is None:
            print('No data for {!s}'.format(ld_str))
            continue

        # Generate MHz Column
        df['MHz']   = np.floor(df['freq']/1000.)

        # Get only HF Bands
        bands_tf = []
        for band in bands:
            tf = df['MHz'] == band
            bands_tf.append(tf)

        tf  = np.logical_or.reduce(bands_tf)
        df  = df[tf].copy()

        new_tx  = []
        new_rx  = []
        for rinx,row in tqdm.tqdm(df.iterrows(),desc='Separating Call Signs',dynamic_ncols=True,total=len(df)):
            occurred        = row['occurred']
            occurred_str    = occurred.strftime('%Y%m%d%H%M')
            tx,rx           = row['rpt_key'].split(occurred_str)
            rx              = rx.split('_')[0]
            new_tx.append(tx)
            new_rx.append(rx)

        df['tx'] = new_tx
        df['rx'] = new_rx

        keys = []
        keys.append('occurred')
        keys.append('source')
        keys.append('freq')
    #    keys.append('MHz')
        keys.append('snr')
        keys.append('tx')
        keys.append('rx')
        keys.append('tx_lat')
        keys.append('tx_long')
        keys.append('rx_lat')
        keys.append('rx_long')
        keys.append('dist_Km')
        keys.append('md_lat')
        keys.append('md_long')
        keys.append('slt_mid')

        df  = df[keys].copy()
        df  = df.rename(columns={'occurred':'datetime_ut','freq':'freq_kHz','md_lat':'midpoint_lat','md_long':'midpoint_long','slt_mid':'midpoint_localTime'})

        df  = df.set_index('datetime_ut')

        for src_inx,source in sources.items():
            tf  = df['source']  == src_inx
            df.loc[tf,'source'] = source

        print('Saving file: {!s}'.format(fpath))
        
        with open(fpath,'w') as fl:
            txt = '# Amateur (Ham) Radio Data File\n'
            fl.write(txt)
            fl.write('#\n')

            for key,item in run_dct.items():
                if key == 'data_out': continue
                txt = '# {!s}: {!s}\n'.format(key,item)
                fl.write(txt)

            txt = '# Sources: {!s}\n'.format(sources_included)
            fl.write(txt)

            txt = '# Bands: {!s}\n'.format(bands)
            fl.write(txt)

            fl.write('#\n')


        df.to_csv(fpath,mode='a')
        
        print('BZip2ing: {!s}'.format(bz2_fpath))
        os.system('bzip2 {!s}'.format(fpath))

if __name__ == '__main__':
    #run_name    = 'Europe'
    run_name    = 'World'
    data_out    = os.path.join('data_out2')
    sTime       = datetime.datetime(2010,1,1)
    eTime       = datetime.datetime(2020,1,1)
    region      = run_name
#    rgc_lim     = (0, 10000)
    rgc_lim     = None

    gl.prep_output({0:data_out})

    # Create histogram NetCDF Files ################################################
    rd  = {}
    rd['sDate']                 = sTime
    rd['eDate']                 = eTime
    rd['rgc_lim']               = rgc_lim
    rd['filter_region']         = run_name
    rd['filter_region_kind']    = 'mids'
    rd['data_out']              = data_out

    main(rd)
