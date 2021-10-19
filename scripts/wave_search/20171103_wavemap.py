#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import itertools
import tqdm

import numpy as np
import pandas as pd
import xarray as xr

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import harc_plot
import harc_plot.gen_lib as gl
from harc_plot.timeutils import daterange, strip_time
from harc_plot import geopack


# Create a 'view' list to quickly see frequencyly used columns.
v = []
v.append('occurred')
v.append('freq')
v.append('source')
v.append('md_lat')
v.append('md_long')
v.append('dist_Km')

#v.append('slt_mid')
#v.append('ut_hrs')
#v.append('tx_lat')
#v.append('tx_long')
#v.append('rx_lat')
#v.append('rx_long')
#
#v.append('rpt_key')
#v.append('tx')
#v.append('rx')
#v.append('rpt_mode')
#v.append('snr')
#v.append('tx_grid')
#v.append('rx_grid')
#v.append('band')
#v.append('tx_mode')
#v.append('tx_loc_source')
#v.append('rx_loc_source')

band_obj    = gl.BandData()

def load_psk(date_str,rgc_lim=None,filter_region=None,filter_region_kind='mids',**kwargs):
    if date_str != '2017-11-03':
        return

    data_dir    = 'data/pskreporter'
    fnames = []
    fnames.append('PSK_data_20171103_p1.csv.bz2')
    fnames.append('PSK_data_20171103_p2.csv.bz2')
    fnames.append('PSK_data_20171103_p3.csv.bz2')

    grid_prsn   = 6

    df  = pd.DataFrame()
    for fname in fnames:
        fpath   = os.path.join(data_dir,fname)
        print(fpath)

        dft = pd.read_csv(fpath,parse_dates=['theTimeStamp'])
        del dft['flowStartSeconds']

        df  = df.append(dft,ignore_index=True)

    df = df.rename(columns={'frequency':'freq','theTimeStamp':'occurred','senderLocator':'tx_grid','receiverLocator':'rx_grid'})
    df['freq'] = pd.to_numeric(df['freq'], errors='coerce') / 1000. # Convert frequency to kHz
    df = df.dropna()

    for prefix in ['tx_','rx_']:
        print('   Converting {!s}grid to {!s}lat and {!s}long...'.format(prefix,prefix,prefix))
        lats, lons  = harc_plot.locator.gridsquare2latlon(df[prefix+'grid'],new_precision=8)
        df[prefix+'lat'] = lats
        df[prefix+'long'] = lons
        print()

    # Compute tx-rx great circle distance.
    df['dist_Km'] = harc_plot.geopack.greatCircleDist(df['tx_lat'],df['tx_long'],df['rx_lat'],df['rx_long'])*6371

    midpoints       = harc_plot.geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat']    = midpoints[0]
    df['md_long']   = midpoints[1]

    # Regional Filtering
    if filter_region is not None:
        df_raw  = df.copy()
        df      = harc_plot.gl.regional_filter(filter_region,df,kind=filter_region_kind)

    if len(df) == 0:
        return

    df["ut_hrs"]    = df['occurred'].map(lambda x: x.hour + x.minute/60. + x.second/3600.)
    df['slt_mid']   = (df['ut_hrs'] + df['md_long']/15.) % 24.

    df['source']    = 3

    return df

class KeoHam(object):
    def __init__(self,run_dct):
        """
        data_sources: list, i.e. [1,2]
            0: dxcluster
            1: WSPRNet
            2: RBN
            3: PSKReporter [limited availability]

        loc_sources: list, i.e. ['P','Q']
            P: user Provided
            Q: QRZ.com or HAMCALL
            E: Estimated using prefix
        """

        # Get Variables from run_dct
        rd      = run_dct
#        rd['rgc_lim']            = rd['rgc_lim']
#        rd['filter_region']      = rd['filter_region']
#        rd['filter_region_kind'] = rd['filter_region_kind']
#        rd['xb_size_min']        = rd['xb_size_min']
#        rd['yb_size_km']         = rd['yb_size_km']
#        rd['band_MHz']           = rd['band_MHz']
#        rd['keo_grid']           = rd['keo_grid']

        rd['output_dir']         = rd.get('output_dir')
        rd['data_sources']       = rd.get('data_sources',[1,2,3])
        rd['reprocess_raw_data'] = rd.get('reprocess_raw_data',True)
        rd['win_len']            = rd.get('win_len',datetime.timedelta(minutes=5))

        sDate   = rd['sDate']
        eDate   = rd['eDate']
        win_len = rd['win_len']

#        rd['band']               = band_obj.band_dict[rd['band_MHz']]['meters']
        self.run_dct             = rd

        self.load_data()

        map_dates   = [sDate]
        while map_dates[-1] < eDate:
            map_dates.append(map_dates[-1]+win_len)

        for map_date in tqdm.tqdm(map_dates,dynamic_ncols=True):
            tqdm.tqdm.write(str(map_date))
            self.plot_map(map_date,prefix='tx_rx')

        for map_date in tqdm.tqdm(map_dates,dynamic_ncols=True):
            tqdm.tqdm.write(str(map_date))
            self.plot_map(map_date,prefix='mids')

        import ipdb; ipdb.set_trace()
    
    def load_data(self):
        """
        Load ham radio data into dataframe from CSV.
        """

        rd                  = self.run_dct
        output_dir          = rd['output_dir']
        reprocess_raw_data  = rd['reprocess_raw_data']
        sDate               = rd['sDate']
        eDate               = rd['eDate']
        data_sources        = rd['data_sources']
        rgc_lim             = rd['rgc_lim']
        filter_region       = rd['filter_region']
        filter_region_kind  = rd['filter_region_kind']
        band_MHz            = rd['band_MHz']

        # Define path for saving NetCDF Files
        h5s_path = os.path.join(output_dir,'raw_data')
        gl.prep_output({0:h5s_path},clear=reprocess_raw_data)

        # Loop through dates
        dates       = list(daterange(sDate, eDate))
        if strip_time(sDate) != strip_time(eDate):
            dates   = dates[:-1]

        df  = pd.DataFrame()
        for dt_inx,dt in enumerate(tqdm.tqdm(dates,dynamic_ncols=True)):
            h5_key  = 'df'
            h5_name = dt.strftime('%Y%m%d') + '.data.bz2.h5'
            h5_path = os.path.join(h5s_path,h5_name)

            if os.path.exists(h5_path):
                dft = pd.read_hdf(h5_path,h5_key,complib='bzip2',complevel=9)
            else:
                ld_str  = dt.strftime("%Y-%m-%d") 
                dft = gl.load_spots_csv(ld_str,data_sources=data_sources,
                                rgc_lim=rgc_lim,loc_sources=['P','Q'],
                                filter_region=filter_region,filter_region_kind=filter_region_kind)

                # Load in PSKReporter Data
                if 3 in data_sources:
                    df_pskr = load_psk(ld_str,rgc_lim=rgc_lim,
                                    filter_region=filter_region,filter_region_kind=filter_region_kind)
                    if df_pskr is not None:
                        if dft is None:
                            dft = df_pskr
                        else:
                            dft = dft.append(df_pskr,ignore_index=True)

                if dft is None:
                    print('No data for {!s}'.format(ld_str))
                    continue

                dft['band_MHz'] = np.floor(dft['freq']/1000.).astype(int)

                dft.to_hdf(h5_path,h5_key,complib='bzip2',complevel=9)

            df  = df.append(dft,ignore_index=True)

        df_0    = df.copy()

        # Double-check sources
        # Select spotting networks
        if data_sources is not None:
            tf  = df.source.map(lambda x: x in data_sources)
            df  = df[tf].copy()


        # Select desired times.
        tf      = np.logical_and(df['occurred'] >= sDate, df['occurred'] < eDate)
        df      = df[tf].copy()

        # Select desired band.
        tf      = df['band_MHz'] == band_MHz
        df      = df[tf].copy()

        # Enforce rgc_lim

        tf = np.logical_and(df['dist_Km'] >= rgc_lim[0], df['dist_Km'] < rgc_lim[1])
        df  = df[tf].copy()

        self.df = df

    def plot_map(self,map_date,prefix='mids'):

        rd                  = self.run_dct
        sDate               = rd['sDate']
        band_MHz            = rd['band_MHz']
        filter_region       = rd.get('filter_region','World')
        output_dir          = os.path.join(rd.get('output_dir'),prefix)
        gl.prep_output({0:output_dir},clear=False)

        fname               = '{!s}-{!s}.png'.format(prefix,map_date.strftime('%Y%m%d.%H%M'))
        fpath               = os.path.join(output_dir,fname)

        fig     = plt.figure(figsize=(10,5))
        ax      = fig.add_subplot(1,1,1,projection=ccrs.PlateCarree())

        df      = self.df
        tf      = np.logical_and(df['occurred'] >= map_date,
                                 df['occurred'] <  map_date + rd['win_len'])

        dft     = df[tf].copy()

        if prefix == 'mids':
            mpbl    = ax.scatter(dft['md_long'],dft['md_lat'],c=dft['dist_Km'],label='Midpoints',
                    vmin=0,vmax=2000)
            plt.colorbar(mpbl,label='dist_Km')
        elif prefix == 'tx_rx':
            ax.scatter(dft['tx_long'],dft['tx_lat'],label='TX')
            ax.scatter(dft['rx_long'],dft['rx_lat'],label='RX')

            for rinx,row in dft.iterrows():
                tx_lon  = row['tx_long']
                tx_lat  = row['tx_lat']
                rx_lon  = row['rx_long']
                rx_lat  = row['rx_lat']

                mpbl = ax.plot([tx_lon,rx_lon],[tx_lat,rx_lat],transform=ccrs.Geodetic())

        ax.legend(loc='upper right')

        ax.coastlines(zorder=10,color='k')
        grd = ax.gridlines()
        grd.top_labels      = True
        grd.bottom_labels   = True
        grd.left_labels     = True
        grd.right_labels    = True

        ax.set_title('')
        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}

        ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(len(dft)),
                ha='center',transform=ax.transAxes,fontdict=fdict)

        plot_rgn    = gl.regions.get(filter_region)
        ax.set_xlim(plot_rgn.get('lon_lim'))
        ax.set_ylim(plot_rgn.get('lat_lim'))

        ax.set_title(map_date.strftime('%Y %b %d %H:%M UT'))

        fig.tight_layout()
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)

if __name__ == '__main__':
    output_dir  = os.path.join('output/wave_maps_2017NOV03')
    gl.prep_output({0:output_dir},clear=False,php=False)

# Defaults
    rd  = {}
    rd['sDate']                 = datetime.datetime(2017,11,3,12)
    rd['eDate']                 = datetime.datetime(2017,11,4)
    rd['rgc_lim']               = (0.,2000)
    rd['data_sources']          = [1,2,3]
    rd['reprocess_raw_data']    = False
    rd['filter_region']         = 'US'
    rd['filter_region_kind']    = 'mids'
    rd['output_dir']            = output_dir

    rd['plot_summary_line']     = False

    rd['band_MHz']              = 14

    keo_ham = KeoHam(rd)
    import ipdb; ipdb.set_trace()
