#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import itertools
import glob
import bz2
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

import pydarnio
import pydarn


def load_fitacf(sTime,eTime,radar,data_dir='data/superdarn-bas',fit_sfx='fitacf3'):
    """
    Load FITACF data from multiple FITACF files by specifying a date range.

    This routine assumes bz2 compression.
    """
    sDate   = datetime.datetime(sTime.year,sTime.month,sTime.day)
    eDate   = datetime.datetime(eTime.year,eTime.month,eTime.day)

    # Create a list of days we need fitacf files from.
    dates   = [sDate]
    while dates[-1] < eDate:
        next_date   = dates[-1] + datetime.timedelta(days=1)
        dates.append(next_date)

    # Find the data files that fall in that date range.
    fitacf_paths_0    = []
    for date in dates:
        date_str        = date.strftime('%Y%m%d')
        fpattern        = os.path.join(data_dir,'{!s}*{!s}*.{!s}.bz2'.format(date_str,radar,fit_sfx))
        fitacf_paths_0   += glob.glob(fpattern)

    # Sort the files by name.
    fitacf_paths_0.sort()

    # Get rid of any files we don't need.
    fitacf_paths = []
    for fpath in fitacf_paths_0:
        date_str    = os.path.basename(fpath)[:13]
        this_time   = datetime.datetime.strptime(date_str,'%Y%m%d.%H%M')

        if this_time <= eTime:
            fitacf_paths.append(fpath)

    # Load and append each data file.

    print()
    fitacf = []
    for fitacf_path in tqdm.tqdm(fitacf_paths,desc='Loading {!s} Files'.format(fit_sfx),dynamic_ncols=True):
        tqdm.tqdm.write(fitacf_path)
        with bz2.open(fitacf_path) as fp:
            fitacf_stream = fp.read()

        reader  = pydarnio.SDarnRead(fitacf_stream, True)
        records = reader.read_fitacf()
        fitacf += records
    return fitacf

class KeoHam(object):
    def __init__(self,run_dct):
        # Get Variables from run_dct
        rd  = run_dct

        rd['xkey']               = rd.get('xkey','ut_hrs')
        rd['xlim']               = rd.get('xlim',(12,24))
        rd['ylim']               = rd.get('ylim',None)
        rd['yticks']             = rd.get('yticks',None)
        rd['output_dir']         = rd.get('output_dir')
        rd['data_sources']       = rd.get('data_sources',[1,2,3])
        rd['reprocess_raw_data'] = rd.get('reprocess_raw_data',True)

        self.run_dct             = rd

        self.plot_timeseries()
    
    def plot_timeseries(self):
        rd                  = self.run_dct
        output_dir          = rd['output_dir']
        sDate               = rd['sDate']
        eDate               = rd['eDate']
        xlim                = rd['xlim']
        ylim                = rd['ylim']
        yticks              = rd['yticks']
        rgc_lim             = rd['rgc_lim']
        xkey                = rd['xkey']

        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
        fname       = '{!s}_timeseries.png'.format(date_str)

        fig     = plt.figure(figsize=(35,10))
        col_0       = 0
        col_0_span  = 30
        col_1       = 35
        col_1_span  = 65
        nrows       = 2
        ncols       = 100

        ################################################################################
        # Plot Map #####################################################################
        ax = plt.subplot2grid((nrows,ncols),(0,col_0),
                projection=ccrs.PlateCarree(),colspan=col_0_span)
        ax_00       = ax

        lbl_size    = 24
        ax.set_title('(a)',{'size':lbl_size},'left')
        
        df = None
        self.map_ax(df,ax)

        ################################################################################ 
        # Plot Time Series ############################################################# 
        ax      = plt.subplot2grid((nrows,ncols),(0,col_1),colspan=col_1_span)
        ax_01   = ax
#        result  = self.time_series_ax(df,ax,vmax=None,log_z=False)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.set_xlabel('')
        ax.set_ylabel('Great Circle Range [km]')
        ax.set_title('GNU Chirpsounder Keogram')
        ax.set_title('(b)',{'size':lbl_size},'left')

        ################################################################################
        # Plot SuperDARN Map ###########################################################
        radar       = 'bks'
        beam        = 13
        fitacf      = load_fitacf(sDate,eDate,radar)
        hdw_data    = pydarn.read_hdw_file(radar,sDate)

        ax = plt.subplot2grid((nrows,ncols),(1,col_0),
                projection=ccrs.PlateCarree(),colspan=col_0_span)
        ax_10       = ax
        self.map_ax_superdarn(ax,hdw_data,beam)
        ax.set_title('{!s} SuperDARN Radar Beam {!s}'.format(radar.upper(),beam))
        ax.set_title('(c)',{'size':lbl_size},'left')

        ################################################################################ 
        # Plot SuperDARN Time Series ################################################### 
        ax      = plt.subplot2grid((nrows,ncols),(1,col_1),colspan=col_1_span)
        ax_11   = ax
        pydarn.RTP.plot_range_time(fitacf, beam_num=beam, parameter='p_l', zmax=50, zmin=0, date_fmt='%H', colorbar_label='Power (dB)', cmap='viridis',ax=ax)

        ax.set_ylabel('Slant Range [km]')
        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_xlim(sDate,eDate)
        ax.set_xlabel('Time [UT]')
        ax.set_title('{!s} SuperDARN Radar Beam {!s}'.format(radar.upper(),beam))
        ax.set_title('(d)',{'size':lbl_size},'left')

        gl.adjust_axes(ax_01,ax_11)
        gl.adjust_axes(ax_10,ax_00)
        
        ax_0        = ax_10
        ax_1        = ax_00

        ax_0_pos    = list(ax_0.get_position().bounds)
        ax_1_pos    = list(ax_1.get_position().bounds)
        ax_0_pos[0] = ax_1_pos[0]
        ax_0_pos[2] = ax_1_pos[2]
        ax_0.set_position(ax_0_pos)

        dfmt    = '%Y %b %d %H%M UT'
        title   = '{!s} - {!s}'.format(sDate.strftime(dfmt), eDate.strftime(dfmt))
        fig.text(0.5,0.95,title,fontdict={'weight':'bold','size':24},ha='center')

        fpath       = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)

    def map_ax_superdarn(self,ax,hdw_data,beam):
        rd                  = self.run_dct
        sDate               = rd['sDate']
        filter_region       = rd.get('filter_region','World')

        map_attrs                       = {}
        map_attrs['xlim']               = (-180,180)
        map_attrs['ylim']               = (-90,90)

#        ax.coastlines(zorder=10,color='k')
#        ax.add_feature(cartopy.feature.LAND)
#        ax.add_feature(cartopy.feature.OCEAN)
        ax.add_feature(cartopy.feature.COASTLINE)
        ax.add_feature(cartopy.feature.BORDERS, linestyle=':')
#        ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
#        ax.add_feature(cartopy.feature.RIVERS)
        ax.set_title('')

        beams_lats, beams_lons  = pydarn.radar_fov(hdw_data.stid,coords='geo',date=sDate)
        fan_shape           = beams_lons.shape

        # FOV Outline ##################################################################
        gate_max            = 75
        fov_lons_left       = beams_lons[0:gate_max,0]
        fov_lats_left       = beams_lats[0:gate_max,0]

        fov_lons_right      = beams_lons[0:gate_max,-1]
        fov_lats_right      = beams_lats[0:gate_max,-1]

        fov_lons_top        = beams_lons[gate_max,0:]
        fov_lats_top        = beams_lats[gate_max,0:]

        fov_lons_bot        = beams_lons[0,0:]
        fov_lats_bot        = beams_lats[0,0:]

        fov_lons            = fov_lons_left.tolist()        \
                            + fov_lons_top.tolist()         \
                            + fov_lons_right.tolist()[::-1] \
                            + fov_lons_bot.tolist()[::-1]

        fov_lats            = fov_lats_left.tolist()        \
                            + fov_lats_top.tolist()         \
                            + fov_lats_right.tolist()[::-1] \
                            + fov_lats_bot.tolist()[::-1]

        ax.fill(fov_lons,fov_lats,color='0.8',ec='k')

        # Beam Outline #################################################################
        beam_lons_left       = beams_lons[0:gate_max,beam]
        beam_lats_left       = beams_lats[0:gate_max,beam]

        beam_lons_right      = beams_lons[0:gate_max,beam+1]
        beam_lats_right      = beams_lats[0:gate_max,beam+1]

        beam_lons_top        = beams_lons[gate_max,beam:beam+1]
        beam_lats_top        = beams_lats[gate_max,beam:beam+1]

        beam_lons_bot        = beams_lons[0,beam:beam+1]
        beam_lats_bot        = beams_lats[0,beam:beam+1]

        beam_lons           = beam_lons_left.tolist()        \
                            + beam_lons_top.tolist()         \
                            + beam_lons_right.tolist()[::-1] \
                            + beam_lons_bot.tolist()[::-1]

        beam_lats           = beam_lats_left.tolist()        \
                            + beam_lats_top.tolist()         \
                            + beam_lats_right.tolist()[::-1] \
                            + beam_lats_bot.tolist()[::-1]

        ax.fill(beam_lons,beam_lats,color='r',ec='k')

        ax.scatter(hdw_data.geographic.lon,hdw_data.geographic.lat,s=25)

        plot_rgn    = gl.regions.get(filter_region)
        ax.set_xlim(plot_rgn.get('lon_lim'))
        ax.set_ylim(plot_rgn.get('lat_lim'))

    def map_ax(self,df,ax):
        rd                  = self.run_dct
        sDate               = rd['sDate']
        band_MHz            = rd['band_MHz']
        filter_region       = rd.get('filter_region','World')

        map_attrs                       = {}
        map_attrs['sTime']              = sDate
        map_attrs['xkey']               = 'md_long'
        map_attrs['xlim']               = (-180,180)
        map_attrs['dx']                 = 1
        map_attrs['ykey']               = 'md_lat'
        map_attrs['ylim']               = (-90,90)
        map_attrs['dy']                 = 1

        ax.coastlines(zorder=10,color='w')
        ax.plot(np.arange(10))
        ax.set_title('')
        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}

#        ax.text(0.5,-0.1,txt,
#                ha='center',transform=ax.transAxes,fontdict=fdict)

        plot_rgn    = gl.regions.get(filter_region)
        ax.set_xlim(plot_rgn.get('lon_lim'))
        ax.set_ylim(plot_rgn.get('lat_lim'))


if __name__ == '__main__':
    output_dir  = os.path.join('output/dev_superdarn')
    gl.prep_output({0:output_dir},clear=False,php=False)

    lat_lims=(  36.,  46., 10./4)
    lon_lims=(-105., -85., 20./4)

    rd  = {}
#    rd['sDate']                 = datetime.datetime(2017,11,3,12)
#    rd['eDate']                 = datetime.datetime(2017,11,4)

    rd['sDate']                 = datetime.datetime(2017,11,3,1)
    rd['eDate']                 = datetime.datetime(2017,11,3,2)
    rd['rgc_lim']               = (0.,3000)
    rd['data_sources']          = [1,2,3]
    rd['reprocess_raw_data']    = False
    rd['filter_region']         = 'US'
    rd['filter_region_kind']    = 'mids'
    rd['output_dir']            = output_dir

    rd['plot_summary_line']     = False

    rd['band_MHz']              = 14
#    rd['xb_size_min']           = 5
#    rd['yb_size_km']            = 50.

    rd['xb_size_min']           = 2
    rd['yb_size_km']            = 25.

#    rd['ylim']                  = (0, 3000)
#    rd['yticks']                = np.arange(0,3500,500)
    rd['ylim']                  = (500.,2500.)
    rd['yticks']                = np.arange(500,3000,500)

    keo_ham = KeoHam(rd)
    import ipdb; ipdb.set_trace()
