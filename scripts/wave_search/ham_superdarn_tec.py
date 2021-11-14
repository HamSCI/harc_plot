#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
# Due to Pandas bug, run with MPLv3.3.2
# conda activate mpl3.3.2

import os
import datetime
import itertools
import glob
import bz2
import tqdm

import pickle as pkl

import numpy as np
import pandas as pd
import xarray as xr
import scipy as sp

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

import pydarnio
import pydarn

import gps_tec_plot

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

def adjust_map(ax_0,ax_1):
    """
    Line up axis locations of the little maps.
    ax_0: map to be adjusted
    ax_1: map to line things up to
    """

    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[0] = ax_1_pos[0]
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

#def load_fitacf(sTime,eTime,radar,data_dir='data/superdarn-bas',fit_sfx='fitacf3'):
def load_fitacf(sTime,eTime,radar,data_dir='superdarn_data',fit_sfx='fitacf'):
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

def load_psk(date_str,rgc_lim=None,filter_region=None,filter_region_kind='mids',**kwargs):
    data_dir    = 'data/pskreporter'
    fnames = []

    if   date_str == '2017-11-03':
        fnames.append('PSK_data_20171103_p1.csv.bz2')
        fnames.append('PSK_data_20171103_p2.csv.bz2')
        fnames.append('PSK_data_20171103_p3.csv.bz2')
    elif date_str == '2017-05-16':
        fnames.append('PSK_data_20170516.csv.bz2')
    else:
        return

    grid_prsn   = 6

    df  = pd.DataFrame()
    for fname in fnames:
        fpath   = os.path.join(data_dir,fname)
        print(fpath)

        dft = pd.read_csv(fpath)
        df  = df.append(dft,ignore_index=True)

    df['occurred'] = pd.to_datetime(df['flowStartSeconds'],unit='s')
    df = df.rename(columns={'frequency':'freq','senderLocator':'tx_grid','receiverLocator':'rx_grid'})
    df = df[['freq','occurred','tx_grid','rx_grid']]
    df = df.copy()
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

def gps_time_series(key='gtidus',hr_lim = (12,24)):
    """
    Load in and process pickle file with GPS dTEC time series information.
    """
    fpath = 'data/gps_tec_haystack/20171103-20171104_gtidus_mids.p'
    with open(fpath,'rb') as fl:
        tec_ts  = pkl.load(fl)


    # Unpack data into lists.
    dts      = []
    dt_hours = []
    gtid     = []
    for tec in tec_ts:
        dts.append(tec['gps_date'])
        dt_hours.append(tec['gps_hour'])
        gtid.append(tec[key])

    # Remove the last datapoint in case the last time stamp
    # goes back to 0.
    dts         = dts[:-1]
    dt_hours    = dt_hours[:-1]
    gtid        = gtid[:-1]

    # Convert everything to numpy arrays.
    dts         = np.array(dts)
    dt_hours    = np.array(dt_hours)
    gtid        = np.array(gtid)

    # Select only the hours specified.
    tf = np.logical_and(dt_hours >= hr_lim[0], dt_hours < hr_lim[1])
    dts         = dts[tf]
    dt_hours    = dt_hours[tf]
    gtid        = gtid[tf]

    # Calculate the sampling period and frequency.
    Ts = np.diff(dts)[0].total_seconds()
    fs = 1/Ts

    # Compute linear detrended dTEC data.
    regress = sp.stats.linregress(dt_hours,gtid)
    line    = regress.slope*dt_hours + regress.intercept
    gtid_detrend = gtid - line

    # Hanning Window GPS dTEC data.
    han_win         = np.hanning(len(gtid_detrend))
    gtid_han        = han_win*gtid_detrend

    # Compute the FFT.
    X0 = np.fft.fft(gtid_han)*Ts*2

    # Reorder values so they go in frequency order from negative to positive.
    X0 = np.fft.fftshift(X0)

    f_vec = np.fft.fftfreq(len(X0),Ts)
    f_vec = np.fft.fftshift(f_vec)

    # Take positive frequencies only.
    tf    = f_vec >= 0
    X0    = X0[tf]
    f_vec = f_vec[tf]

    # Compute Period Vector in Hours
    T_hr_vec = 1./(f_vec*3600.)

    # Save important variables to dictionary and return.
    result  = {}
    result['dts']                   = dts
    result['dt_hours']              = dt_hours
    result['gtid']                  = gtid
    result['gtid_detrend']          = gtid_detrend
    result['f_vec']                 = f_vec
    result['T_hr_vec']              = T_hr_vec
    result['X0']                    = X0

    return result

def calc_histogram(frame,attrs):
    xkey    = attrs['xkey']
    xlim    = attrs['xlim']
    dx      = attrs['dx']
    ykey    = attrs['ykey']
    ylim    = attrs['ylim']
    dy      = attrs['dy']

    xbins   = gl.get_bins(xlim,dx)
    ybins   = gl.get_bins(ylim,dy)

    if len(frame) > 2:
       hist, xb, yb = np.histogram2d(frame[xkey], frame[ykey], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    crds    = {}
    crds['ut_sTime']    = attrs['sTime']
    crds['band_MHz']    = attrs['band_MHz']
    crds[xkey]          = xb[:-1]
    crds[ykey]          = yb[:-1]
    
    attrs   = attrs.copy()
    for key,val in attrs.items():
        attrs[key] = str(val)
    da = xr.DataArray(hist,crds,attrs=attrs,dims=[xkey,ykey])
    return da 

def calc_sinusoid(t0,t1,A,DC,T0,phi):
    """
    Calculate a sinusoid to put on top of data.
    """
    f0  = 1/T0
    tt  = np.arange(t0,t1,0.01)
    yy  = A*np.sin(2.*np.pi*f0*(tt-t0+phi)) + DC
    return (tt,yy)

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
        rd  = run_dct
#        rd['sDate']              = rd['sDate']
#        rd['eDate']              = rd['eDate']
#        rd['rgc_lim']            = rd['rgc_lim']
#        rd['filter_region']      = rd['filter_region']
#        rd['filter_region_kind'] = rd['filter_region_kind']
#        rd['xb_size_min']        = rd['xb_size_min']
#        rd['yb_size_km']         = rd['yb_size_km']
#        rd['band_MHz']           = rd['band_MHz']
#        rd['keo_grid']           = rd['keo_grid']

        rd['xkey']               = rd.get('xkey','ut_hrs')
        rd['xlim']               = rd.get('xlim',(12,24))
        rd['ylim']               = rd.get('ylim',None)
        rd['yticks']             = rd.get('yticks',None)
        rd['output_dir']         = rd.get('output_dir')
        rd['data_sources']       = rd.get('data_sources',[1,2,3])
        rd['reprocess_raw_data'] = rd.get('reprocess_raw_data',True)

#        rd['band']               = band_obj.band_dict[rd['band_MHz']]['meters']
        self.run_dct             = rd

        self.load_data()
        self.plot_timeseries()
    
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
        xkey                = rd['xkey']
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

            dft[xkey]    = dt_inx*24 + dft[xkey]

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

        self.df = df

    def plot_timeseries(self):
        rd                  = self.run_dct
        output_dir          = rd['output_dir']
        sDate               = rd['sDate']
        eDate               = rd['eDate']
        band_MHz            = rd['band_MHz']
        xb_size_min         = rd['xb_size_min']
        yb_size_km          = rd['yb_size_km']
        xlim                = rd['xlim']
        ylim                = rd['ylim']
        yticks              = rd['yticks']
        rgc_lim             = rd['rgc_lim']
        filter_region       = rd['filter_region']
        xkey                = rd['xkey']
        ham_vmax            = rd.get('ham_vmax',None)
        ham_log_z           = rd.get('ham_log_z',False)
        plot_summary_line   = rd.get('plot_summary_line',True)

        df                  = self.df

        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
        fname       = '{!s}_ham_superdarn.png'.format(date_str)

        fig     = plt.figure(figsize=(35,20))
        col_0       = 0
        col_0_span  = 30
        col_1       = 35
        col_1_span  = 65
        nrows       = 4
        ncols       = 100

        ################################################################################
        # Plot Map #####################################################################
        ax = plt.subplot2grid((nrows,ncols),(0,col_0),
                projection=ccrs.PlateCarree(),colspan=col_0_span)
        ax_00       = ax

        lbl_size    = 24
        ax.set_title('(a)',{'size':lbl_size},'left')
        
        self.map_ax(df,ax)

        ################################################################################ 
        # Plot Time Series ############################################################# 
        ax      = plt.subplot2grid((nrows,ncols),(0,col_1),colspan=col_1_span)
        ax_01   = ax
        result  = self.time_series_ax(df,ax,vmax=ham_vmax,log_z=ham_log_z)

        ham_cbar        = result['cbar']
        ham_cax         = ham_cbar.ax
        ham_cax_pos     = list(ham_cax.get_position().bounds)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.set_xlabel('Time [UT]')
        ax.set_ylabel('Great Circle Range [km]')
#        ax.set_title('{!s} MHz RBN, PSKReporter, and WSPRNet'.format(rd['band_MHz']))
#        ax.set_title('(b)',{'size':lbl_size},'left')
        ax.set_title('')
        ax.set_title('(b) {!s} MHz RBN, PSKReporter, and WSPRNet'.format(rd['band_MHz']),loc='left')

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
        result  = pydarn.RTP.plot_range_time(fitacf, beam_num=beam, parameter='p_l', zmax=50, zmin=0, date_fmt='%H', colorbar_label='Power (dB)', cmap='viridis',ax=ax)
        sd_cbar = result[1]
        sd_cax  = sd_cbar.ax
        sd_cax_pos      = list(sd_cax.get_position().bounds)

        # Plot visually fitted sinusoid.
        sinDct  = {'t0':13.1+0.2,'t1':18.25+0.2,'A':200,
                   'DC':950,'T0':2.5,'phi':0.}
        tt,yy   = calc_sinusoid(**sinDct)
        s0      = datetime.datetime(sDate.year,sDate.month,sDate.day)
        ttd     = [s0+datetime.timedelta(hours=x) for x in tt]
        ax.plot(ttd,yy,ls=':',lw=4,color='k')

        ax.set_ylabel('Slant Range [km]')
        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.set_xlim(sDate,eDate)
        ax.set_xlabel('Time [UT]')
#        ax.set_title('{!s} SuperDARN Radar Beam {!s}'.format(radar.upper(),beam))
#        ax.set_title('(d)',{'size':lbl_size},'left')
        ax.set_title('(d) {!s} SuperDARN Radar Beam {!s}'.format(radar.upper(),beam),loc='left')

        ################################################################################
        # Plot GPS dTEC Map ############################################################ 
        ax = plt.subplot2grid((nrows,ncols),(2,col_0),
                projection=ccrs.PlateCarree(),colspan=col_0_span)
        ax_20       = ax

        self.tec_obj    = gps_tec_plot.TecPlotter(sDate)

        frame       = datetime.datetime(2017,11,3,13,43)
        gps_dates   = self.tec_obj.get_dates()
        gps_inx     = np.argmin(np.abs(np.array(gps_dates) - frame))
        gps_date    = gps_dates[gps_inx]

        tec_plot    = self.tec_obj.plot_tec_ax(gps_date,zz=20,title=False,
                        ax=ax,maplim_region=filter_region,
                        grid = False)

        x0  = -120.
        y0  = 30.
        ww  = -70. - x0
        hh  = 50. - y0
        p   = mpl.patches.Rectangle((x0,y0),ww,hh,fill=False,zorder=50000,color='k',lw=2)
        ax.add_patch(p)

        ax.set_title('GPS dTEC - {!s} UT'.format(gps_date.strftime('%H%M')))
        ax.set_title('(e)',{'size':lbl_size},'left')


        ################################################################################ 
        # Plot GPS dTEC Time Series Data ############################################### 
        gps_ts  = gps_time_series()

        ax      = plt.subplot2grid((nrows,ncols),(2,col_1),colspan=col_1_span)
        ax_21   = ax

        xx = gps_ts['dt_hours']
        yy = gps_ts['gtid']
        ax.plot(xx,yy)
        ax.set_xlim(xlim)
        ax.set_ylim(-0.1,0.1)
        ax.grid(True,ls=':')

        ax.set_xlabel('Time [UT]')
        ax.set_ylabel('Median dTEC')

        # Plot visually fitted sinusoid.
        sinDct  = {'t0':13.1,'t1':18.25,'A':0.05,
                   'DC':0,'T0':2.5,'phi':0.0}
        tt,yy   = calc_sinusoid(**sinDct)
        ax.plot(tt,yy,ls=':',lw=4,color='k')

#        ax.set_title('GPS dTEC Time Series')
#        ax.set_title('(f)',{'size':lbl_size},'left')
        ax.set_title('(f) GPS dTEC Time Series',loc='left')

        ################################################################################ 
        # Plot GPS dTEC Magnitude Spectrum ############################################# 
        ax      = plt.subplot2grid((nrows,ncols),(3,col_1),colspan=col_1_span)
        ax_31   = ax

        xx = gps_ts['T_hr_vec']
        yy = np.abs(gps_ts['X0'])
        ax.plot(xx,yy,marker='o')
        ax.grid(True,ls=':')
        ax.set_xlim(0,6)

        ax.set_ylabel('|FFT{Median dTEC}|')

        ax.set_xlabel('Period [hr]')
#        ax.set_title('GPS dTEC Magnitude Spectrum')
#        ax.set_title('(g)',{'size':lbl_size},'left')
        ax.set_title('(g) GPS dTEC Magnitude Spectrum',loc='left')

        ################################################################################ 
        # Cleanup Figure ############################################################### 

        adjust_map(ax_10,ax_00)
        adjust_map(ax_20,ax_00)

        gl.adjust_axes(ax_11,ax_01)
        gl.adjust_axes(ax_21,ax_01)
        gl.adjust_axes(ax_31,ax_01)

        dfmt    = '%Y %b %d %H%M UT'
        title   = '{!s} - {!s}'.format(sDate.strftime(dfmt), eDate.strftime(dfmt))
        fig.text(0.5,0.90,title,fontdict={'weight':'bold','size':24},ha='center')

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
        map_attrs['band_MHz']           = band_MHz
        map_attrs['band_name']          = band_obj.band_dict[band_MHz]['name']
        map_attrs['band_freq_name']     = band_obj.band_dict[band_MHz]['freq_name']
        map_attrs['sTime']              = sDate
        map_attrs['xkey']               = 'md_long'
        map_attrs['xlim']               = (-180,180)
        map_attrs['dx']                 = 1
        map_attrs['ykey']               = 'md_lat'
        map_attrs['ylim']               = (-90,90)
        map_attrs['dy']                 = 1
        map_data                        = calc_histogram(df,map_attrs)
        map_data.name                   = 'Counts'

        ax.coastlines(zorder=10,color='w')
        ax.plot(np.arange(10))
        tf          = map_data < 1
        map_n       = int(np.sum(map_data))
        map_data    = np.log10(map_data)
        map_data.values[tf] = 0
        map_data.name   = 'log({})'.format(map_data.name)
        map_data.plot.contourf(x=map_attrs['xkey'],y=map_attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
        ax.set_title('')
        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}

        txt     = 'Radio Spots (N = {!s})'.format(map_n)
        ax.set_title(txt)

#        ax.text(0.5,-0.1,txt,
#                ha='center',transform=ax.transAxes,fontdict=fdict)

        plot_rgn    = gl.regions.get(filter_region)
        ax.set_xlim(plot_rgn.get('lon_lim'))
        ax.set_ylim(plot_rgn.get('lat_lim'))

    def time_series_ax(self,df,ax,vmax=None,log_z=False):
        rd                  = self.run_dct
        sDate               = rd['sDate']
        band_MHz            = rd['band_MHz']
        xb_size_min         = rd['xb_size_min']
        yb_size_km          = rd['yb_size_km']
        rgc_lim             = rd['rgc_lim']
        xkey                = rd['xkey']
        xlim                = rd['xlim']
        plot_summary_line   = rd.get('plot_summary_line',True)

        attrs                       = {}
        attrs['band_MHz']           = band_MHz
        attrs['band_name']          = band_obj.band_dict[band_MHz]['name']
        attrs['band_freq_name']     = band_obj.band_dict[band_MHz]['freq_name']
        attrs['sTime']              = sDate
        attrs['xkey']               = xkey
        attrs['dx']                 = xb_size_min/60.
        attrs['xlim']               = (0,24)
        attrs['ykey']               = 'dist_Km'
        attrs['ylim']               = rgc_lim
        attrs['dy']                 = yb_size_km
        data                        = calc_histogram(df,attrs)
        data.name                   = 'Counts'

        if log_z:
            tf          = data < 1.
            data        = np.log10(data)
            data        = xr.where(tf,0,data)
            data.name   = 'log({})'.format(data.name)

        # Plot th1e Pcolormesh
#        cbar_kwargs={'pad':0.08}
        cbar_kwargs={'fraction':0.150,'aspect':10,'pad':0.01}
        result      = data.plot.pcolormesh(x=xkey,y='dist_Km',ax=ax,vmin=0,vmax=vmax,cbar_kwargs=cbar_kwargs)

        title   = 'Bin Size:\n{!s} min x {!s} km'.format(rd['xb_size_min'],rd['yb_size_km'])
        ax.set_title(title,loc='right',fontsize='large')

        # Calculate Derived Line
        sum_cnts    = data.sum('dist_Km').data
        avg_dist    = (data.dist_Km.data @ data.data.T) / sum_cnts

        if plot_summary_line:
            ax2     = ax.twinx()
            ax2.plot(data.ut_hrs,avg_dist,lw=2,color='w')
            ax2.set_ylim(0,3000)
            ax2.set_ylabel('Avg Dist\n[km]')

        ax.set_xlim(xlim)

        # Plot visually fitted sinusoid.
        sinDct  = {'t0':13.1,'t1':18.25,'A':200,
                   'DC':1050,'T0':2.5,'phi':0.0}
        tt,yy   = calc_sinusoid(**sinDct)
        ax.plot(tt,yy,ls=':',lw=4,color='w')

        return {'data':data,'avg_dist':avg_dist,'cbar':result.colorbar}


if __name__ == '__main__':
    output_dir  = os.path.join('output/ham_superdarn_tec')
    gl.prep_output({0:output_dir},clear=False,php=False)

    lat_lims=(  36.,  46., 10./4)
    lon_lims=(-105., -85., 20./4)

    rd  = {}
    rd['sDate']                 = datetime.datetime(2017,11,3,12)
    rd['eDate']                 = datetime.datetime(2017,11,4)

#    rd['sDate']                 = datetime.datetime(2017,5,16,12)
#    rd['eDate']                 = datetime.datetime(2017,5,17)

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

    rd['ham_log_z']             = False
    rd['ham_vmax']              = 30.

    keo_ham = KeoHam(rd)
