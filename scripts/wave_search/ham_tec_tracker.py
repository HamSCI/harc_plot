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

        sDate = rd['sDate']
        eDate = rd['eDate']
        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)

        png_dir     = os.path.join(output_dir,date_str)
        gl.prep_output({0:png_dir},clear=True)
        self.run_dct['png_dir'] = png_dir

        # Load TEC Data
        self.tec_obj    = gps_tec_plot.TecPlotter(sDate)
        self.tec_obj.grid_data(dlat=1.,dlon=1.)
        gps_dates       = self.tec_obj.get_dates(sDate,eDate)

        for gps_date in gps_dates:
            print(gps_date)
            self.plot_timeseries(gps_date)
    
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

    def plot_timeseries(self,gps_date):
        rd                  = self.run_dct
        png_dir             = rd['png_dir']
        sDate               = rd['sDate']
        eDate               = rd['eDate']
        band_MHz            = rd['band_MHz']
        xb_size_min         = rd['xb_size_min']
        yb_size_km          = rd['yb_size_km']
        xlim                = rd['xlim']
        ylim                = rd['ylim']
        yticks              = rd['yticks']
        rgc_lim             = rd['rgc_lim']
        xkey                = rd['xkey']
        plot_summary_line   = rd.get('plot_summary_line',True)
        filter_region       = rd.get('filter_region','World')

        df                  = self.df

#        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
#        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
#        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
        gpsDate_str = gps_date.strftime('%Y%m%d.%H%M')
        fname       = '{!s}_ham_tec.png'.format(gpsDate_str)

        fig     = plt.figure(figsize=(20,10))

        ################################################################################ 
        # Plot Time Series ############################################################# 
        ax      = fig.add_subplot(3,1,1)
        ax_01   = ax

        log_z   = False
        vmax    = 40
        result  = self.time_series_ax(df,ax,vmax=vmax,log_z=log_z)

        cbar       = result['cbar']
        cax        = cbar.ax
        cax_pos    = list(cax.get_position().bounds)

        cax_pos[0] = 0.750 # X0
        cax_pos[1] = 0.654 # Y0
        cax_pos[2] = 0.150 # Width
        cax_pos[3] = 0.226 # Height
        cax.set_position(cax_pos)
        

        gps_hr  = gps_date.hour + gps_date.minute/60. + gps_date.second/3600.
        ax.axvline(gps_hr,lw=3,color='w',zorder=1500)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks)

        ax.set_xlabel('')
        ax.set_ylabel('Great Circle\nRange [km]')
        ax.set_title('{!s} MHz RBN, PSKReporter, and WSPRNet'.format(rd['band_MHz']))

        dfmt    = '%Y %b %d %H%M UT'
        title   = '{!s} -\n{!s}'.format(sDate.strftime(dfmt), eDate.strftime(dfmt))
        ax.set_title(title,loc='left',fontsize='medium')
        

        ################################################################################
        # Plot GPS TEC Map #############################################################
        cax         = fig.add_subplot(444)
        ax          = fig.add_subplot(3,1,(2,3),projection=ccrs.PlateCarree())
#        tec_plot    = self.tec_obj.plot_tec_ax(gps_date,zz=20,title=False,
#                        ax=ax,maplim_region=filter_region,cax=cax)

        tec_plot    = self.tec_obj.plot_tec_gridded_ax(gps_date,param='mean_dtecs',title=False,
                        ax=ax,maplim_region=filter_region,cax=cax)

        # Draw line showing wavefront and calculate wavelength and direction.
        lat_0,lon_0 = (44.0, -93.)
        lat_1,lon_1 = (30., -88.)

        gc_dist = gl.geopack.greatCircleDist(lat_0,lon_0,lat_1,lon_1) * (6371+250)
        gc_azm  = gl.geopack.greatCircleAzm(lat_0,lon_0,lat_1,lon_1)
        print('Wavelength: {:.0f} km'.format(gc_dist))
        print('Azimuth: {:.0f} deg'.format(gc_azm))
        ax.annotate("",xy=(lon_1,lat_1),xytext=(lon_0,lat_0),
                arrowprops={'arrowstyle':'->','lw':3,'ls':'-','mutation_scale':30},
                transform=ccrs.PlateCarree())

        ax.annotate("",xy=(lon_1,lat_1),xytext=(lon_0,lat_0),
                arrowprops={'arrowstyle':'|-|','lw':3,'ls':'-'},
                transform=ccrs.PlateCarree())

        lbl = '$\lambda = ${:.0f} km; $\phi = ${:.0f}$^\circ$'.format(gc_dist,gc_azm)
        ax.text(0.01,0.01,lbl,transform=ax.transAxes)


        ax_pos      = list(ax.get_position().bounds)
        ax_pos[0]   = 0.125 # X0
        ax_pos[1]   = 0.00  # Y0
        ax_pos[2]   = 0.59  # Width
        ax_pos[3]   = 0.59  # Height
        ax.set_position(ax_pos)

        cbar        = tec_plot['cbar']
        cax        = cbar.ax
        cax_pos    = list(cax.get_position().bounds)
        cax_pos[0] = 0.750 # X0
        cax_pos[1] = 0.000 # Y0
        cax_pos[2] = 0.0225 # Width
        cax_pos[3] = 0.590 # Height
        cax.set_position(cax_pos)

        dfmt    = '%Y %b %d %H%M UT'
#        title   = '{!s} - {!s}'.format(sDate.strftime(dfmt), eDate.strftime(dfmt))
        title   = '{!s}'.format(gps_date.strftime(dfmt))
        fig.text(0.425,0.93,title,fontdict={'weight':'bold','size':24},ha='center')

        fpath       = os.path.join(png_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)

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
        result      = data.plot.pcolormesh(x=xkey,y='dist_Km',ax=ax,vmin=0,vmax=vmax,
                cbar_kwargs={'aspect':5,'pad':0.08})

        title   = 'Bin Size:\n{!s} min x {!s} km'.format(rd['xb_size_min'],rd['yb_size_km'])
        ax.set_title(title,loc='right',fontsize='medium')

        # Calculate Derived Line
        sum_cnts    = data.sum('dist_Km').data
        avg_dist    = (data.dist_Km.data @ data.data.T) / sum_cnts

        if plot_summary_line:
            ax2     = ax.twinx()
            ax2.plot(data.ut_hrs,avg_dist,lw=2,color='w')
            ax2.set_ylim(0,3000)
            ax2.set_ylabel('Avg Dist\n[km]')

        ax.set_xlim(xlim)

        return {'data':data,'avg_dist':avg_dist,'cbar':result.colorbar}


if __name__ == '__main__':
    output_dir  = os.path.join('output/ham_tec_tracker')
    gl.prep_output({0:output_dir},clear=False,php=False)

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

    keo_ham = KeoHam(rd)
    import ipdb; ipdb.set_trace()
