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

class KeoGrid(object):
    def __init__(self,lat_lims=(36.,46.,0.5),lon_lims=(-105.,-85.,1.0)):
        """
        Create a grid for creating both latitudinal and longitudinal keograms.
        lat_lims: (lat_min,lat_max,dlat)
        lon_lims: (lon_min,lon_max,dlon)
        """

        lats    = np.arange(*lat_lims)
        lons    = np.arange(*lon_lims)

        # Define grid that varies with latitude.
        lat_grid = []
        for lat in lats:
            dct = {}
            dct['lat_lim']  = (lat,lat+lat_lims[2])
            dct['lon_lim']  = lon_lims[:2]
            dct['lavg']     = np.mean(dct['lat_lim'])
            lat_grid.append(dct)

        # Define grid that varies with longitude.
        lon_grid    = []
        for lon in lons:
            dct = {}
            dct['lon_lim']  = (lon,lon+lon_lims[2])
            dct['lat_lim']  = lat_lims[:2]
            dct['lavg']     = np.mean(dct['lon_lim'])
            lon_grid.append(dct)
        
        self.lims           = {}
        self.lims['lat']    = lat_lims
        self.lims['lon']    = lon_lims
        
        self.grid           = {}
        self.grid['lat']    = lat_grid
        self.grid['lon']    = lon_grid

        self.label           = {}
        self.label['lat']    = 'Latitude'
        self.label['lon']    = 'Longitude'

    def plot_maps(self,maplim_region='US',
                  output_dir='output/map_check',fname='map_grid.png'):
        """
        Plot maps of the Keogram Grid.
        """
        gl.prep_output({0:output_dir},clear=False,php=False)

        fig = plt.figure(figsize=(20,22))

        plot_dicts = []
        pdct = {}
        pdct['maplim_region']   = maplim_region
        pdct['lkey']            = 'lat'
        plot_dicts.append(pdct)

        pdct = {}
        pdct['maplim_region']   = maplim_region
        pdct['lkey']            = 'lon'
        plot_dicts.append(pdct)

        projection  = ccrs.PlateCarree()

        plt_inx = 1
        for pdct in plot_dicts:
            ax  = fig.add_subplot(2,1,plt_inx, projection=projection)
            pdct.update(ax=ax)
            self.plot_maps_ax(**pdct)
            plt_inx += 1

        fig.tight_layout()

        fpath   = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)

    def plot_maps_ax(self,lkey,ax,maplim_region='US'):

        grid        = self.grid[lkey]
        title       = self.label[lkey] + ' Grid'
        lat_lims    = self.lims['lat']
        lon_lims    = self.lims['lon']


        prop_cycle  = plt.rcParams['axes.prop_cycle']
        colors      = prop_cycle.by_key()['color']

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

        ax.coastlines(lw=2)
        ax.gridlines(draw_labels=True)
        ax.add_feature(cartopy.feature.BORDERS,lw=2)

        for ginx,rgn in enumerate(grid):
            x0  = rgn['lon_lim'][0]
            y0  = rgn['lat_lim'][0]
            ww  = rgn['lon_lim'][1] - x0
            hh  = rgn['lat_lim'][1] - y0

            clr = colors[ginx % len(colors)]
            p   = mpl.patches.Rectangle((x0,y0),ww,hh,ec=clr,fc=clr,zorder=500)
            ax.add_patch(p)

        lat_avg = np.mean(lat_lims[:2])
        lon_avg = np.mean(lon_lims[:2])

        ax.plot(lon_lims[:2],[lat_avg,lat_avg],lw=5,color='k',zorder=600)
        ew_km = geopack.greatCircleDist(lat_avg,lon_lims[0],lat_avg,lon_lims[1]) * 6371
        ax.text(lon_avg,lat_lims[0],'{:0.0f} km'.format(ew_km),ha='center',va='top',
                fontdict={'size':22,'weight':'bold'})

        ax.plot([lon_avg,lon_avg],lat_lims[:2],lw=3,color='k',zorder=600)
        ns_km = geopack.greatCircleDist(lat_lims[0],lon_avg,lat_lims[1],lon_avg) * 6371
        ax.text(lon_lims[0],lat_avg,'{:0.0f} km'.format(ns_km),ha='right',va='center',
                rotation=90.,fontdict={'size':22,'weight':'bold'})

        if title is not None:
            ax.set_title(title)

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
        rd['output_dir']         = rd.get('output_dir')
        rd['data_sources']       = rd.get('data_sources',[1,2,3])
        rd['reprocess_raw_data'] = rd.get('reprocess_raw_data',True)

#        rd['band']               = band_obj.band_dict[rd['band_MHz']]['meters']
        self.run_dct             = rd

        self.load_data()
        self.plot_timeseries()
        self.plot_stackplot_and_keogram()
    
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

                dft['band_MHz'] = np.floor(dft['freq']/1000.).astype(np.int)

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
        rgc_lim             = rd['rgc_lim']
        xkey                = rd['xkey']
        plot_summary_line   = rd.get('plot_summary_line',True)

        df                  = self.df

        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
        fname       = '{!s}_timeseries.png'.format(date_str)

        fig     = plt.figure(figsize=(35,10))
        col_0       = 0
        col_0_span  = 30
        col_1       = 36
        col_1_span  = 65
        nrows       = 1
        ncols       = 100

        ################################################################################
        # Plot Map #####################################################################
        ax = plt.subplot2grid((nrows,ncols),(0,col_0),
                projection=ccrs.PlateCarree(),colspan=col_0_span)
        self.map_ax(df,ax)

        ################################################################################ 
        # Plot Time Series ############################################################# 
        ax      = plt.subplot2grid((nrows,ncols),(0,col_1),colspan=col_1_span)
        result  = self.time_series_ax(df,ax,vmax=None,log_z=False)

        ax.set_xlim(xlim)

        fig.text(0.5,1.0,fname,fontdict={'weight':'bold','size':24},ha='center')

        fpath       = os.path.join(output_dir,fname)
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

#        text = 'Radio Spots (N = {!s})'.format(map_n)
        text = 'N = {!s}'.format(map_n)
        ax.text(0.5,-0.125,text,
                ha='center',transform=ax.transAxes,fontdict=fdict)

        plot_rgn    = gl.regions.get(filter_region)
#        ax.set_xlim(plot_rgn.get('lon_lim'))
#        ax.set_ylim(plot_rgn.get('lat_lim'))
#
#        ax.set_xlim(-130.,-60.)
#        ax.set_ylim(20.,55.)

        ax.set_xlim(-100.,-60.)
        ax.set_ylim(20.,55.)

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

        vmin = 0
        vmax = 5
#        vmax = None
        # Plot the Pcolormesh
#        result      = data.plot.pcolormesh(x=xkey,y='dist_Km',ax=ax,vmin=vmin,vmax=vmax,cbar_kwargs={'pad':0.08})a
#        result      = data.plot.contour(x=xkey,y='dist_Km',ax=ax,vmin=vmin,vmax=vmax,levels=5,zorder=1000)
        cbar_kwargs={'fraction':0.150,'aspect':10,'pad':0.01}
        result      = data.plot.contourf(x=xkey,y='dist_Km',ax=ax,vmin=vmin,vmax=vmax,cbar_kwargs=cbar_kwargs,levels=5)

        ax.set_ylim(750,1750)

        # Calculate Derived Line
        sum_cnts    = data.sum('dist_Km').data
        avg_dist    = (data.dist_Km.data @ data.data.T) / sum_cnts

        if plot_summary_line:
            ax2     = ax.twinx()
            ax2.plot(data.ut_hrs,avg_dist,lw=2,color='w')
            ax2.set_ylim(0,3000)
            ax2.set_ylabel('Avg Dist\n[km]')

        ax.set_xlim(xlim)

        return {'data':data,'avg_dist':avg_dist}

    def plot_stackplot_and_keogram(self):
        rd                  = self.run_dct
        keo_grid            = rd['keo_grid']
        output_dir          = rd['output_dir']
        sDate               = rd['sDate']
        eDate               = rd['eDate']
        band_MHz            = rd['band_MHz']
        xb_size_min         = rd['xb_size_min']
        yb_size_km          = rd['yb_size_km']
        rgc_lim             = rd['rgc_lim']
        xkey                = rd['xkey']
        xlim                = rd['xlim']
        filter_region       = rd.get('filter_region','World')
        plot_summary_line   = rd.get('plot_summary_line',True)
        stackplot_vmax      = rd.get('stackplot_vmax')

        df                  = self.df

        # Set up stackplot. ####################
        lkeys   = ['lat', 'lon']

        nlats   = len(keo_grid.grid['lat'])
        nlons   = len(keo_grid.grid['lon'])

        nrows   = nlats + nlons + 1

        ncols   = 100

        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
        fname       = '{!s}_stackplot.png'.format(date_str)

        fig         = plt.figure(figsize=(np.array([35,nrows*3.0])*1.00))
        col_0       = 0
        col_0_span  = 30
        col_1       = 36
        col_1_span  = 65

        keo     = {} # Create dictionary to store keograms.
        row_inx    = 0
        for linx,lkey in enumerate(lkeys):
            grid    = keo_grid.grid[lkey]
            for ginx,rgn in enumerate(grid[::-1]):

                lat_0   = rgn['lat_lim'][0]
                lat_1   = rgn['lat_lim'][1]

                lon_0   = rgn['lon_lim'][0]
                lon_1   = rgn['lon_lim'][1]

                dft     = df.copy()

                tf      = np.logical_and(dft['md_lat'] >= lat_0, dft['md_lat'] < lat_1)
                dft     = dft[tf].copy()

                tf      = np.logical_and(dft['md_long'] >= lon_0, dft['md_long'] < lon_1)
                dft     = dft[tf].copy()

                ################################################################################
                # Plot Map #####################################################################

                ax = plt.subplot2grid((nrows,ncols),(row_inx,col_0),
                        projection=ccrs.PlateCarree(),colspan=col_0_span)
                self.map_ax(dft,ax)

                ################################################################################ 
                # Plot Time Series ############################################################# 
                ax      = plt.subplot2grid((nrows,ncols),(row_inx,col_1),colspan=col_1_span)
                result  = self.time_series_ax(dft,ax,vmax=stackplot_vmax,log_z=False)

                data        = result['data']
                avg_dist    = result['avg_dist']

                if lkey not in keo:
                    keo[lkey]   = np.zeros((len(data.ut_hrs),len(grid)))
                    keo_y       = []

                    keo_attrs   = {}
                    keo_attrs['name']           = 'avg_dist_km'
                    keo_attrs['band_MHz']       = band_MHz
                    keo_attrs['band_name']      = band_obj.band_dict[band_MHz]['name']
                    keo_attrs['band_freq_name'] = band_obj.band_dict[band_MHz]['freq_name']
                    keo_attrs['sTime']          = sDate
                    keo_attrs['xkey']           = xkey
                    keo_attrs['ykey']           = lkey

                keo[lkey][:,ginx] = avg_dist
                keo_y.append(rgn['lavg'])

                if lkey == 'lat':
                    lavg    = np.mean(rgn['lat_lim'])
                    ylabel  = '{:0.1f}'.format(lavg)+'$^{\circ}$N'
                elif lkey == 'lon':
                    lavg    = np.mean(rgn['lon_lim'])
                    ylabel  = '{:0.1f}'.format(lavg)+'$^{\circ}$E'
                
                ax.set_ylabel('Rgc [km]')
                ax.text(-0.110,0.5,ylabel,fontdict={'weight':'bold','size':28},transform=ax.transAxes,
                        rotation=90.,va='center')
                ax.set_title('')

#                if ginx != nrows-1:
                if row_inx != (nlats-1) and row_inx != (nlats+nlons):
#                    ax.set_xticklabels([])
                    ax.set_xlabel('')
                else:
                    ax.set_xlabel('Time [UT]')

                data_sources_dct = {1:'WSPRNet',2:'RBN',3:'PSKReporter'}
                data_sources    = []
                for data_source in rd['data_sources']:
                    data_sources.append(data_sources_dct[data_source])

                if row_inx == 0:
                    text    = []
                    text.append('Latitudinal Slices')
                    text.append(sDate.strftime('%Y %b %d') + ' - WSPRNet, RBN, and PSKReporter')
                    ax.text(0.50,1.375,text[0],fontdict={'weight':'bold','size':36},ha='center',transform=ax.transAxes)
                    ax.text(0.50,1.10,text[1],fontdict={'weight':'normal','size':34},ha='center',transform=ax.transAxes)

                if row_inx == nlats+1:
                    text    = []
                    text.append('Longitudinal Slices')
                    text.append(sDate.strftime('%Y %b %d') + ' - WSPRNet, RBN, and PSKReporter')

                    ax.text(0.50,1.375,text[0],fontdict={'weight':'bold','size':36},ha='center',transform=ax.transAxes)
                    ax.text(0.50,1.10,text[1],fontdict={'weight':'normal','size':34},ha='center',transform=ax.transAxes)

                row_inx += 1


            # Convert Keogram array into Xarray
            keo_xkey            = keo_attrs['xkey']
            keo_ykey            = keo_attrs['ykey']

            keo_crds = {}
            keo_crds['ut_sTime']    = keo_attrs['sTime']
            keo_crds['band_MHz']    = keo_attrs['band_MHz']
            keo_crds[keo_xkey]      = data.ut_hrs.data
            keo_crds[keo_ykey]      = np.array(keo_y)
        
            keo[lkey]               = xr.DataArray(keo[lkey],keo_crds,attrs=keo_attrs,dims=[keo_xkey,keo_ykey]).sortby(lkey)
            self.keo                = keo

            row_inx += 1

        fpath       = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)


        # Plot keogram
        fig = plt.figure(figsize=(40,10))
        for linx, lkey in enumerate(['lat','lon']):
            ax  = fig.add_subplot(1,2,linx+1)

            this_keo    = keo[lkey]
            result      = this_keo.plot.pcolormesh(x=xkey,y=lkey,ax=ax)
#            result      = this_keo.plot.contourf(x=xkey,y=lkey,ax=ax)
            ax.set_xlim(xlim)

        fig.tight_layout()
        fname   = 'keogram.png'
        fpath   = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')

if __name__ == '__main__':
    output_dir  = os.path.join('output/wave_search_2017NOV03')
    gl.prep_output({0:output_dir},clear=False,php=False)

#    def __init__(self,lat_lims=(36.,46.,0.5),lon_lims=(-105.,-85.,1.0)):

# Defaults
#    lat_lims=(  36.,   46., 0.5)
#    lon_lims=(-105., -85., 1.0)

#    lat_lims=(  36.,  46., 10./4)
#    lon_lims=(-105., -85., 20./4)

##    lat_lims=( 36.,  46., 1.0)
##    lon_lims=(-90., -70., 2.0)

    lat_lims=( 37.,  44., 1.0)
    lon_lims=(-86., -74., 2.0)

#    lat_lims=( 38.,  42., 1.0)
#    lon_lims=(-86., -78., 2.0)
    keo_grid    = KeoGrid(lat_lims=lat_lims,lon_lims=lon_lims)

    ################################################
    # Explicitly choose and order lat, lon values. #
    ################################################
    lats    = [ 38.5,  39.5,  40.5,  41.5]
    lons    = [-85.0, -83.0, -81.0, -79.0]
    lons    = lons[::-1]

    keo_lat = []
    for lat in lats:
        for lat_dct in keo_grid.grid['lat']:
            if lat_dct['lavg'] == lat:
                keo_lat.append(lat_dct)
    keo_grid.grid['lat'] = keo_lat

    keo_lon = []
    for lon in lons:
        for lon_dct in keo_grid.grid['lon']:
            if lon_dct['lavg'] == lon:
                keo_lon.append(lon_dct)
    keo_grid.grid['lon'] = keo_lon

    ################################################

#    keo_grid    = KeoGrid()
    keo_grid.plot_maps(output_dir=output_dir)

    rd  = {}
    rd['sDate']                 = datetime.datetime(2017,11,3,12)
    rd['eDate']                 = datetime.datetime(2017,11,3,18)
#    rd['eDate']                 = datetime.datetime(2017,11,4)
    rd['rgc_lim']               = (0.,3000)
    rd['xlim']                  = (12,18)
    rd['data_sources']          = [1,2,3]
#    rd['data_sources']          = [1,2]
    rd['reprocess_raw_data']    = False
    rd['filter_region']         = 'US'
    rd['filter_region_kind']    = 'mids'
    rd['output_dir']            = output_dir
    rd['keo_grid']              = keo_grid

    rd['plot_summary_line']     = False

    rd['band_MHz']              = 14
#    rd['xb_size_min']           = 2.
#    rd['yb_size_km']            = 25.

#    rd['stackplot_vmax']        = 5
    rd['xb_size_min']           = 5
    rd['yb_size_km']            = 50.

    keo_ham = KeoHam(rd)
    import ipdb; ipdb.set_trace()
