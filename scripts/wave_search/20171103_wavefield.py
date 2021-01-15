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

        ax.coastlines()
        ax.gridlines(draw_labels=True)
        ax.add_feature(cartopy.feature.BORDERS)

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

        loc_sources: list, i.e. ['P','Q']
            P: user Provided
            Q: QRZ.com or HAMCALL
            E: Estimated using prefix
        """

        # Get Variables from run_dct
        rd  = run_dct
#        rd['sDate']              = rd['sDate']
#        rd['eDate']              = rd['eDate']
#        rd['xkeys']              = rd['xkeys']
#        rd['rgc_lim']            = rd['rgc_lim']
#        rd['filter_region']      = rd['filter_region']
#        rd['filter_region_kind'] = rd['filter_region_kind']
#        rd['xb_size_min']        = rd['xb_size_min']
#        rd['yb_size_km']         = rd['yb_size_km']
#        rd['band_MHz']           = rd['band_MHz']
#        rd['keo_grid']           = rd['keo_grid']

        rd['xlim']               = rd.get('xlim',(12,24))
        rd['output_dir']         = rd.get('output_dir')
        rd['data_sources']       = rd.get('data_sources',[1,2])
        rd['reprocess_raw_data'] = rd.get('reprocess_raw_data',True)
        rd['band']               = band_obj.band_dict[rd['band_MHz']]['meters']
        self.run_dct             = rd

        self.load_data()
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
        xkeys               = rd['xkeys']
        band                = rd['band']

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

                if dft is None:
                    print('No data for {!s}'.format(ld_str))
                    continue

                dft.to_hdf(h5_path,h5_key,complib='bzip2',complevel=9)

            for xkey in xkeys:
                dft[xkey]    = dt_inx*24 + dft[xkey]

            df  = df.append(dft,ignore_index=True)

        df_0    = df.copy()

        # Select desired times.
        tf      = np.logical_and(df['occurred'] >= sDate, df['occurred'] < eDate)
        df      = df[tf].copy()

        # Select desired band.
        tf      = df['band'] == band
        df      = df[tf].copy()

        self.df = df

    def plot_stackplot_and_keogram(self):
        rd                  = self.run_dct
        keo_grid            = rd['keo_grid']
        output_dir          = rd['output_dir']
        sDate               = rd['sDate']
        eDate               = rd['eDate']
        band                = rd['band']
        band_MHz            = rd['band_MHz']
        xb_size_min         = rd['xb_size_min']
        yb_size_km          = rd['yb_size_km']
        xlim                = rd['xlim']
        rgc_lim             = rd['rgc_lim']

        df                  = self.df

        xkey                = 'ut_hrs'

        # Set up stackplot. ####################
        lkeys   = ['lat', 'lon']

        keo     = {} # Create dictionary to store keograms.
        for linx,lkey in enumerate(lkeys):
            grid    = keo_grid.grid[lkey]

            nrows   = len(grid) # Number of rows in figure
            ncols   = 1
            ax_inx  = 1

            sDate_str   = sDate.strftime('%Y%m%d.%H%M')
            eDate_str   = eDate.strftime('%Y%m%d.%H%M')
            date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
            fname       = '{!s}_{!s}_stackplot.png'.format(date_str,lkey)

            fig     = plt.figure(figsize=(20,nrows*2))
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

                ax  = fig.add_subplot(nrows,ncols,ax_inx)

                attrs                       = {}
    #            attrs['filter_region']      = filter_region
    #            attrs['filter_region_kind'] = filter_region_kind
                attrs['band_MHz']           = band_MHz
                attrs['band_name']          = band_obj.band_dict[band_MHz]['name']
                attrs['band_freq_name']     = band_obj.band_dict[band_MHz]['freq_name']
                attrs['sTime']              = sDate
                attrs['xkey']               = 'ut_hrs'
                attrs['dx']                 = xb_size_min/60.
                attrs['xlim']               = (0,24)
                attrs['ykey']               = 'dist_Km'
                attrs['ylim']               = rgc_lim
                attrs['dy']                 = yb_size_km
                data                        = calc_histogram(dft,attrs)
                data.name                   = 'Counts'

                log_z = False
                if log_z:
                    tf          = data < 1.
                    data        = np.log10(data)
                    data        = xr.where(tf,0,data)
                    data.name   = 'log({})'.format(data.name)

                # Plot the Pcolormesh
    #            result      = data.plot.contourf(x=xkey,y='dist_Km',ax=ax,levels=30,vmin=0)
                result      = data.plot.pcolormesh(x=xkey,y='dist_Km',ax=ax,vmin=0,cbar_kwargs={'pad':0.08})

                # Calculate Derived Line
                sum_cnts    = data.sum('dist_Km').data
                avg_dist    = (data.dist_Km.data @ data.data.T) / sum_cnts

                ax2     = ax.twinx()
                ax2.plot(data.ut_hrs,avg_dist,lw=2,color='w')
                ax2.set_ylim(0,3000)
                ax2.set_ylabel('Avg Dist\n[km]')

                if lkey not in keo:
                    keo[lkey]   = np.zeros((len(data.ut_hrs),len(grid)))
                    keo_y       = []

                    keo_attrs   = {}
                    keo_attrs['name']           = 'avg_dist_km'
                    keo_attrs['band_MHz']       = attrs['band_MHz']
                    keo_attrs['band_name']      = attrs['band_name']
                    keo_attrs['band_freq_name'] = attrs['band_freq_name']
                    keo_attrs['sTime']          = sDate
                    keo_attrs['xkey']           = 'ut_hrs'
                    keo_attrs['ykey']           = lkey

                keo[lkey][:,ginx] = avg_dist
                keo_y.append(rgn['lavg'])

                if lkey == 'lat':
                    lavg    = np.mean(rgn['lat_lim'])
                    ylabel  = '{:0.1f}'.format(lavg)+'$^{\circ}$N'
                elif lkey == 'lon':
                    lavg    = np.mean(rgn['lon_lim'])
                    ylabel  = '{:0.1f}'.format(lavg)+'$^{\circ}$E'
                
                ax.set_ylabel(ylabel)
                ax.set_xlabel('')
                ax.set_title('')

    #            ax.set_xlim(sDate,eDate)
                ax.set_xlim(xlim)
                if ginx != nrows-1:
                    ax.set_xticklabels([])
                ax_inx += 1

            fig.tight_layout()
            fig.text(0.5,1.0,fname,fontdict={'weight':'bold','size':36},ha='center')

            fpath       = os.path.join(output_dir,fname)
            fig.savefig(fpath,bbox_inches='tight')
            plt.close(fig)

            # Convert Keogram array into Xarray
            keo_xkey            = keo_attrs['xkey']
            keo_ykey            = keo_attrs['ykey']

            keo_crds = {}
            keo_crds['ut_sTime']    = keo_attrs['sTime']
            keo_crds['band_MHz']    = keo_attrs['band_MHz']
            keo_crds[keo_xkey]      = data.ut_hrs.data
            keo_crds[keo_ykey]      = np.array(keo_y)
        
            keo[lkey]               = xr.DataArray(keo[lkey],keo_crds,attrs=keo_attrs,dims=[keo_xkey,keo_ykey])
            self.keo                = keo

        # Plot keogram
        fig = plt.figure(figsize=(40,10))
        for linx, lkey in enumerate(['lat','lon']):
            ax  = fig.add_subplot(1,2,linx+1)

            this_keo    = keo[lkey]
            result      = this_keo.plot.pcolormesh(x='ut_hrs',y=lkey,ax=ax)
            ax.set_xlim(xlim)

        fig.tight_layout()
        fname   = 'keogram.png'
        fpath   = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')

if __name__ == '__main__':
    output_dir  = os.path.join('output/wave_search_2017NOV03')
    gl.prep_output({0:output_dir},clear=False,php=False)


#    lat_lims=( 36.,  46., 1.0)
#    lon_lims=(-90., -70., 2.0)
#    keo_grid    = KeoGrid(lat_lims=lat_lims,lon_lims=lon_lims)
    keo_grid    = KeoGrid()
    keo_grid.plot_maps(output_dir=output_dir)

    rd  = {}
    rd['sDate']                 = datetime.datetime(2017,11,3,12)
    rd['eDate']                 = datetime.datetime(2017,11,4)
    rd['rgc_lim']               = (0.,3000)
    rd['xkeys']                 = ['ut_hrs']
    rd['reprocess_raw_data']    = False
    rd['filter_region']         = 'US'
    rd['filter_region_kind']    = 'mids'
    rd['output_dir']            = output_dir
    rd['keo_grid']              = keo_grid

    rd['band_MHz']              = 14
#    rd['xb_size_min']           = 2.
#    rd['yb_size_km']            = 25.

    rd['xb_size_min']           = 5
    rd['yb_size_km']            = 75.

    keo_ham = KeoHam(rd)
    import ipdb; ipdb.set_trace()
