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
            lat_grid.append(dct)

        # Define grid that varies with longitude.
        lon_grid    = []
        for lon in lons:
            dct = {}
            dct['lon_lim']  = (lon,lon+lon_lims[2])
            dct['lat_lim']  = lat_lims[:2]
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
    xkeys               = run_dct['xkeys']
    rgc_lim             = run_dct['rgc_lim']
    filter_region       = run_dct['filter_region']
    filter_region_kind  = run_dct['filter_region_kind']
    output_dir          = run_dct.get('output_dir')
    data_sources        = run_dct.get('data_sources',[1,2])
    reprocess_raw_data  = run_dct.get('reprocess_raw_data',True)

    xb_size_min         = run_dct['xb_size_min']
    yb_size_km          = run_dct['yb_size_km']
    band                = run_dct['band']
    keo_grid            = run_dct['keo_grid']

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

    # Set up stackplot. ####################
    lkeys   = ['lat', 'lon']
    for lkey in lkeys:
        grid    = keo_grid.grid[lkey]
        nrows   = len(grid) # Number of rows in figure
        ncols   = 1
        ax_inx  = 1

        fig     = plt.figure(figsize=(20,nrows*2))
        for ginx,rgn in enumerate(grid[::-1]):
            ax  = fig.add_subplot(nrows,ncols,ax_inx)

            if lkey == 'lat':
                lavg    = np.mean(rgn['lat_lim'])
                ylabel  = '{:0.1f}'.format(lavg)+'$^{\circ}$N'
            elif lkey == 'lon':
                lavg    = np.mean(rgn['lon_lim'])
                ylabel  = '{:0.1f}'.format(lavg)+'$^{\circ}$E'
            
            ax.set_ylabel(ylabel)

            ax.set_xlim(sDate,eDate)
            if ginx != nrows-1:
                ax.set_xticklabels([])
            ax_inx += 1

        fig.tight_layout()
        sDate_str   = sDate.strftime('%Y%m%d.%H%M')
        eDate_str   = eDate.strftime('%Y%m%d.%H%M')
        date_str    = '{!s}-{!s}'.format(sDate_str,eDate_str)
        fname       = '{!s}_{!s}_stackplot.png'.format(date_str,lkey)
        fpath       = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')
        plt.close(fig)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    output_dir  = os.path.join('output/wave_search_2017NOV03')
    gl.prep_output({0:output_dir},clear=False,php=False)

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

    rd['band']                  = 20
    rd['xb_size_min']           = 2.
    rd['yb_size_km']            = 25.

    main(rd)
