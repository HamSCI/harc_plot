#!/usr/bin/env python3
import os
import datetime
import numpy as np

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

import gps_tec_plot
import harc_plot.gen_lib as gl

output_dir  = os.path.join('output','tec_keogram')
gl.prep_output({0:output_dir},clear=True,php=False)

col_0           = 0
col_0_span      = 30
col_1           = 35
col_1_span      = 65
nrows           = 2
ncols           = 100
filter_region   = 'US'
lbl_size        = 18

def map_ax(row,letter):
    ax = plt.subplot2grid((nrows,ncols),(row,col_0),projection=ccrs.PlateCarree(),colspan=col_0_span)
    ax.add_feature(cartopy.feature.COASTLINE)
    ax.add_feature(cartopy.feature.BORDERS, linestyle=':')

    plot_rgn      = gl.regions.get(filter_region)
    ax.set_xlim(plot_rgn.get('lon_lim'))
    ax.set_ylim(plot_rgn.get('lat_lim'))

    ax.set_title('({!s})'.format(letter),{'size':lbl_size},'left')
    return ax

def ts_ax(row,letter):
    ax      = plt.subplot2grid((nrows,ncols),(row,col_1),colspan=col_1_span)
    ax.set_title('({!s})'.format(letter),{'size':lbl_size},'left')
    return ax

def draw_box(lat_0,lat_1,lon_0,lon_1,ax):
    x0  = lon_0
    y0  = lat_0
    ww  = lon_1 - x0
    hh  = lat_1 - y0
    
    p   = mpl.patches.Rectangle((x0,y0),ww,hh,fill=False,zorder=500)
    ax.add_patch(p)


lat_lim = ( 37., 44.)
dlat    = 1.0

lon_lim = (-88., -74.)
dlon    = 2.0

sDate = datetime.datetime(2017,11,3,12)
eDate = datetime.datetime(2017,11,4)
tec_obj = gps_tec_plot.TecPlotter(sDate)

fig     = plt.figure(figsize=(35,10))

# Latitudinal Keogram ##########################################################
ax      = map_ax(0,'a')
ax_00   = ax
#draw_box(*lat_lim,*lon_lim,ax)

ax      =  ts_ax(0,'b')
ax_01   = ax
result  = tec_obj.plot_keogram(ax,*lat_lim,*lon_lim,sDate,eDate,keotype='lat',map_ax=ax_00)
ax.set_title('')
ax.set_title('(b) {!s}'.format(result['title']),{'size':lbl_size},'left')

# Longitudinal Keogram #########################################################
ax      = map_ax(1,'c')
ax_10   = ax
#draw_box(*lat_lim,*lon_lim,ax)

ax      =  ts_ax(1,'d')
ax_11   = ax
result  = tec_obj.plot_keogram(ax,*lat_lim,*lon_lim,sDate,eDate,keotype='lon',map_ax=ax_10)
ax.set_title('')
ax.set_title('(d) {!s}'.format(result['title']),{'size':lbl_size},'left')

fpath   = os.path.join(output_dir,'tec_keogram.png')
fig.savefig(fpath,bbox_inches='tight')
plt.close(fig)

import ipdb; ipdb.set_trace()
