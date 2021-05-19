#!/usr/bin/env python3
import os
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs

import numpy as np
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from harc_plot import gen_lib as gl

goess   = OrderedDict()
tmp     = {}
key     = 'GOES-EAST'
tmp['lon']      = -75
tmp['label']    = key
tmp['color']    = 'blue'
goess[key]      = tmp

tmp     = {}
key     = 'GOES-WEST'
tmp['lon']      = -135
tmp['label']    = key
tmp['color']    = 'red'
goess[key]      = tmp

def plot_map(maplim_region='World',output_dir='output',plot_goes=False,box=None):
    projection  = ccrs.PlateCarree()

    fig = plt.figure(figsize=(10,8))
    ax  = fig.add_subplot(1,1,1, projection=projection)

    ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
    ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

    ax.coastlines()
    ax.gridlines(draw_labels=True)

    if plot_goes:
        for key,item in goess.items():
            lon     = item.get('lon')
            label   = item.get('label',str(key))
            color   = item.get('color','blue')

            ax.axvline(lon,ls='--',label=label,color=color)
        ax.legend(loc='lower right')

    if box is not None:
        rgn = gl.regions.get(box)
        x0  = rgn['lon_lim'][0]
        y0  = rgn['lat_lim'][0]
        ww  = rgn['lon_lim'][1] - x0
        hh  = rgn['lat_lim'][1] - y0
        
        p   = matplotlib.patches.Rectangle((x0,y0),ww,hh,fill=False,zorder=500)
        ax.add_patch(p)

    fname   = 'map-{!s}.png'.format(maplim_region)
    fpath   = os.path.join(output_dir,fname)
    fig.savefig(fpath,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    output_dir  = 'output/galleries/map_check'
    gl.prep_output({0:output_dir},clear=False,php=False)


    run_dcts = []
    rd = {}
#    rd['maplim_region'] = 'US'
    rd['box']           = 'NH'
    rd['plot_goes']     = True
    rd['output_dir']    = output_dir
    run_dcts.append(rd)

    for rd in run_dcts:
        plot_map(**rd)
