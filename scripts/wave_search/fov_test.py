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
from harc_plot.timeutils import daterange, strip_time
from harc_plot import geopack

import pydarnio
import pydarn


fig = plt.figure()

sDate       = datetime.datetime(2017,11,3,12)

radar       = 'bks'
beam        = 13
hdw_data    = pydarn.read_hdw_file(radar,sDate)

ax  = fig.add_subplot(111,projection=ccrs.PlateCarree())

ax.add_feature(cartopy.feature.COASTLINE,color='0.6')
ax.add_feature(cartopy.feature.BORDERS,color='0.6', linestyle=':')
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

plot_rgn    = gl.regions.get('US')
ax.set_xlim(plot_rgn.get('lon_lim'))
ax.set_ylim(plot_rgn.get('lat_lim'))

#ax.set_xlim(-130.,-60.)
#ax.set_ylim(20.,75.)

fig.savefig('output/swo2r/fov.png',bbox_inches='tight')

import ipdb; ipdb.set_trace()
