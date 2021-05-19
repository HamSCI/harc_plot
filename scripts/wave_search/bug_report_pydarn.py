#!/usr/bin/env python3
import os
import datetime
import glob
import bz2
import tqdm

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import cartopy
import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#import harc_plot
#import harc_plot.gen_lib as gl
#from harc_plot.timeutils import daterange, strip_time
#from harc_plot import geopack

import pydarnio
import pydarn

def load_fitacf(sTime,eTime,radar,data_dir='fitacf3',fit_sfx='fitacf3'):
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

################################################################################ 
# Plot SuperDARN Time Series ################################################### 

sDate       = datetime.datetime(2017,11,3)
eDate       = datetime.datetime(2017,11,4)
radar       = 'bks'
beam        = 13
fitacf      = load_fitacf(sDate,eDate,radar)

#fig = plt.figure(figsize=(12,6))
#ax  = fig.add_subplot(111)

fig         = plt.figure(figsize=(35,10))
col_0       = 0
col_0_span  = 30
col_1       = 35
col_1_span  = 65
nrows       = 2
ncols       = 100

ax = plt.subplot2grid((nrows,ncols),(1,col_0),
        projection=ccrs.PlateCarree(),colspan=col_0_span)

ax      = plt.subplot2grid((nrows,ncols),(1,col_1),colspan=col_1_span)
pydarn.RTP.plot_range_time(fitacf, beam_num=beam, parameter='p_l', zmax=50, zmin=0, 
        date_fmt='%H', colorbar_label='Power (dB)', cmap='viridis',ax=ax)

ax.set_ylabel('Slant Range [km]')
ax.set_xlim(sDate,eDate)
ax.set_xlabel('Time [UT]')

dfmt    = '%Y %b %d %H%M UT'
dates   = '{!s} - {!s}'.format(sDate.strftime(dfmt), eDate.strftime(dfmt))
ax.set_title('{!s} Beam {!s}\n{!s}'.format(radar.upper(),beam,dates))

fig.savefig('debug.png',bbox_inches='tight')
