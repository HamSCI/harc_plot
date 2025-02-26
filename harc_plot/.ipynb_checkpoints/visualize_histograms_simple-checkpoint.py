import os
import glob
import datetime
import dateutil
from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PolyCollection
import cartopy.crs as ccrs

import numpy as np
import pandas as pd
import xarray as xr
import netCDF4

import tqdm

from . import gen_lib as gl

pdict   = {}

#dct = {}
#dct['log_z']    = False
#pdict['mean']   = dct
#
#dct = {}
#dct['log_z']    = False
#pdict['median'] = dct
#
#dct = {}
#dct['log_z']   = False
#pdict['std']   = dct
#
#dct = {}
#dct['log_z']   = False
#pdict['sum']   = dct

dct = {}
dct['log_z']        = False
pdict['z_score']    = dct

dct = {}
dct['log_z']        = False
pdict['pct_err']    = dct

def plot_nc(data_da,map_da,png_path,xlim=(0,24),ylim=None,**kwargs):

    stat    = data_da.attrs.get('stat')
    pdct    = pdict.get(stat,{})
    log_z   = pdct.get('log_z',True)

    freqs       = np.sort(data_da['freq_MHz'])[::-1]

    nx      = 100
    ny      = len(freqs)

    fig     = plt.figure(figsize=(30,4*ny))

    plt_nr  = 0
    for inx,freq in enumerate(freqs):
        plt_nr += 1
        ax = plt.subplot2grid((ny,nx),(inx,0),projection=ccrs.PlateCarree(),colspan=30)

        ax.coastlines(zorder=10,color='w')
        ax.plot(np.arange(10))
        map_data            = map_da.sel(freq_MHz=freq).copy()
        xkey                = data_da.attrs['xkey']

        if xkey in list(map_data.coords):
            xvec                = np.array(map_data[xkey])
            tf                  = np.logical_and(xvec >= xlim[0], xvec < xlim[1])
            map_data            = map_data[{xkey:tf}].sum(dim=xkey)

        map_n               = int(np.sum(map_data))
        tf                  = map_data < 1
        map_data            = np.log10(map_data)
        map_data.values[tf] = 0
        map_data.name       = 'log({})'.format(map_data.name)

        map_data.plot.contourf(x=map_da.attrs['xkey'],y=map_da.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)

        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}
        ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(map_n),
                ha='center',transform=ax.transAxes,fontdict=fdict)
        
        plt_nr += 1
        ax = plt.subplot2grid((ny,nx),(inx,35),colspan=65)
        data    = data_da.sel(freq_MHz=freq).copy()
        if log_z:
            tf          = data < 1.
            data        = np.log10(data)
            data.values[tf] = 0
            data.name   = 'log({})'.format(data.name)

        data.plot.contourf(x=data_da.attrs['xkey'],y=data_da.attrs['ykey'],ax=ax,levels=30)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.tight_layout()
    fig.savefig(png_path,bbox_inches='tight')
    plt.close(fig)

class ncLoader(object):
    def __init__(self,nc):
        with netCDF4.Dataset(nc) as nc_fl:
            groups  = [group for group in nc_fl.groups['time_series'].groups.keys()]

        das = OrderedDict()
        for group in groups:
            das[group] = OrderedDict()
            grp = '/'.join(['time_series',group])
            with xr.open_dataset(nc,group=grp) as fl:
                ds      = fl.load()

            for param in ds.data_vars:
                das[group][param] = ds[param]
        xkeys   = groups.copy()

        map_das = OrderedDict()
        for xkey in xkeys:
            grp = '/'.join(['map',xkey])
            with xr.open_dataset(nc,group=grp) as fl:
                ds      = fl.load()
            map_das[xkey]   = ds[list(ds.data_vars)[0]]

        self.nc         = nc
        self.das        = das
        self.xkeys      = xkeys
        self.map_das    = map_das

def main(run_dct):
    srcs        = run_dct['srcs']
    baseout_dir = run_dct['baseout_dir']

    ncs = glob.glob(srcs)
    ncs.sort()

    for nc_bz2 in ncs:
        mbz2    = gl.MyBz2(nc_bz2)
        mbz2.uncompress()
        nc      = mbz2.unc_name

        ncl     = ncLoader(nc)
        bname   = os.path.basename(nc)[:-3]
        for xkey in ncl.xkeys:
            outdir  = os.path.join(baseout_dir,xkey)
            gl.prep_output({0:outdir})
            map_da  = ncl.map_das[xkey]

            for param,data_da in ncl.das[xkey].items():
                fname   = '.'.join([bname,xkey,param,'png'])
                fpath   = os.path.join(outdir,fname)
                print(fpath)
                plot_nc(data_da,map_da,png_path=fpath,**run_dct)

        mbz2.remove()
