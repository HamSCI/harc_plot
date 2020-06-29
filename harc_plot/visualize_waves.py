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

from scipy import signal

import tqdm

from . import gen_lib as gl

class StepName(object):
    def __init__(self,family='WAVE'):
        self.count = 0
        self.family = family

    def next_name(self,process_name):
        self.name  = '{!s}_{:03d}_{!s}'.format(self.family,self.count,process_name)
        self.count += 1
        return self.name

class WavePlot(object):
    def __init__(self,data_da,map_da,png_path,xlim=(0,24),ylim=None,**kwargs):
        self.data_set = xr.Dataset()
        self.data_set['map']  = map_da
        self.data_set['map'].attrs.update({'label':map_da.name})

        self.data_set['hist'] = data_da
        self.data_set['hist'].attrs.update({'label':data_da.name})

        self.xlim       = xlim
        self.ylim       = ylim

        self.png_path   = png_path

        self.calculate_waves()
        self.plot_summary(**kwargs)
        self.plot_waves()

    def calculate_waves(self,log_z=True):
        hist        = self.data_set['hist']
        xkey        = hist.attrs['xkey']
        ykey        = hist.attrs['ykey']
        freqs       = np.sort(hist['freq_MHz'])[::-1]
        
        # Compute Log10 of Spot Density
        if log_z:
            for freq in freqs:
                data        = hist.sel(freq_MHz=freq).copy()

                data_attrs  = data.attrs
                tf          = data < 1.
                data        = np.log10(data)
                data.values[tf] = 0
                data.attrs  = data_attrs
                hist.loc[{'freq_MHz':freq}] = data

            hist.attrs['label'] = 'log({})'.format(hist.attrs['label'])
        self.data_set['hist'] = hist

        ################################################################################
        names = StepName()

        # Compute Index Line
        name        = names.next_name('sum')
        name_inx    = name
        da_new      = hist.sum(ykey)

        for freq in freqs:
            data            = hist.sel(freq_MHz=freq).sum(ykey).values
            da_new.loc[{'freq_MHz':freq}] = data

        da_new.attrs = hist.attrs
        da_new.attrs['color'] = 'white'
        da_new.attrs['lw']    = 2
        self.data_set[name] = da_new.copy()

        # Compute Background
        fit_order           = 5
        name                = names.next_name('polyfit_{!s}'.format(fit_order))
        name_pfit           = name
        da_new              = da_new.copy()
        hrs                 = da_new[xkey]

        for freq in freqs:
            data            = da_new.sel(freq_MHz=freq).values
            coefs           = np.polyfit(hrs,data,fit_order)
            fit_fn          = np.poly1d(coefs)
            wave_bkgrnd     = fit_fn(hrs)
            da_new.loc[{'freq_MHz':freq}] = wave_bkgrnd

        da_new.attrs['color'] = 'red'
        da_new.attrs['lw']    = 2
        self.data_set[name] = da_new.copy()

        # Detrend
        name    = 'DETREND'
        da_new  = self.data_set[name_inx] - self.data_set[name_pfit]

        da_new.attrs = hist.attrs
        self.data_set[name] = da_new.copy()

    def plot_map_ax(self,ax,freq):
        map_da      = self.data_set['map']
        data_da     = self.data_set['hist']
        xlim        = self.xlim

        ax.coastlines(zorder=10,color='w')
        map_data            = map_da.sel(freq_MHz=freq).copy()
        xkey                = data_da.attrs['xkey']
        ykey                = data_da.attrs['ykey']

        if xkey in list(map_data.coords):
            xvec                = np.array(map_data[xkey])
            tf                  = np.logical_and(xvec >= xlim[0], xvec < xlim[1])
            map_data            = map_data[{xkey:tf}].sum(dim=xkey)

        map_n               = int(np.sum(map_data))
        tf                  = map_data < 1
        map_data            = np.log10(map_data)
        map_data.values[tf] = 0
        map_label           = 'log({})'.format(map_da.attrs['label'])

        map_data.plot.contourf(x=map_da.attrs['xkey'],y=map_da.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno,
                cbar_kwargs={'label':map_label})

        lweight = mpl.rcParams['axes.labelweight']
        lsize   = mpl.rcParams['axes.labelsize']
        fdict   = {'weight':lweight,'size':lsize}
        ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(map_n),
                ha='center',transform=ax.transAxes,fontdict=fdict)

    def plot_hist_ax(self,ax,freq):
        data_da     = self.data_set['hist']
        xkey        = data_da.attrs['xkey']
        ykey        = data_da.attrs['ykey']
        xlim        = self.xlim
        ylim        = self.ylim

        data        = data_da.sel(freq_MHz=freq).copy()

        data.plot.contourf(x=xkey,y=ykey,ax=ax,levels=30,
                    cbar_kwargs={'label':data.attrs['label']})

        # Overplot Wave index lines
        ax2     = ax.twinx()
        for key in self.data_set.keys():
            if key.startswith('WAVE_'):
                ovp_da  = self.data_set[key]
                yy      = ovp_da.sel(freq_MHz=freq)
                xx      = yy[xkey]
               
                lw      = ovp_da.attrs.get('lw',2)
                color   = ovp_da.attrs.get('color','white')
                
                ax2.plot(xx,yy,lw=lw,color=color)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def plot_detrend_ax(self,ax,freq):
        data_da     = self.data_set['DETREND']
        xkey        = data_da.attrs['xkey']
        ykey        = data_da.attrs['ykey']
        xlim        = self.xlim
        ylim        = self.ylim

        data        = data_da.sel(freq_MHz=freq).copy()

        yy          = data_da.sel(freq_MHz=freq)
        xx          = yy[xkey]
       
        ax.plot(xx,yy)
        
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def plot_spectrogram_ax(self,ax,freq):
        data_da     = self.data_set['DETREND']
        xkey        = data_da.attrs['xkey']
        ykey        = data_da.attrs['ykey']
        xlim        = self.xlim
        ylim        = self.ylim

        data        = data_da.sel(freq_MHz=freq).copy()

        # Sampling period in seconds
        Ts          = float(data.attrs['dx'])*60.*60.
        fs          = 1./Ts # Sampling Frequency [Hz]

        yy          = data.values
        sTime_hr    = np.min(data[xkey].values)

        win_len_hr  = 3.
        win_len_N   = int((win_len_hr*3600.)/Ts)

        sliding_min = 10.
        noverlap    = int( (win_len_hr*60. - sliding_min)/(Ts/60.) )

        f, t, Sxx   = signal.spectrogram(yy, fs,nperseg=win_len_N,noverlap=noverlap)

        t_hr        = t/3600. - sTime_hr

        T           = (1./f)/60.    # Period in minutes
        pcol        = ax.pcolormesh(t_hr, T[1:], Sxx[1:,:], shading='gouraud')

#        ax.set_ylabel('Frequency [Hz]')
        ax.set_ylabel('Period [min]')
        ax.set_xlabel(xkey)
        ax.set_xlim(xlim)
        ax.set_ylim(180,4)

        fig = ax.get_figure()
        fig.colorbar(pcol,label='Pwr Spct Dnsty')

    def plot_summary(self,**kwargs):
        xlim        = self.xlim
        ylim        = self.ylim

        data_da     = self.data_set['hist']
        freqs       = np.sort(data_da['freq_MHz'])[::-1]

        nx          = 100
        ny          = len(freqs)

        fig         = plt.figure(figsize=(30,4*ny))
        plt_nr  = 0
        for inx,freq in enumerate(freqs):
            ax      = plt.subplot2grid((ny,nx),(inx,0),projection=ccrs.PlateCarree(),colspan=30)
            self.plot_map_ax(ax,freq)
            
            ax      = plt.subplot2grid((ny,nx),(inx,35),colspan=65)
            self.plot_hist_ax(ax,freq)

        fig.savefig(self.png_path,bbox_inches='tight')
        plt.close(fig)

    def plot_waves(self,**kwargs):
        xlim        = self.xlim
        ylim        = self.ylim

        data_da     = self.data_set['hist']
        freqs       = np.sort(data_da['freq_MHz'])[::-1]

        nx          = 100
        ny          = 3

        for freq in freqs:
            fig         = plt.figure(figsize=(30,4*ny))

            ## Plot the map of all data points.
            plt_nr      = 0
            ax          = plt.subplot2grid((ny,nx),(plt_nr,0),projection=ccrs.PlateCarree(),colspan=30)
            self.plot_map_ax(ax,freq)
            
            ## Plot the histogram with wave lines.
            ax          = plt.subplot2grid((ny,nx),(plt_nr,35),colspan=65)
            self.plot_hist_ax(ax,freq)
            ax_pos_0    = np.array(ax.get_position().bounds) # Keep track of axis position.

            ## Plot the detrended wave line.
            plt_nr     += 1
            ax          = plt.subplot2grid((ny,nx),(plt_nr,35),colspan=65)
            self.plot_detrend_ax(ax,freq)
            # Adjust plot location because this one doesn't have a colorbar.
            ax_pos_1    = np.array(ax.get_position().bounds)
            ax_pos_1[2] = ax_pos_0[2]
            ax.set_position(ax_pos_1)

            plt_nr += 1
            ax      = plt.subplot2grid((ny,nx),(plt_nr,35),colspan=65)
            self.plot_spectrogram_ax(ax,freq)

            fname   = '{!s}_WAVE_{:02d}MHz.png'.format(self.png_path[:-4],freq)
            fig.savefig(fname,bbox_inches='tight')
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
                wave_plot = WavePlot(data_da,map_da,png_path=fpath,**run_dct)

        mbz2.remove()
