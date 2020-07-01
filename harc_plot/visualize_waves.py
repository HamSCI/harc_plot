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
    def __init__(self,data_da,map_da,png_path,xlim=(0,24),truncate=(-4,28),ylim=None,**kwargs):
        self.data_set = xr.Dataset()
        # Make sure to add the histograms before the maps to ensure all needed
        # timesteps are kept.
        self.data_set['hist'] = data_da
        self.data_set['hist'].attrs.update({'label':data_da.name})

        self.calculate_waves()
        self.data_set = self.data_set.sel({'ut_hrs':slice(*truncate)}).copy()

        self.data_set['map']  = map_da
        self.data_set['map'].attrs.update({'label':map_da.name})

        self.xlim       = xlim
        self.ylim       = ylim

        self.png_path   = png_path

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

                data_attrs      = data.attrs
                tf              = data < 1.
                data            = np.log10(data)
                data.values[tf] = 0
                data.attrs      = data_attrs
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
        nperseg     = int((win_len_hr*3600.)/Ts)
        nfft        = max([nperseg,2**12])

        sliding_min = 10.
        noverlap    = int( (win_len_hr*60. - sliding_min)/(Ts/60.) )

        f, t, Sxx   = signal.spectrogram(yy, fs,nperseg=nperseg,noverlap=noverlap,nfft=nfft)

        t_hr        = t/3600. + sTime_hr

        T           = (1./f)/60.    # Period in minutes
        pcol        = ax.pcolormesh(t_hr, f, Sxx, shading='gouraud')#,norm=mpl.colors.LogNorm())

        ax.set_ylabel('Period [min]')
        ax.set_xlabel(xkey)
        ax.set_xlim(xlim)
        ax.set_ylim(0,1./(10.*60.))

        ytkls = []
        for ytk in ax.get_yticks():
            if ytk == 0:
                ytkls.append('Inf')
            else:
                T = (1./ytk)/60.
                ytkls.append('{:0.1f}'.format(T))
        ax.set_yticklabels(ytkls)

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

class ncLoaderLite(object):
    def __init__(self,nc):
        # Store NetCDF filename of interest
        self.nc         = nc

        # Keep track of NetCDF filenames to be loaded.
        ncs             = [nc]

        # Keep track of uncompressed filenames
        self.mbz2_list  = []
        
        # Extract the basename that will be used to create output filenames.
        self.basename   = os.path.basename(nc).rstrip('.nc.bz2')

        # Extract the startdate of the day of interest.
        date_str        = os.path.split(nc)[1][:8]
        date            = datetime.datetime.strptime(date_str,'%Y%m%d')
        self.date       = date

        # Data file for the day prior
        date_prior      = date - datetime.timedelta(days=1)
        date_prior_str  = date_prior.strftime('%Y%m%d')
        nc_prior        = nc.replace(date_str,date_prior_str)
        ncs.append(nc_prior)

        # Date file for the day post
        date_post       = date + datetime.timedelta(days=1)
        date_post_str   = date_post.strftime('%Y%m%d')
        nc_post         = nc.replace(date_str,date_post_str)
        ncs.append(nc_post)

        # Load the data from each file.
        data_das_list   = []
        map_das_list    = []
        for this_nc in ncs:
            result  = self.load_nc(this_nc)
            if result is None:
                continue

            data_das_list.append((result[0],this_nc))
            if this_nc == nc:
                map_das_list.append( (result[1],this_nc))
            del result

        # Concatenate data
        data_das    = self.cat_data(data_das_list)
        map_das     = self.cat_maps(map_das_list)

        del data_das_list
        del map_das_list

        self.data_das   = data_das
        self.map_das    = map_das

        for mbz2 in self.mbz2_list:
            mbz2.remove()

    def load_nc(self,nc):
        if not os.path.exists(nc):
            return

        if nc.endswith('.bz2'):
            mbz2    = gl.MyBz2(nc)
            mbz2.uncompress()
            nc      = mbz2.unc_name
            self.mbz2_list.append(mbz2)

        with netCDF4.Dataset(nc) as nc_fl:
            groups  = [group for group in nc_fl.groups['time_series'].groups.keys()]

        # Data Arrays (Histograms)
        data_das = OrderedDict()
        for group in groups:
            data_das[group] = OrderedDict()
            grp = '/'.join(['time_series',group])
            with xr.open_dataset(nc,group=grp) as fl:
                ds      = fl.load()

            for param in ds.data_vars:
                data_das[group][param] = ds[param]
        xkeys       = groups.copy()
        self.xkeys  = xkeys

        # Map Arrays
        map_das = OrderedDict()
        for xkey in xkeys:
            grp = '/'.join(['map',xkey])
            with xr.open_dataset(nc,group=grp) as fl:
                ds      = fl.load()
            map_das[xkey]   = ds[list(ds.data_vars)[0]]

        return data_das, map_das

    def cat_data(self,data_das_list):
        """
        Concatenate data from muliple files into a single timeseries.
        """

        data_das = {}
        for inx,(das,this_nc) in enumerate(data_das_list):
            for xkey in self.xkeys:
                if xkey not in data_das:
                    data_das[xkey] = {}

                for param in das[xkey].keys():
                    this_ds     = das[xkey][param].copy()
                    dt_0        = pd.Timestamp(np.array(this_ds['ut_sTime']).tolist())
                    time_vec    = [dt_0 + datetime.timedelta(hours=x) for x in this_ds[xkey].values]
                    this_ds     = this_ds.assign_coords({xkey:time_vec})

                    if param not in data_das[xkey]:
                        data_das[xkey][param] = None
                        ds  = this_ds.copy()
                    else:
                        ds  = data_das[xkey][param].combine_first(this_ds)

                    if inx == len(data_das_list)-1:
                        # Calculate the new time vectors in hours relative to the date of interest.
                        time_vec    = (ds[xkey].values - np.datetime64(self.date)).astype(np.float) * 1e-9 / 3600.
                        ds          = ds.assign_coords({xkey:time_vec})
                        ds          = ds.assign_coords({'ut_sTime':self.date})

                    # Save attributes only of the dataset of interest
                    if this_nc == self.nc:
                        ds.attrs = this_ds.attrs

                    data_das[xkey][param] = ds
        return data_das

    def cat_maps(self,map_das_list):
        """
        Concatenate data from muliple files into a single timeseries.
        """
        map_das = {}
        for inx,(das,this_nc) in enumerate(map_das_list):
            for xkey in self.xkeys:
                this_ds     = das[xkey]
                dt_0        = pd.Timestamp(np.array(this_ds['ut_sTime']).tolist())
                time_vec    = [dt_0 + datetime.timedelta(hours=x) for x in this_ds[xkey].values]
                this_ds     = this_ds.assign_coords({xkey:time_vec})

                if xkey not in map_das:
                    ds  = this_ds.copy()
                else:
#                    ds  = map_das[xkey].combine_first(this_ds)
                    ds  = xr.concat([map_das[xkey],this_ds],xkey) # xr.concat uses less memory than combine_first

                if inx == len(map_das_list)-1:
                    # Calculate the new time vectors in hours relative to the date of interest.
                    time_vec    = (ds[xkey].values - np.datetime64(self.date)).astype(np.float) * 1e-9 / 3600.
                    ds          = ds.assign_coords({xkey:time_vec})

                    ds = ds.drop('ut_sTime')
                    ds = ds.assign_coords({'ut_sTime':self.date})

                # Save attributes only of the dataset of interest
                if this_nc == self.nc:
                    ds.attrs = this_ds.attrs

                del this_ds
                map_das[xkey] = ds

        del map_das_list
        return map_das

def main(run_dct):
    srcs        = run_dct['srcs']
    baseout_dir = run_dct['baseout_dir']

    ncs = glob.glob(srcs)
    ncs.sort()

    for nc_bz2 in ncs:
        ncl     = ncLoaderLite(nc_bz2)
        for xkey in ncl.xkeys:
            outdir  = os.path.join(baseout_dir,xkey)
            gl.prep_output({0:outdir})
            map_da  = ncl.map_das[xkey]

            for param,data_da in ncl.data_das[xkey].items():
                fname   = '.'.join([ncl.basename,xkey,param,'png'])
                fpath   = os.path.join(outdir,fname)
                print(fpath)
                wave_plot = WavePlot(data_da,map_da,png_path=fpath,**run_dct)
