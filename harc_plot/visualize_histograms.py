import os
import glob
import datetime
from collections import OrderedDict
import string
import ast

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

import numpy as np
np.seterr(divide = 'ignore') 
import pandas as pd
import xarray as xr
import netCDF4

import tqdm

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from . import gen_lib as gl
from .timeutils import daterange
from .geospace_env import GeospaceEnv
from . import goes

pdict   = {}

dct = {}
dct['log_z']        = False
pdict['z_score']    = dct

dct = {}
dct['log_z']        = False
pdict['pct_err']    = dct

dct = {}
dct['log_z']            = False
pdict['mean_subtract']  = dct

label_dict  = {}
label_dict['spot_density']          = 'spots bin$^{-1}$'
label_dict['spot_density_z_score']  = '$z$(spots bin$^{-1}$)'

band_dct = OrderedDict()
dct             = {'label':'28 MHz'}
band_dct[28]    = dct

dct             = {'label':'21 MHz'}
band_dct[21]    = dct

dct             = {'label':'14 MHz'}
band_dct[14]    = dct

dct             = {'label':'7 MHz'}
band_dct[7]     = dct

dct             = {'label':'3.5 MHz'}
band_dct[3]     = dct

dct             = {'label':'1.8 MHz'}
band_dct[1]     = dct

tick_params = dict(direction='out', length=8, width=2, colors='k',zorder=600)

def plot_letter(inx,ax):
    txt = '({!s})'.format(string.ascii_lowercase[inx])
    fontdict    = {'weight':'bold','size':36}
    ax.text(-0.095,0.875,txt,fontdict=fontdict,transform=ax.transAxes,ha='center')


def plot_axvspans(axvspans,ax):
    if axvspans is None:
        return

    for axv in axvspans:
        ax.axvspan(*axv,color='0.8',zorder=-10,alpha=0.5)

def plot_axv(axvlines,ax,label_time=False,**kwargs):
    if axvlines is None:
        return

    for axv in axvlines:
        ax.axvline(axv,ls=':',lw=4,**kwargs)
        txt = axv.strftime('%H%M')
        trans = mpl.transforms.blended_transform_factory(ax.transData,ax.transAxes)
        if label_time:
            ax.text(axv,0.98,txt,rotation=90,fontdict={'weight':'bold','size':'x-large'},transform=trans,va='top',ha='right')

class SrcCounts(object):
    def __init__(self,xdct):
        """
        Calculate percentages of data sources.
        xdct: dictionary of datasets that have src_cnt attributes.
        """
        src_cnts    = {}
        # Add up the sources for each day.
        for group,ds_list in xdct.items():
            if group not in src_cnts:
                src_cnts[group] = {}
            for ds_inx,ds in enumerate(ds_list):
                for data_var in ds.data_vars:

                    ds_uts  = ds['ut_sTime'].values[0]
                    ds_t0   = ds_uts + pd.Timedelta(ds[group].values[0],'h')


                    print('Counting: {:05d}: {!s}'.format(ds_inx,ds_t0))

                    src_cnt = pd.read_json(ds[data_var].attrs.get('src_cnt'))

                    # Replace NaNs with 0s for the count.
                    # This should eventually be fixed in the raw
                    # histogram computing code so that NaNs don't appear
                    # here to begin with.
                    src_cnt = src_cnt.fillna(0)

                    if data_var not in src_cnts[group]:
                        src_cnts[group][data_var] = src_cnt
                    else:
                        # We need to use the .add() method with fill_value=0 or pandas will
                        # put in NaNs if there is a day that is missing one of the already
                        # listed sources (like WSPRNet or RBN)
                        src_cnts[group][data_var] = src_cnts[group][data_var].add(src_cnt,fill_value=0)

        # Compute percentages.
        for group,ds_list in xdct.items():
            for data_var in ds.data_vars:
                sc          = src_cnts[group][data_var]
                sc['sum']   = sc.sum(axis=1)
                sum0        = sc.sum(axis=0)
                sum0.name   = 'sum'
                sc          = sc.append(sum0)
                sc['pct']   = (sc['sum']/sc.loc['sum','sum']) * 100.
                src_cnts[group][data_var]   = sc

        self.src_cnts   = src_cnts

    def get_text(self,group,data_var):
        
        sc      = self.src_cnts[group][data_var]
        srcs    = list(sc.index)
        srcs.remove('sum')
        srcs.sort()

        lines   = []
        for src in srcs:
            pct = sc.loc[src,'pct']
            line    = '   {!s}: {:.0f}%'.format(src,pct)
            lines.append(line)

        return lines

def center_of_mass(da,c0,c1):
    """
    Calculate the center of mass of a 2D xarray.
    Used to find the lat/lon center of a group of
    spots.
    """

    c0s     = np.array(da.sum(c1))
    crds    = np.array(da.coords[c0])
    com     = np.sum(c0s*crds)/np.sum(c0s)
    return com
    
class Sza(object):
    def __init__(self,xdct,sTime,eTime):
        """
        Calculate percentages of data sources.
        xdct: dictionary of datasets that have src_cnt attributes.
        """
        sza_dct = {}
        # Add up the sources for each day.
        for group,ds in xdct.items():
            for data_var in ds.data_vars:
                da = ds[data_var].sum('freq_MHz')

                for dim in da.dims:
                    if '_long' in dim:
                        lon_dim = dim

                    if '_lat' in dim:
                        lat_dim = dim

                sza_lat = center_of_mass(da,lat_dim,lon_dim)
                sza_lon = center_of_mass(da,lon_dim,lat_dim)

                if 'slt' in group:
                    offset      = datetime.timedelta(hours=(sza_lon/15.))
                    sza         = gl.calc_solar_zenith(sTime-offset,eTime-offset,sza_lat,sza_lon)
                    sza.index   = sza.index + offset
                else:
                    sza         = gl.calc_solar_zenith(sTime,eTime,sza_lat,sza_lon)

                if 'spot_density' in data_var:
                    sza_dct[group] = {'sza':sza,'lat':sza_lat,'lon':sza_lon}

        self.sza    = sza_dct

    def plot(self,group,ax,ls='--',lw=4,color='white'):
        sza_lat = self.sza[group]['lat'] 
        sza_lon = self.sza[group]['lon'] 
        sza     = self.sza[group]['sza'] 

        xx      = sza.index
        yy      = sza['els']

        sza_ax  = ax.twinx()
        sza_ax.plot(xx,yy,ls=ls,lw=lw,color=color)
        ylabel  = u'Solar Zenith \u2220\n@ ({:.0f}\N{DEGREE SIGN} N, {:.0f}\N{DEGREE SIGN} E)'.format(sza_lat,sza_lon)
        sza_ax.set_ylabel(ylabel)
        sza_ax.set_ylim(110,0)

class Goeser(object):
    def __init__(self,sTime,eTime,sats=[13,15]):
        self.sTime  = sTime
        self.eTime  = eTime
        self.sats   = sats

        self.load_goes()

    def load_goes(self):
        sTime   = self.sTime
        eTime   = self.eTime
        sats    = self.sats

        goes_dcts       = OrderedDict()
        for sat in sats:
            goes_dcts[sat]  = {}

        flares_combined = pd.DataFrame()
        for sat_nr,gd in goes_dcts.items():
            gd['data']      = goes.read_goes(sTime,eTime,sat_nr=sat_nr)

            # Skip if no GOES data present
            if gd['data'] is None:
                continue

            flares          = goes.find_flares(gd['data'],min_class='M1',window_minutes=60)
            flares['sat']   = sat_nr
            gd['flares']    = flares
            flares_combined = flares_combined.append(flares).sort_index()
            gd['var_tags']  = ['B_AVG']
            gd['labels']    = ['GOES {!s}'.format(sat_nr)]

        self.flares_combined = flares_combined[~flares_combined.index.duplicated()].sort_index()
        self.goes_dcts  = goes_dcts

    def plot(self,ax,lw=2):
        sTime   = self.sTime
        eTime   = self.eTime

        for sat_nr,gd in self.goes_dcts.items():
            if gd['data'] is None: continue
            goes.goes_plot(gd['data'],sTime,eTime,ax=ax,
                    var_tags=gd['var_tags'],labels=gd['labels'],
                    legendLoc='upper right',lw=lw)

        title   = 'NOAA GOES X-Ray (0.1 - 0.8 nm) Irradiance'
        size    = 20
        ax.text(0.01,0.05,title,transform=ax.transAxes,ha='left',fontdict={'size':size,'weight':'bold'})


class ncLoader(object):
    def __init__(self,sTime,eTime=None,srcs=None,band_keys=None,**kwargs):
        if eTime is None:
            eTime = sTime + datetime.timedelta(hours=24)

        self.sTime      = sTime
        self.eTime      = eTime
        self.srcs       = srcs
        self.kwargs     = kwargs
        self.band_keys  = band_keys

        self._set_basename()
        self._get_fnames()
        self._load_ncs()

    def _set_basename(self):
        bname   = os.path.basename(self.srcs)
        bname   = bname.strip('*.')
        bname   = bname.strip('.nc')
        self.basename   = bname

    def _get_fnames(self):
        tmp = glob.glob(self.srcs)
        tmp.sort()
        
        dates       = daterange(self.sTime,self.eTime)
        date_strs   = [x.strftime('%Y%m%d') for x in dates]

        fnames      = []
        for fn in tmp:
            bn  = os.path.basename(fn)
            if bn[:8] in date_strs:
                fnames.append(fn)

        self.fnames = fnames
        return fnames
            
    def _load_ncs(self):
        prefixes    = ['map','time_series']

        # Return None if no data to load.
        if self.fnames == []:
            self.maps       = None
            self.datasets   = None
            return

        dss         = {}
        print(' Loading files...')
        for nc_bz2 in self.fnames:
            print(' --> {!s}'.format(nc_bz2))
            mbz2    = gl.MyBz2(nc_bz2)
            mbz2.uncompress()

            nc      = mbz2.unc_name
            # Identify Groups in netCDF File
            with netCDF4.Dataset(nc) as nc_fl:
                groups  = [group for group in nc_fl.groups['time_series'].groups.keys()]

            # Only plot specified xkeys.
            xkeys = self.kwargs.get('xkeys')
            if xkeys is not None:
                new_groups = []
                for group in groups:
                    if group in xkeys:
                        new_groups.append(group)
                groups = new_groups

            # Store DataSets (dss) from each group in an OrderedDict()
            for prefix in prefixes:
                if prefix not in dss:
                    dss[prefix] = OrderedDict()

                for group in groups:
                    grp = '/'.join([prefix,group])
                    with xr.open_dataset(nc,group=grp) as fl:
                        ds      = fl.load()

                    # Select only bands that we will be plotting.
                    if self.band_keys is not None:
                        ds      = ds.loc[{'freq_MHz':self.band_keys}]

                    # Calculate time vector relative to self.sTime
                    hrs         = np.array(ds.coords[group])
                    dt_0        = pd.Timestamp(np.array(ds['ut_sTime']).tolist())
                    time_vec    = [(dt_0 + pd.Timedelta(hours=x) - self.sTime).total_seconds()/3600. for x in hrs]

                    ds.coords[group]        = time_vec
                    ds.coords['ut_sTime']   = [self.sTime]

                    if prefix == 'map':
                        dt_vec      = np.array([dt_0 + pd.Timedelta(hours=x) for x in hrs])
                        tf          = np.logical_and(dt_vec >= self.sTime, dt_vec < self.eTime)
                        tmp_map_ds  = ds[{group:tf}].sum(group,keep_attrs=True)

                        map_ds      = dss[prefix].get(group)
                        if map_ds is None:
                            map_ds      = tmp_map_ds
                        else:
                            map_attrs = map_ds['spot_density'].attrs
                            map_ds += tmp_map_ds
                            map_ds['spot_density'].attrs = map_attrs
                        dss[prefix][group]  = map_ds
                    else:
                        if group not in dss[prefix]:
                            dss[prefix][group]  = []
                        dss[prefix][group].append(ds)

            mbz2.remove()

        # Process source counts - know what percentage of spots came from what sources.
        self.sza        = Sza(dss['map'],self.sTime,self.eTime)
        self.src_cnts   = SrcCounts(dss['time_series'])

        # Concatenate Time Series Data
        xlim    = (0, (self.eTime-self.sTime).total_seconds()/3600.)
        print(' Concatenating data...')
        prefix  = 'time_series'
        xdct    = dss[prefix]
        for group,ds_list in xdct.items():
            ds          = xr.concat(ds_list,group)
            for data_var in ds.data_vars:
                print(prefix,group,data_var)
                attrs   = ds[data_var].attrs
                attrs.update({'xlim':str(xlim)})
                ds[data_var].attrs = attrs
            dss[prefix][group]  = ds

        self.datasets   = dss

    def _format_timeticklabels(self,ax):
        xlim        = self.xlim
        tmf         = self.time_format

        # "Smart" Tick Formatting Logic ########  
        if xlim[1] - xlim[0] < datetime.timedelta(days=1):
            for tl in ax.get_xticklabels():
                tl.set_rotation(0)
                tl.set_ha('center')

            xtls    = []
            for xtk in ax.get_xticks():
                xtkd    = mpl.dates.num2date(xtk)

                dec_hr  = xtkd.hour + xtkd.minute/60.
                xtl     = '{:g}'.format(dec_hr)
                xtls.append(xtl)
            ax.set_xticklabels(xtls)

            xlbl    = ax.get_xlabel()
            if xlbl == 'Date Time [UT]':
                ax.set_xlabel('Hours [UT]')
        else:
            for tl in ax.get_xticklabels():
                tl.set_rotation(10)
                tl.set_ha('right')

        # Manual Override ######################
        fmt = tmf.get('format')
        if fmt is not None:
            xtls    = []
            for xtk in ax.get_xticks():
                xtkd    = mpl.dates.num2date(xtk)
                xtls.append(xtkd.strftime(fmt))
            ax.set_xticklabels(xtls)

        rotation    = tmf.get('rotation')
        if rotation is not None:
            for tl in ax.get_xticklabels():
                tl.set_rotation(rotation)

        ha          = tmf.get('ha')
        if ha is not None:
            for tl in ax.get_xticklabels():
                tl.set_ha(ha)

        label       = tmf.get('label')
        if label is not None:
            ax.set_xlabel(label)

    def plot(self,baseout_dir='output',xlim=None,ylim=None,xunits='datetime',
            plot_sza=True,subdir=None,geospace_env=None,plot_region=None,
            plot_kpsymh=True,plot_goes=True,plot_f107=False,axvlines=None,axvlines_kw={},axvspans=None,time_format={},
            xkeys=None,log_z=None,**kwargs):
        if self.datasets is None:
            return

        if geospace_env is None:
            geospace_env    = GeospaceEnv()

        self.time_format    = time_format
        xlim_in = xlim
        if axvlines_kw is None:
            axvlines_kw = {}

        fpaths = [] # Keep track of paths of all plotted figures.
        for group,ds in self.datasets['time_series'].items():

            #Only plot specified xkeys (i.e. ut_hrs and not slt_mid)
            if xkeys is not None:
                if group not in xkeys: continue

            map_da  = self.datasets['map'][group]['spot_density']

            outdir  = os.path.join(baseout_dir,group)
            if subdir is not None:
                outdir = os.path.join(outdir,subdir)
            gl.prep_output({0:outdir},clear=False)

            for data_var in ds.data_vars:
                data_da = ds[data_var].copy()

               # Set Axis Limits ###################### 
                if ylim is None:
                    ylim = ast.literal_eval(data_da.attrs.get('ylim','None'))

                xlim    = xlim_in
                if xlim is None:
                    xlim = ast.literal_eval(data_da.attrs.get('xlim','None'))

                if xunits == 'datetime':
                    hrs         = np.array(data_da.coords[group])
                    dt_vec      = [self.sTime + pd.Timedelta(hours=x) for x in hrs]
                    data_da.coords[group] = dt_vec

                    if xlim is not None:
                        xlim_0      = pd.Timedelta(hours=xlim[0]) + self.sTime
                        xlim_1      = pd.Timedelta(hours=xlim[1]) + self.sTime
                        xlim        = (xlim_0,xlim_1)

                self.xlim   = xlim
               ######################################## 

                stat        = data_da.attrs.get('stat')
                pdct        = pdict.get(stat,{})
                
                if log_z is None:
                    log_z       = pdct.get('log_z',True)

                freqs       = np.sort(data_da['freq_MHz'])[::-1]

                nx      = 100
                ny      = len(freqs)
                if plot_kpsymh:
                    ny += 1

                if plot_goes:
                    ny += 1

                if plot_f107:
                    ny += 1

                fig         = plt.figure(figsize=(33,4*ny))
#                fig         = plt.figure(figsize=(50,4*ny))
                col_0       = 0
                col_0_span  = 30
                col_1       = 36
                col_1_span  = 65

#                fig         = plt.figure(figsize=(30,4*ny))
#                col_0       = 0
#                col_0_span  = 30
#                col_1       = 35
#                col_1_span  = 65

                axs_to_adjust   = []

                pinx = -1
                if plot_kpsymh:
                    pinx    += 1
                    ax      = plt.subplot2grid((ny,nx),(pinx,col_1),colspan=col_1_span)
                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)

                    omni_axs        = geospace_env.omni.plot_dst_kp(self.sTime,self.eTime,ax,xlabels=True,
                                        kp_markersize=10,dst_lw=2,dst_param='SYM-H')
                    ax.tick_params(**tick_params)
                    plot_letter(pinx,ax)

                    label_time  = axvlines_kw.get('label_time',True)
                    plot_axv(axvlines,omni_axs[0],color='k',label_time=label_time)
                    plot_axvspans(axvspans,omni_axs[0])
                    for ax in omni_axs:
                        self._format_timeticklabels(ax)
                        ax.set_xlabel('')
                    axs_to_adjust   += omni_axs

                ######################################## 
                if plot_goes:
                    goeser  = Goeser(self.sTime,self.eTime)
                    pinx    +=1 
                    ax      = plt.subplot2grid((ny,nx),(pinx,col_1),colspan=col_1_span)
                    goeser.plot(ax)
                    ax.set_xlim(xlim)
                    plot_axv(axvlines,ax,color='k')
                    plot_axvspans(axvspans,ax)
                    ax.tick_params(**tick_params)
                    plot_letter(pinx,ax)
                    axs_to_adjust.append(ax)
                    self._format_timeticklabels(ax)
                    ax.set_xlabel('')

                ######################################## 
                if plot_f107:
                    pinx    +=1 
                    ax      = plt.subplot2grid((ny,nx),(pinx,col_1),colspan=col_1_span)
                    geospace_env.omni.plot_f107(self.sTime,self.eTime,ax,xlabels=True)
                    ax.set_xlim(xlim)
                    plot_axv(axvlines,ax,color='k')
                    plot_axvspans(axvspans,ax)
                    ax.tick_params(**tick_params)
                    plot_letter(pinx,ax)
                    axs_to_adjust.append(ax)
                    self._format_timeticklabels(ax)
                    ax.set_xlabel('')
                
                map_sum         = 0
                for inx,freq in enumerate(freqs):
                    plt_row = inx+pinx+1

                    # Plot Map #############################
                    ax = plt.subplot2grid((ny,nx),(plt_row,col_0),projection=ccrs.PlateCarree(),colspan=col_0_span)

                    ax.coastlines(zorder=10,color='w')
                    ax.plot(np.arange(10))
                    map_data    = map_da.sel(freq_MHz=freq).copy()
                    map_data.name   = label_dict.get(map_da.name,map_da.name)
                    tf          = map_data < 1
                    map_n       = int(np.sum(map_data))
                    map_sum     += map_n
                    map_data    = np.log10(map_data)
                    map_data.values[tf] = 0
                    map_data.name   = 'log({})'.format(map_data.name)
                    map_data.plot.contourf(x=map_da.attrs['xkey'],y=map_da.attrs['ykey'],ax=ax,levels=30,cmap=mpl.cm.inferno)
                    ax.set_title('')
                    lweight = mpl.rcParams['axes.labelweight']
                    lsize   = mpl.rcParams['axes.labelsize']
                    fdict   = {'weight':lweight,'size':lsize}
                    ax.text(0.5,-0.1,'Radio Spots (N = {!s})'.format(map_n),
                            ha='center',transform=ax.transAxes,fontdict=fdict)

                    if plot_sza:
                        sza_lat = self.sza.sza[group]['lat'] 
                        sza_lon = self.sza.sza[group]['lon'] 
                        ax.scatter([sza_lon],[sza_lat],marker='*',s=600,color='yellow',
                                        edgecolors='black',zorder=500,lw=3)

                    if plot_region is not None:
                        rgn         = gl.regions.get(plot_region)
                        lat_lim     = rgn.get('lat_lim')
                        lon_lim     = rgn.get('lon_lim')

                        ax.set_xlim(lon_lim)
                        ax.set_ylim(lat_lim)

                    # Plot Time Series ##################### 
                    ax      = plt.subplot2grid((ny,nx),(plt_row,col_1),colspan=col_1_span)
                    plot_letter(plt_row,ax)
                    data    = data_da.sel(freq_MHz=freq).copy()

                    data.name   = label_dict.get(data_da.name,data_da.name)

                    if log_z:
                        tf          = data < 1.
                        data        = np.log10(data)
                        data        = xr.where(tf,0,data)
                        data.name   = 'log({})'.format(data.name)

                    if plot_sza:
                        cbar_kwargs = {'pad':0.080}
                    else:
                        cbar_kwargs = {}

                    robust_dict = self.kwargs.get('robust_dict',{})
                    robust      = robust_dict.get(freq,True)
                    result      = data.plot.contourf(x=data_da.attrs['xkey'],y=data_da.attrs['ykey'],ax=ax,levels=30,robust=robust,cbar_kwargs=cbar_kwargs)
#                    result      = data.plot.pcolormesh(x=data_da.attrs['xkey'],y=data_da.attrs['ykey'],ax=ax,robust=robust,cbar_kwargs=cbar_kwargs)

                    if plot_sza:
                        self.sza.plot(group,ax)

                    xlbl    = ax.get_xlabel()
                    if xlbl == 'ut_hrs':
                        ax.set_xlabel('Date Time [UT]')
                    elif xlbl == 'slt_mid':
                        ax.set_xlabel('Midpoint Solar Local Time')

                    ylbl    = ax.get_ylabel()
                    if ylbl == 'dist_Km':
                        ax.set_ylabel('$R_{gc}$ [km]')

                    ax.set_title('')

                    bdct    = band_dct.get(freq,{})
                    label   = bdct.get('label','{!s} MHz'.format(freq))

                    ax.text(-0.12,0.5,label,transform=ax.transAxes,va='center',
                            rotation=90,fontdict={'weight':'bold','size':30})


                    ax.set_xlim(xlim)
                    ax.set_ylim(ylim)
                    if result.cmap.name == 'viridis':
                        color = 'w'
                    else:
                        color = 'k'
                    plot_axv(axvlines,ax,color=color)
                    ax.tick_params(**tick_params)
                    self._format_timeticklabels(ax)
                    if inx != len(freqs)-1:
                        ax.set_xlabel('')

                    hist_ax = ax

                # Place Information in Upper Left Corner of Figure
                date_str_0  = self.sTime.strftime('%d %b %Y')
                date_str_1  = self.eTime.strftime('%d %b %Y')

                lines   = []
                l       = lines.append

                if self.eTime-self.sTime < datetime.timedelta(hours=24):
                    l(date_str_0)
                else:
                    l('{!s}-\n{!s}'.format(date_str_0,date_str_1))

                l('Ham Radio Networks')
                l('N Spots = {!s}'.format(map_sum))
                lines   += self.src_cnts.get_text(group,data_var)
                txt     = '\n'.join(lines)

                if not plot_kpsymh and not plot_goes:
                    xpos        = 0.025
                    ypos        = 1.005
                    fdict       = {'size':38,'weight':'bold'}
                    va          = 'bottom'
                else:
                    xpos        = 0.025
                    ypos        = 0.995
                    fdict       = {'size':38,'weight':'bold'}
                    va          = 'top'

                fig.text(xpos,ypos,txt,fontdict=fdict,va=va)

                ######################################## 
                fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
                for ax_0 in axs_to_adjust:
                    gl.adjust_axes(ax_0,hist_ax)

                sTime_str   = self.sTime.strftime('%Y%m%d.%H%MUT')
                eTime_str   = self.eTime.strftime('%Y%m%d.%H%MUT')
                date_str    = '-'.join([sTime_str,eTime_str])

                fname   = '.'.join([date_str,self.basename,group,data_var,'png']).replace('.bz2','')
                fpath   = os.path.join(outdir,fname)
                fig.savefig(fpath,bbox_inches='tight')
                print('--> {!s}'.format(fpath))
                plt.close(fig)
                fpaths.append(fpath)

        return fpaths

def plot_dailies(run_dct):
    sTime   = run_dct['sTime']
    eTime   = run_dct['eTime']
    dates   = daterange(sTime,eTime)

    print('Plotting Dailies: {!s}'.format(run_dct['srcs']))
    for this_sTime in dates:
        this_eTime  = this_sTime + pd.Timedelta('1D')

        rd              = run_dct.copy()
        rd['sTime']     = this_sTime
        rd['eTime']     = this_eTime
        rd['subdir']    = 'dailies'
        nc_obj          = ncLoader(**rd)
        nc_obj.plot(**rd)

def main(run_dct):
    print('Starting main plotting routine...')
    nc_obj      = ncLoader(**run_dct)
    nc_obj.plot(**run_dct)
