import shutil,os
import datetime
from collections import OrderedDict

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patches as mpatches

import numpy as np
import pandas as pd

import pickle
import bz2

from . import geopack
from . import calcSun

de_prop         = {'marker':'^','edgecolor':'k','facecolor':'white'}
dxf_prop        = {'marker':'*','color':'blue'}
dxf_leg_size    = 150
dxf_plot_size   = 50

rcp = matplotlib.rcParams
rcp['figure.titlesize']     = 'xx-large'
rcp['axes.titlesize']       = 'xx-large'
rcp['axes.labelsize']       = 'xx-large'
rcp['xtick.labelsize']      = 'xx-large'
rcp['ytick.labelsize']      = 'xx-large'
rcp['legend.fontsize']      = 'large'

rcp['figure.titleweight']   = 'bold'
rcp['axes.titleweight']     = 'bold'
rcp['axes.labelweight']     = 'bold'

# Parameter Dictionary
prmd = {}
tmp = {}
tmp['label']            = 'Solar Local Time [hr]'
prmd['slt_mid']         = tmp

tmp = {}
tmp['label']            = 'UT Hours'
prmd['ut_hrs']          = tmp

tmp = {}
tmp['label']            = 'Date Time [UT]'
prmd['occurred']        = tmp

tmp = {}
tmp['label']            = 'f [MHz]'
prmd['freq']            = tmp

tmp = {}
tmp['label']            = '$R_{gc}$ [km]'
prmd['dist_Km']         = tmp

tmp = {}
tmp['label']            = 'TEC'
prmd['tec']         = tmp

# Region Dictionary
regions = {}
tmp     = {}
tmp['lon_lim']  = (-180.,180.)
tmp['lat_lim']  = ( -90., 90.)
regions['World']    = tmp

tmp     = {}
tmp['lon_lim']  = (-130.,-60.)
tmp['lat_lim']  = (  20., 55.)
regions['US']   = tmp

tmp     = {}
tmp['lon_lim']  = ( -15., 55.)
tmp['lat_lim']  = (  30., 65.)
regions['Europe']   = tmp

tmp     = {}
tmp['lon_lim']  = ( -86.,-65.)
tmp['lat_lim']  = (  17., 24.)
regions['Caribbean']    = tmp

tmp     = {}
tmp['lon_lim']  = ( -86.,-65.)
tmp['lat_lim']  = (  17., 30.)
regions['Greater Caribbean']    = tmp

tmp     = {}
tmp['lon_lim']  = ( -86.,-80.)
tmp['lat_lim']  = (  24., 30.)
regions['Florida']    = tmp

tmp     = {}
tmp['lon_lim']  = ( -130.,-55.)
tmp['lat_lim']  = (   10., 55.)
regions['Greater Greater Caribbean']    = tmp

tmp     = {}
tmp['lon_lim']  = ( -130.,-60.)
tmp['lat_lim']  = (  0., 40.)
regions['Mexico']    = tmp

tmp     = {}
tmp['lon_lim']  = (-180.,180.)
tmp['lat_lim']  = ( 0., 90.)
regions['NH']    = tmp

sources     = OrderedDict()
tmp = {}
tmp['name'] = 'DXCluster'
sources[0]  = tmp

tmp = {}
tmp['name'] = 'WSPRNet'
sources[1]  = tmp

tmp = {}
tmp['name'] = 'RBN'
sources[2]  = tmp

tmp = {}
tmp['name'] = 'PSKReporter'
sources[3]  = tmp

def make_dir(path,clear=False,php=False):
    prep_output({0:path},clear=clear,php=php)

def clear_dir(path,clear=True,php=False):
    prep_output({0:path},clear=clear,php=php)

def prep_output(output_dirs={0:'output'},clear=False,width_100=False,img_extra='',php=False):
    if width_100:
        img_extra = "width='100%'"

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> ";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt = '\n'.join(txt)

    txt = []
    txt.append('<?php')
    txt.append('foreach (glob("*.png") as $filename) {')
    txt.append('    echo "<img src=\'$filename\' {img_extra}> <br />";'.format(img_extra=img_extra))
    txt.append('}')
    txt.append('?>')
    show_all_txt_breaks = '\n'.join(txt)

    for value in output_dirs.values():
        if clear:
            try:
#                shutil.rmtree(value)
                os.system('rm -rf {}/*'.format(value))
            except:
                pass
        try:
            os.makedirs(value)
        except:
            pass
        if php:
            with open(os.path.join(value,'0000-show_all.php'),'w') as file_obj:
                file_obj.write(show_all_txt)
            with open(os.path.join(value,'0000-show_all_breaks.php'),'w') as file_obj:
                file_obj.write(show_all_txt_breaks)

def cc255(color):
    cc = matplotlib.colors.ColorConverter().to_rgb
    trip = np.array(cc(color))*255
    trip = [int(x) for x in trip]
    return tuple(trip)

class BandData(object):
    def __init__(self,bands=None,cmap='HFRadio',vmin=0.,vmax=30.,cb_safe=False):
        """
        bands:  None for all HF contest bands.
                List to select bands (i.e. [7,14])
        """
        if cmap == 'HFRadio':
            self.cmap   = self.hf_cmap(vmin=vmin,vmax=vmax)
        else:
            self.cmap   = matplotlib.cm.get_cmap(cmap)

        self.cb_safe = cb_safe
        self.__cb_safe()

        self.norm   = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)

        self.__gen_band_dict__()

        # Delete unwanted bands from the band dictionary.
        if bands is not None:
            keys    = list(self.band_dict.keys())
            for band in keys:
                if band not in bands:
                    del self.band_dict[band]

    def __gen_band_dict__(self):
        bands   = []
        bands.append((28.0,  10))
        bands.append((21.0,  15))
        bands.append((14.0,  20))
        bands.append(( 7.0,  40))
        bands.append(( 3.5,  80))
        bands.append(( 1.8, 160))

        dct = OrderedDict()
        for freq,meters in bands:
            key = int(freq)
            tmp = {}
            tmp['meters']       = meters
            tmp['name']         = '{!s} m'.format(meters)
            tmp['freq']         = freq
            tmp['freq_name']    = '{:g} MHz'.format(freq)
            if self.cb_safe:
                tmp['color']        = self.cb_safe_dct.get(freq)
            else:
                tmp['color']        = self.get_rgba(freq)
            dct[key]            = tmp
        self.band_dict          = dct

    def get_rgba(self,freq):
        nrm     = self.norm(freq)
        rgba    = self.cmap(nrm)
        return rgba

    def get_hex(self,freq):

        freq    = np.array(freq)
        shape   = freq.shape
        if shape == ():
            freq.shape = (1,)

        freq    = freq.flatten()
        rgbas   = self.get_rgba(freq)

        hexes   = []
        for rgba in rgbas:
            hexes.append(matplotlib.colors.rgb2hex(rgba))

        hexes   = np.array(hexes)
        hexes.shape = shape
        return hexes

    def hf_cmap(self,name='HFRadio',vmin=0.,vmax=30.):
        fc = {}
        my_cdict = fc
        fc[ 0.0] = (  0,   0,   0)
        fc[ 1.8] = cc255('violet')
        fc[ 3.0] = cc255('blue')
        fc[ 8.0] = cc255('aqua')
        fc[10.0] = cc255('green')
        fc[13.0] = cc255('green')
        fc[17.0] = cc255('yellow')
        fc[21.0] = cc255('orange')
        fc[28.0] = cc255('red')
        fc[30.0] = cc255('red')

        cmap    = cdict_to_cmap(fc,name=name,vmin=vmin,vmax=vmax)
        return cmap

    def __cb_safe(self):
        """Color blind safe"""
        reddish_purple  = np.array((204,121,167))/255.
        vermillion      = np.array((213, 94,  0))/255.
        blue            = np.array((  0,114,178))/255.
        yellow          = np.array((240,228, 65))/255.
        bluish_green    = np.array((  0,158,115))/255.
        sky_blue        = np.array(( 86,180,233))/255.
        orange          = np.array((230,159,  0))/255.
        black           = np.array((  0,  0,  0))/255.

        fc = {}
        fc[ 1.8] = blue          
        fc[ 3.5] = sky_blue
        fc[ 7.0] = vermillion    
        fc[14.0] = bluish_green  
        fc[21.0] = orange        
        fc[28.0] = reddish_purple
        self.cb_safe_dct    = fc

def cdict_to_cmap(cdict,name='CustomCMAP',vmin=0.,vmax=30.):
	norm = matplotlib.colors.Normalize(vmin=vmin,vmax=vmax)
	
	red   = []
	green = []
	blue  = []
	
	keys = list(cdict.keys())
	keys.sort()
	
	for x in keys:
	    r,g,b, = cdict[x]
	    x = norm(x)
	    r = r/255.
	    g = g/255.
	    b = b/255.
	    red.append(   (x, r, r))
	    green.append( (x, g, g))
	    blue.append(  (x, b, b))
	cdict = {'red'   : tuple(red),
		 'green' : tuple(green),
		 'blue'  : tuple(blue)}
	cmap  = matplotlib.colors.LinearSegmentedColormap(name, cdict)
	return cmap

def sun_pos(dt=None):
    """This function computes a rough estimate of the coordinates for
    the point on the surface of the Earth where the Sun is directly
    overhead at the time dt. Precision is down to a few degrees. This
    means that the equinoxes (when the sign of the latitude changes)
    will be off by a few days.

    The function is intended only for visualization. For more precise
    calculations consider for example the PyEphem package.

    Parameters
    ----------
    dt: datetime
        Defaults to datetime.utcnow()

    Returns
    -------
    lat, lng: tuple of floats
        Approximate coordinates of the point where the sun is
        in zenith at the time dt.

    """
    if dt is None:
        dt = datetime.datetime.utcnow()

    axial_tilt = 23.4
    ref_solstice = datetime.datetime(2016, 6, 21, 22, 22)
    days_per_year = 365.2425
    seconds_per_day = 24*60*60.0

    days_since_ref = (dt - ref_solstice).total_seconds()/seconds_per_day
    lat = axial_tilt*np.cos(2*np.pi*days_since_ref/days_per_year)
    sec_since_midnight = (dt - datetime.datetime(dt.year, dt.month, dt.day)).seconds
    lng = -(sec_since_midnight/seconds_per_day - 0.5)*360
    return lat, lng


def fill_dark_side(ax, time=None, *args, **kwargs):
    """
    Plot a fill on the dark side of the planet (without refraction).

    Parameters
    ----------
        ax : Matplotlib axes
            The axes to plot on.
        time : datetime
            The time to calculate terminator for. Defaults to datetime.utcnow()
        **kwargs :
            Passed on to Matplotlib's ax.fill()

    """
    lat, lng = sun_pos(time)
    pole_lng = lng
    if lat > 0:
        pole_lat = -90 + lat
        central_rot_lng = 180
    else:
        pole_lat = 90 + lat
        central_rot_lng = 0

    rotated_pole = ccrs.RotatedPole(pole_latitude=pole_lat,
                                    pole_longitude=pole_lng,
                                    central_rotated_longitude=central_rot_lng)

    x = np.empty(360)
    y = np.empty(360)
    x[:180] = -90
    y[:180] = np.arange(-90, 90.)
    x[180:] = 90
    y[180:] = np.arange(90, -90., -1)

    ax.fill(x, y, transform=rotated_pole, **kwargs)

def band_legend(ax,loc='lower center',markerscale=0.5,prop={'size':10},
        title=None,bbox_to_anchor=None,rbn_rx=True,ncdxf=False,ncol=None,band_data=None):

    if band_data is None:
        band_data = BandData()

    handles = []
    labels  = []

    # Force freqs to go low to high regardless of plotting order.
    band_list   = list(band_data.band_dict.keys())
    band_list.sort()
    for band in band_list:
        color   = band_data.band_dict[band]['color']
        label   = band_data.band_dict[band]['freq_name']

        count   = band_data.band_dict[band].get('count')
        if count is not None:
            label += '\n(n={!s})'.format(count)

        handles.append(mpatches.Patch(color=color,label=label))
        labels.append(label)

    fig_tmp = plt.figure()
    ax_tmp = fig_tmp.add_subplot(111)
    ax_tmp.set_visible(False)
    if rbn_rx:
        scat = ax_tmp.scatter(0,0,s=50,**de_prop)
        labels.append('Receiver')
        handles.append(scat)
    if ncdxf:
        scat = ax_tmp.scatter(0,0,s=dxf_leg_size,**dxf_prop)
        labels.append('NCDXF Beacon')
        handles.append(scat)

    if ncol is None:
        ncol = len(labels)
    
    legend = ax.legend(handles,labels,ncol=ncol,loc=loc,markerscale=markerscale,prop=prop,
            title=title,bbox_to_anchor=bbox_to_anchor,scatterpoints=1)
    legend.set_zorder(500)
    plt.close(fig_tmp)
    return legend

def get_bins(lim, bin_size):
    """ Helper function to split a limit into bins of the proper size """
    bins    = np.arange(lim[0], lim[1]+2*bin_size, bin_size)
    return bins

def adjust_axes(ax_0,ax_1):
    """
    Force geospace environment axes to line up with histogram
    axes even though it doesn't have a color bar.
    """
    ax_0_pos    = list(ax_0.get_position().bounds)
    ax_1_pos    = list(ax_1.get_position().bounds)
    ax_0_pos[2] = ax_1_pos[2]
    ax_0.set_position(ax_0_pos)

def regional_filter(region,df,kind='mids'):
    rgnd    = regions[region]
    lat_lim = rgnd['lat_lim']
    lon_lim = rgnd['lon_lim']

    if kind == 'mids':
        tf_md_lat   = np.logical_and(df.md_lat >= lat_lim[0], df.md_lat < lat_lim[1])
        tf_md_long  = np.logical_and(df.md_long >= lon_lim[0], df.md_long < lon_lim[1])
        tf_0        = np.logical_and(tf_md_lat,tf_md_long)
        tf          = tf_0
        df          = df[tf].copy()
    elif kind == 'endpoints':
        tf_rx_lat   = np.logical_and(df.rx_lat >= lat_lim[0], df.rx_lat < lat_lim[1])
        tf_rx_long  = np.logical_and(df.rx_long >= lon_lim[0], df.rx_long < lon_lim[1])
        tf_rx       = np.logical_and(tf_rx_lat,tf_rx_long)

        tf_tx_lat   = np.logical_and(df.tx_lat >= lat_lim[0], df.tx_lat < lat_lim[1])
        tf_tx_long  = np.logical_and(df.tx_long >= lon_lim[0], df.tx_long < lon_lim[1])
        tf_tx       = np.logical_and(tf_tx_lat,tf_tx_long)
        tf          = np.logical_or(tf_rx,tf_tx)

        df          = df[tf].copy()

    return df

def load_spots_csv(date_str,data_sources=[1,2],loc_sources=None,
        rgc_lim=None,filter_region=None,filter_region_kind='mids'):
    """
    Load spots from CSV file and filter for network/location source quality.
    Also provide range and regional filtering, compute midpoints, ut_hrs,
    and slt_mid.

    data_sources: list, i.e. [1,2]
        0: dxcluster
        1: WSPRNet
        2: RBN

    loc_sources: list, i.e. ['P','Q']
        P: user Provided
        Q: QRZ.com or HAMCALL
        E: Estimated using prefix
    """

    hdf_path = "data/spot_csvs/{}.hdf".format(date_str)
    csv_path = "data/spot_csvs/{}.csv.bz2".format(date_str)
    if os.path.exists(hdf_path):
        df  = pd.read_hdf(hdf_path)
    elif os.path.exists(csv_path):
        df  = pd.read_csv(csv_path,parse_dates=['occurred'])
    else:
        return

    # Return if no data.
    if len(df) == 0:
        return

    # Select spotting networks
    if data_sources is not None:
        tf  = df.source.map(lambda x: x in data_sources)
        df  = df[tf].copy()

    if len(df) == 0:
        return

    # Filter location source
    if loc_sources is not None:
        tf  = df.tx_loc_source.map(lambda x: x in loc_sources)
        df  = df[tf].copy()

        tf  = df.rx_loc_source.map(lambda x: x in loc_sources)
        df  = df[tf].copy()

    if len(df) == 0:
        return

    # Path Length Filtering
    if rgc_lim is not None:
        tf  = np.logical_and(df['dist_Km'] >= rgc_lim[0],
                             df['dist_Km'] <  rgc_lim[1])
        df  = df[tf].copy()

    if len(df) == 0:
        return

    midpoints       = geopack.midpoint(df["tx_lat"], df["tx_long"], df["rx_lat"], df["rx_long"])
    df['md_lat']    = midpoints[0]
    df['md_long']   = midpoints[1]

    # Regional Filtering
    if filter_region is not None:
        df_raw  = df.copy()
        df      = regional_filter(filter_region,df,kind=filter_region_kind)

    if len(df) == 0:
        return

    df["ut_hrs"]    = df['occurred'].map(lambda x: x.hour + x.minute/60. + x.second/3600.)
    df['slt_mid']   = (df['ut_hrs'] + df['md_long']/15.) % 24.

    return df

def list_sources(df,count=True,bands=None):
    srcs    = df.source.unique()
    srcs.sort()

    if bands is not None:
        tf  = df['band'].apply(lambda x: x in bands)
        df  = df[tf].copy()

    names   = []
    for src in srcs:
        name    = sources[src].get('name')
        if count:
            cnt     = np.sum(df.source==src)
            name    = '{!s} (N={!s})'.format(name,cnt)
        names.append(name)
    return names

def count_sources(df,bands=None):
    """
    Like list sources, but returns a dict.
    """
    srcs    = df.source.unique()
    srcs.sort()

    if bands is not None:
        tf  = df['band'].apply(lambda x: x in bands)
        df  = df[tf].copy()

    cnt_dct = {}
    for src in srcs:
        name    = sources[src].get('name')
        cnt     = np.sum(df.source==src)
        cnt_dct[name] = cnt
    return cnt_dct

def sunAzEl(dates,lat,lon):
    azs, els = [], []
    for date in dates:
        jd    = calcSun.getJD(date) 
        t     = calcSun.calcTimeJulianCent(jd)
        ut    = ( jd - (int(jd - 0.5) + 0.5) )*1440.
        az,el = calcSun.calcAzEl(t, ut, lat, lon, 0.)
        azs.append(az)
        els.append(el)
    return azs,els

def calc_solar_zenith(sTime,eTime,sza_lat,sza_lon,minutes=5):
    sza_dts = [sTime]
    while sza_dts[-1] < eTime:
        sza_dts.append(sza_dts[-1]+datetime.timedelta(minutes=minutes))

    azs,els = sunAzEl(sza_dts,sza_lat,sza_lon)
    
    sza = pd.DataFrame({'els':els},index=sza_dts)
    return sza

def calc_solar_zenith_region(sTime,eTime,region='World'):
    rgn     = regions.get(region)
    lat_lim = rgn.get('lat_lim')
    lon_lim = rgn.get('lon_lim')

    sza_lat = (lat_lim[1]-lat_lim[0])/2. + lat_lim[0]
    sza_lon = (lon_lim[1]-lon_lim[0])/2. + lon_lim[0]

    sza     = calc_solar_zenith(sTime,eTime,sza_lat,sza_lon)

    return (sza,sza_lat,sza_lon)

class MyBz2(object):
    def __init__(self,fname):
        if fname[-4:] == '.bz2':
            self.bz2_name   = fname
            self.unc_name   = fname[:-4]
        else:
            self.bz2_name   = fname + '.bz2'
            self.unc_name   = fname

    def compress(self):
        cmd = 'bzip2 -f {!s}'.format(self.unc_name)
        os.system(cmd)

    def uncompress(self):
        cmd = 'bunzip2 -kf {!s}'.format(self.bz2_name)
        os.system(cmd)

    def remove(self):
        cmd = 'rm {!s}'.format(self.unc_name)
        os.system(cmd)
