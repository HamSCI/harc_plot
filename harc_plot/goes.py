#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (C) 2014  VT SuperDARN Lab
# Full license can be found in LICENSE.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""GOES module

A module for working with GOES data.

Module Author:: N.A. Frissell, 6 Sept 2014

Functions
--------------------------------------------------------
read_goes       download GOES data
goes_plot       plot GOES data
classify_flare  convert GOES data to string classifier
flare_value     convert string classifier to lower bound
find_flares     find flares in a certain class
--------------------------------------------------------
"""

import logging
import os
import re
import shutil
import datetime
import fnmatch
import glob
import ftplib
import calendar
import urllib.request
import urllib.error
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import netCDF4

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def add_months(sourcedate,months=1):
    """Add 1 month to a datetime object.

    Parameters
    ----------
    sourcedate : datetime

    months : Optional[int]

    """
    month   = int(sourcedate.month - 1 + months)
    year    = int(sourcedate.year + month / 12)
    month   = int(month % 12 + 1)
    day     = min(sourcedate.day,calendar.monthrange(year,month)[1])
    return datetime.date(year,month,day)

def ut_hours(dt_obj):
    ut_hr   = dt_obj.hour + dt_obj.minute/60. + dt_obj.second/3600.
    return ut_hr

def dtGreg_to_datetime(dtg):
    pdt = datetime.datetime(dtg.year,dtg.month,dtg.day,dtg.hour,dtg.minute,dtg.second)
    return pdt

def read_goes(sTime,eTime=None,sat_nr=15,data_dir='data/goes'):
    """Download GOES X-Ray Flux data from the NOAA FTP Site and return a
    dictionary containing the metadata and a dataframe.

    Parameters
    ----------
    sTime : datetime.datetime
        Starting datetime for data.
    eTime : Optional[datetime.datetime]
        Ending datetime for data.  If None, eTime will be set to sTime
        + 1 day.
    sat_nr : Optional[int]
        GOES Satellite number.  Defaults to 15.

    Returns
    -------
    Dictionary containing metadata, pandas dataframe with GOES data.

    Notes
    -----
    Data is downloaded from
    ftp://satdat.ngdc.noaa.gov/sem/goes/data/avg/2014/08/goes15/netcdf/

    Currently, 1-m averaged x-ray spectrum in two bands
    (0.5-4.0 A and 1.0-8.0 A).

    Example
    -------
        goes_data = read_goes(datetime.datetime(2014,6,21))
      
    written by N.A. Frissell, 6 Sept 2014

    """

    if eTime is None: eTime = sTime + datetime.timedelta(days=1)

    #Determine which months of data to download.
    ym_list     = [datetime.date(sTime.year,sTime.month,1)]
    eMonth      = datetime.date(eTime.year,eTime.month,1)
    while ym_list[-1] < eMonth:
        ym_list.append(add_months(ym_list[-1]))

    # Download Files from NOAA FTP #################################################
    host        = 'satdat.ngdc.noaa.gov'

    if data_dir.endswith('/'): data_dir = data_dir[:-1]
    
    try:
        os.makedirs(data_dir)
    except:
        pass

    #rem_file    = '/sem/goes/data/avg/2014/08/goes15/netcdf/g15_xrs_1m_20140801_20140831.nc' #Example file.
    file_paths  = []
    try:
        for myTime in ym_list:
            #Check to see if we already have a matcing file...
            local_files = glob.glob(os.path.join(data_dir,'g{sat_nr:02d}_xrs_1m_{year:d}{month:02d}*.nc'.format(year=myTime.year,month=myTime.month,sat_nr=sat_nr)))
            if len(local_files) > 0:
                logging.info('Using locally cached file: {0}'.format(local_files[0]))
                file_paths.append(local_files[0])
                continue

            rem_path    = '/sem/goes/data/avg/{year:d}/{month:02d}/goes{sat_nr:d}/netcdf'.format(year=myTime.year,month=myTime.month,sat_nr=sat_nr)
            ftp         = ftplib.FTP(host,'anonymous','@anonymous')
            s           = ftp.cwd(rem_path)
            file_list   = ftp.nlst()
            dl_list     = [x for x in file_list if fnmatch.fnmatch(x,'g*_xrs_1m_*')]
            filename    = dl_list[0]

            #Figure out where to save the file locally...
            file_path   = os.path.join(data_dir,filename)
            file_paths.append(file_path)

            #Go retrieve the file...
            logging.info('Downloading {0}...'.format(filename))
            ftp.retrbinary('RETR {0}'.format(filename), open(file_path, 'wb').write)
    except:
        print('GOES Data ERROR.')
        return

    # Load data into memory. #######################################################
    df_xray     = None
    df_orbit    = None

    data_dict   = {}
    data_dict['metadata']               = {}
    data_dict['metadata']['variables']  = {}

    for file_path in file_paths:
        nc = netCDF4.Dataset(file_path)

        #Put metadata into dictionary.
        fn  = os.path.basename(file_path)
        data_dict['metadata'][fn] = {}
        md_keys = ['NOAA_scaling_factors','archiving_agency','creation_date','end_date',
                   'institution','instrument','originating_agency','satellite_id','start_date','title']
        for md_key in md_keys:
            try:
                data_dict['metadata'][fn][md_key] = getattr(nc,md_key)
            except:
                pass

        #Store Orbit Data
        tt = nc.variables['time_tag_orbit']
        jd = np.array(netCDF4.num2date(tt[:],tt.units))

        orbit_vars = ['west_longitude','inclination']
        data    = {}
        for var in orbit_vars:
            data[var] = nc.variables[var][:]
        
        df_tmp = pd.DataFrame(data,index=jd)
        if df_orbit is None:
            df_orbit = df_tmp
        else:
            df_orbit = df_orbit.append(df_tmp)

        #Store X-Ray Data
        tt = nc.variables['time_tag']
        jd = np.array(netCDF4.num2date(tt[:],tt.units))

        myVars = ['A_QUAL_FLAG','A_NUM_PTS','A_AVG','B_QUAL_FLAG','B_NUM_PTS','B_AVG']
        data = {}
        for var in myVars:
            data[var] = nc.variables[var][:]
        
        df_tmp = pd.DataFrame(data,index=jd)
        if df_xray is None:
            df_xray = df_tmp
        else:
            df_xray = df_xray.append(df_tmp)

        keys    = ['A_AVG','B_AVG']
        for key in keys:
            tf  = df_xray[key]  == -99999
            df_xray[key][tf]    = np.nan

        #Store info about units
        for var in (myVars+orbit_vars):
            data_dict['metadata']['variables'][var] = {}
            var_info_keys = ['description','dtype','long_label','missing_value','nominal_max','nominal_min','plot_label','short_label','units']
            for var_info_key in var_info_keys:
                try:
                    data_dict['metadata']['variables'][var][var_info_key] = getattr(nc.variables[var],var_info_key)
                except:
                    pass

    df_xray     = df_xray[np.logical_and(df_xray.index >= sTime,df_xray.index < eTime)]
    df_orbit    = df_orbit[np.logical_and(df_orbit.index >= sTime,df_orbit.index < eTime)]

    df_orbit['longitude']   = -1*df_orbit['west_longitude']
    del df_orbit['west_longitude']

    df_xray = pd.concat([df_xray,df_orbit],axis=1)
    for key in df_orbit.keys():
        df_xray[key].fillna(method='ffill',inplace=True)

    df_xray['ut_hrs']   = df_xray.index.map(ut_hours)
    df_xray['slt_mid']  = (df_xray['ut_hrs'] + df_xray['longitude']/15.) % 24.

    data_dict['xray']   = df_xray
    data_dict['orbit']  = df_orbit

    data_dict['xray'].index  = [dtGreg_to_datetime(x) for x in data_dict['xray'].index]
    data_dict['orbit'].index = [dtGreg_to_datetime(x) for x in data_dict['orbit'].index]
    return data_dict

# NCEI HTTPS archives. read_goes() above targets ftp://satdat.ngdc.noaa.gov, which NOAA
# retired along with anonymous FTP, so it can no longer download anything -- it fails
# closed, printing 'GOES Data ERROR.' and returning None. These two archives replace it
# and between them span 2009 to present.
#
# Two eras with different file layouts and variable names:
#   pre-R  (GOES 8-15):  one netCDF per month, vars A_AVG / B_AVG, time_tag
#   GOES-R (GOES 16-19): one netCDF per day,   vars xrsa_flux / xrsb_flux, time
# read_goes_ncei() normalizes both to the A_AVG / B_AVG names the plotting and
# flare-finding routines here already expect.
_NCEI_PRE_R_DIR  = ('https://www.ncei.noaa.gov/data/goes-space-environment-monitor/'
                    'access/avg/{year:04d}/{month:02d}/goes{sat:02d}/netcdf/')
_NCEI_GOES_R_DIR = ('https://data.ngdc.noaa.gov/platforms/solar-space-observing-satellites/'
                    'goes/goes{sat:02d}/l2/data/xrsf-l2-avg1m/{year:04d}/{month:02d}/')

# GOES-R quality flags: 0 good_data, 1 eclipse, 2 bad_data, 4 interpolated.
# Eclipse and bad_data are dropped; interpolated is kept.
_GOES_R_BAD_FLAGS = (1, 2)

def goes_xrs_candidates(year):
    """Prioritized [(sat_nr, era)] candidates carrying XRS for a given year.

    Ordered most- to least-preferred; read_goes_ncei() tries them in turn and takes
    the first that actually has data, so these windows only need to be roughly right.
    'era' selects the archive layout: 'pre_r' or 'goes_r'.

    Ordering follows the *primary operational* satellite for each epoch, because that
    is the one whose fluxes reproduce the standard flare classifications. This matters:
    for the 2017-09-06 flare, GOES-15 (operational primary) peaks at 9.33e-4 W/m^2,
    i.e. X9.3, matching the accepted classification, while GOES-16 -- then still in
    post-launch test and reporting on a different flux scale -- peaks at 5.17e-4, i.e.
    X5.2. GOES-13 reads X10.4. So GOES-15 is preferred through its 2020-03 end of life
    even after GOES-16 data becomes available.

    Corollary worth remembering when comparing events across the archive: X-class
    numbers are NOT directly comparable between the pre-R and GOES-R eras. Flare
    magnitudes from 2015 and 2024 sit on different scales.
    """
    if year >= 2025: return [(19,'goes_r'),(18,'goes_r'),(16,'goes_r')]
    if year >= 2023: return [(18,'goes_r'),(16,'goes_r')]
    # GOES-15 XRS ended 2020-03-04, so 2020 leads with the GOES-R series but keeps 15
    # as a fallback for January through early March.
    if year >= 2020: return [(16,'goes_r'),(17,'goes_r'),(18,'goes_r'),(15,'pre_r')]
    if year >= 2010: return [(15,'pre_r'),(14,'pre_r'),(13,'pre_r')]
    return [(14,'pre_r'),(11,'pre_r'),(12,'pre_r'),(10,'pre_r')]

def _ncei_listing(url,timeout=60):
    """Filenames in an NCEI Apache directory index. [] if unreachable."""
    try:
        with urllib.request.urlopen(url,timeout=timeout) as resp:
            html = resp.read().decode('utf-8','replace')
    except (urllib.error.URLError,OSError) as e:
        logging.info('NCEI listing failed for {0}: {1}'.format(url,e))
        return []
    # Apache indexes repeat each name in href and link text; dedupe but keep order.
    names,seen = [],set()
    for name in re.findall(r'href="([^"?/][^"]*\.nc)"',html):
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names

def _ncei_download(url,dest,timeout=300):
    """Download to dest, returning True on success.

    Writes to a .part file and renames, so an interrupted download never leaves a
    truncated file that later runs would treat as a valid cache hit.
    """
    part = dest + '.part'
    try:
        with urllib.request.urlopen(url,timeout=timeout) as resp, open(part,'wb') as fh:
            shutil.copyfileobj(resp,fh)
        os.replace(part,dest)
        return True
    except (urllib.error.URLError,OSError) as e:
        logging.info('NCEI download failed for {0}: {1}'.format(url,e))
        if os.path.exists(part):
            os.remove(part)
        return False

def _months_in_range(sTime,eTime):
    """First-of-month dates covering [sTime,eTime)."""
    months  = [datetime.date(sTime.year,sTime.month,1)]
    end     = datetime.date(eTime.year,eTime.month,1)
    while months[-1] < end:
        months.append(add_months(months[-1]))
    return months

def _read_pre_r_nc(file_path):
    """(DataFrame[A_AVG,B_AVG], metadata dict) from a pre-R monthly XRS file."""
    with netCDF4.Dataset(file_path) as nc:
        tt      = nc.variables['time_tag']
        index   = pd.to_datetime(netCDF4.num2date(
                    tt[:],tt.units,only_use_cftime_datetimes=False))
        data    = {}
        for out_col,var in (('A_AVG','A_AVG'),('B_AVG','B_AVG')):
            arr = np.ma.filled(nc.variables[var][:].astype(float),np.nan)
            arr[arr == -99999] = np.nan
            data[out_col] = arr
        md = {k:getattr(nc,k,None) for k in
                ('institution','satellite_id','instrument','creation_date')}
    return pd.DataFrame(data,index=index),md

def _read_goes_r_nc(file_path):
    """(DataFrame[A_AVG,B_AVG], metadata dict) from a GOES-R daily XRS file.

    xrsa_flux/xrsb_flux are renamed to A_AVG/B_AVG so downstream code that predates
    the GOES-R series keeps working unchanged. The wavelength bands are equivalent:
    XRS-A is 0.05-0.4 nm, XRS-B is 0.1-0.8 nm.
    """
    with netCDF4.Dataset(file_path) as nc:
        tt      = nc.variables['time']
        index   = pd.to_datetime(netCDF4.num2date(
                    tt[:],tt.units,only_use_cftime_datetimes=False))
        data    = {}
        for out_col,var,flag_var in (('A_AVG','xrsa_flux','xrsa_flag'),
                                     ('B_AVG','xrsb_flux','xrsb_flag')):
            arr = np.ma.filled(nc.variables[var][:].astype(float),np.nan)
            if flag_var in nc.variables:
                flags = np.ma.filled(nc.variables[flag_var][:],0)
                arr[np.isin(flags,_GOES_R_BAD_FLAGS)] = np.nan
            data[out_col] = arr
        md = {'institution': getattr(nc,'institution',None),
              'satellite_id': getattr(nc,'platform',None),
              'instrument':   getattr(nc,'instrument',None),
              'creation_date':getattr(nc,'date_created',None)}
    return pd.DataFrame(data,index=index),md

def read_goes_ncei(sTime,eTime=None,sat_nr=None,era=None,data_dir='data/goes'):
    """Load 1-minute GOES XRS data from the NOAA NCEI HTTPS archives.

    Drop-in replacement for read_goes() with the same return contract, but backed by
    endpoints that still exist and spanning 2009 to present rather than only the
    GOES-13/15 era. Downloaded files are cached in data_dir and reused.

    X-ray flux resolves individual solar flares (1-minute cadence, and the 0.1-0.8 nm
    B channel is what defines the A/B/C/M/X classes). This is the observable to use for
    flare timing -- F10.7 cannot serve that purpose at any cadence, because DRAO
    Penticton determines it once per day.

    Parameters
    ----------
    sTime : datetime.datetime
        Start of the interval.
    eTime : Optional[datetime.datetime]
        End of the interval, exclusive. Defaults to sTime + 1 day.
    sat_nr : Optional[int]
        Force a specific GOES satellite. Default None auto-selects per year via
        goes_xrs_candidates(), trying each until one yields data.
    era : Optional[str]
        'pre_r' or 'goes_r'. Only consulted when sat_nr is given explicitly.
    data_dir : Optional[str]
        Local cache directory.

    Returns
    -------
    Dict with the same shape read_goes() returns: 'xray' (DataFrame indexed by
    datetime with A_AVG, B_AVG, ut_hrs) and 'metadata'. None if no data could be
    obtained for the interval.

    Note that 'orbit' data, and therefore the 'slt_mid' column, are not provided --
    the GOES-R XRS product does not carry the spacecraft ephemeris that read_goes()
    pulled from the pre-R files.
    """
    if eTime is None:
        eTime = sTime + datetime.timedelta(days=1)

    os.makedirs(data_dir,exist_ok=True)

    if sat_nr is not None:
        # An explicit satellite with no era stated: infer from the series number.
        candidates = [(sat_nr, era if era is not None
                               else ('goes_r' if sat_nr >= 16 else 'pre_r'))]
    else:
        # Candidates are year-keyed; a window spanning a year boundary needs both.
        candidates = []
        for year in sorted({d.year for d in _months_in_range(sTime,eTime)}):
            for cand in goes_xrs_candidates(year):
                if cand not in candidates:
                    candidates.append(cand)

    frames,metadata = [],{}
    used_sats = set()
    for cand_sat,cand_era in candidates:
        got_any = False
        for month in _months_in_range(sTime,eTime):
            if cand_era == 'pre_r':
                base    = _NCEI_PRE_R_DIR.format(year=month.year,month=month.month,sat=cand_sat)
                pattern = 'g{0:02d}_xrs_1m_*.nc'.format(cand_sat)
                reader  = _read_pre_r_nc
                wanted  = None                  # monthly file; take whatever is there
            else:
                base    = _NCEI_GOES_R_DIR.format(year=month.year,month=month.month,sat=cand_sat)
                pattern = 'dn_xrsf-l2-avg1m_g{0:02d}_d*.nc'.format(cand_sat)
                reader  = _read_goes_r_nc
                # Daily files: fetch only the days actually in range.
                wanted  = set()
                day     = datetime.date(sTime.year,sTime.month,sTime.day)
                while day < eTime.date() + datetime.timedelta(days=1):
                    wanted.add(day.strftime('%Y%m%d'))
                    day += datetime.timedelta(days=1)

            cached = [os.path.basename(p) for p in
                      glob.glob(os.path.join(data_dir,pattern))]
            names  = [n for n in _ncei_listing(base) if fnmatch.fnmatch(n,pattern)]
            names  = sorted(set(names) | set(cached))
            if wanted is not None:
                names = [n for n in names
                         if any('_d{0}_'.format(d) in n for d in wanted)]

            for name in names:
                path = os.path.join(data_dir,name)
                if not os.path.exists(path):
                    if not _ncei_download(base + name,path):
                        continue
                try:
                    df,md = reader(path)
                except Exception as e:
                    # A corrupt cached file should not kill the whole request; drop it
                    # so the next run re-downloads.
                    warnings.warn('{0}: unreadable GOES file ({1}: {2}); removing from '
                                  'cache'.format(name,type(e).__name__,e),RuntimeWarning)
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                    continue
                frames.append(df)
                metadata[name] = md
                got_any = True

        if got_any:
            used_sats.add(cand_sat)
            break   # first candidate with data wins; don't stack satellites

    if not frames:
        warnings.warn('No GOES XRS data available from NCEI for {0} to {1} '
                      '(tried satellites {2}).'.format(
                          sTime,eTime,[c[0] for c in candidates]),RuntimeWarning)
        return

    df_xray = pd.concat(frames).sort_index()
    df_xray = df_xray[~df_xray.index.duplicated(keep='first')]
    df_xray = df_xray[(df_xray.index >= sTime) & (df_xray.index < eTime)]
    if len(df_xray) == 0:
        warnings.warn('GOES XRS files found but no samples fall in {0} to {1}.'.format(
                          sTime,eTime),RuntimeWarning)
        return

    df_xray['ut_hrs'] = df_xray.index.map(ut_hours)

    data_dict = {'xray':df_xray,'metadata':metadata}
    data_dict['metadata']['variables'] = {
        'A_AVG': {'long_label':'x-ray (0.05-0.4 nm) irradiance','units':'W/m^2'},
        'B_AVG': {'long_label':'x-ray (0.1-0.8 nm) irradiance', 'units':'W/m^2'},
    }
    data_dict['sat_nrs'] = sorted(used_sats)
    return data_dict

def goes_plot_hr(goes_data,ax,var_tags = ['B_AVG'],xkey='ut_hr',xlim=(0,24),ymin=1e-9,ymax=1e-2,
        legendSize=10,legendLoc=None,labels=None,**kwargs):
    """Plot GOES X-Ray Data.

    Parameters
    ----------
    goes_data   : dict
        data dictionary returned by read_goes()
    ax          : matplotlib.axes
    var_tags    : List of variables, i.e. ['A_AVG','B_AVG']
        'A_AVG' --> X-Ray (0.05-0.4 nm) Irradiance [W/m^2]
        'B_AVG' --> X-Ray (0.1 -0.8 nm) Irradiance [W/m^2]
    ymin : Optional[float]
        Y-Axis minimum limit
    ymax : Optional[float]
        Y-Axis maximum limit
    legendSize : Optional[int]
        Character size of the legend
    legendLoc : Optional[ ]

    Returns
    -------
    fig : matplotlib.figure
        matplotlib figure object that was plotted to

    Notes
    -----
    If a matplotlib figure currently exists, it will be modified
    by this routine.  If not, a new one will be created.

    Written by Nathaniel Frissell 2014 Sept 06

    """
    xx  = goes_data['xray'][xkey]
    for var_inx,var_tag in enumerate(var_tags):
        if labels is None:
            label   = goes_data['metadata']['variables'][var_tag]['long_label']
        else:
            label   = labels[var_inx]
        ax.plot(xx,goes_data['xray'][var_tag],label=label,**kwargs)

    ax.set_xlim(xlim)

    #Label Flare classes
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    classes = ['A', 'B', 'C', 'M', 'X']
    decades = [  8,   7,   6,   5,   4]

    for cls,dec in zip(classes,decades):
        ax.text(1.01,2.5*10**(-dec),cls,transform=trans,fontdict={'size':14})

    #Format the y-axis
    ax.set_ylabel(r'W m$^{-2}$')
    ax.set_yscale('log')
    ax.set_ylim(1e-9,1e-2)

#    minor_ticks = np.array([1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2])
#    ax.set_yticks(minor_ticks,minor=True)
#
#    minor_ticks = np.array([])
#    ax.set_xticks(minor_ticks,minor=True)
#
#    ax.minorticks_on()
#    ax.grid(True,which='minor')
    ax.grid(True,which='major')
    ax.legend(prop={'size':legendSize},numpoints=1,loc=legendLoc)

    file_keys = list(goes_data['metadata'].keys()) 
    file_keys.remove('variables')
    file_keys.sort()
    md      = goes_data['metadata'][file_keys[-1]]
    title   = ' '.join([md['institution'],md['satellite_id'],'-',md['instrument']])
#    ax.set_title(title)

def goes_plot(goes_data,sTime=None,eTime=None,var_tags = ['B_AVG'],labels=None,ymin=1e-9,ymax=1e-2,legendSize=None,legendLoc=None,ax=None,**kwargs):
    """Plot GOES X-Ray Data.

    Parameters
    ----------
    goes_data : dict
        data dictionary returned by read_goes()
    sTime : Optional[datetime.datetime]
        object for start of plotting.
    eTime : Optional[datetime.datetime]
        object for end of plotting.
    ymin : Optional[float]
        Y-Axis minimum limit
    ymax : Optional[float]
        Y-Axis maximum limit
    legendSize : Optional[int]
        Character size of the legend
    legendLoc : Optional[ ]
    ax : Optional[ ]

    Returns
    -------
    fig : matplotlib.figure
        matplotlib figure object that was plotted to

    Notes
    -----
    If a matplotlib figure currently exists, it will be modified
    by this routine.  If not, a new one will be created.

    Written by Nathaniel Frissell 2014 Sept 06

    """
    if ax is None:
        fig     = plt.figure(figsize=(10,6))
        ax      = fig.add_subplot(111)
    else:
        fig     = ax.get_figure()

    if sTime is None: sTime = goes_data['xray'].index.min()
    if eTime is None: eTime = goes_data['xray'].index.max()

    xx = goes_data['xray'].index
    for var_inx,var_tag in enumerate(var_tags):
        if labels is None:
            label   = goes_data['metadata']['variables'][var_tag]['long_label']
        else:
            label   = labels[var_inx]
        ax.plot(xx,goes_data['xray'][var_tag],label=label,**kwargs)

#    #Format the x-axis
#    if eTime - sTime > datetime.timedelta(days=1):
#        ax.xaxis.set_major_formatter(
#                matplotlib.dates.DateFormatter('%H%M\n%d %b %Y')
#                )
#    else:
#        ax.xaxis.set_major_formatter(
#                matplotlib.dates.DateFormatter('%H%M')
#                )

    sTime_label = sTime.strftime('%Y %b %d')
    eTime_label = eTime.strftime('%Y %b %d')
    if sTime_label == eTime_label:
        time_label = sTime_label
    else:
        time_label = sTime_label + ' - ' + eTime_label

#    ax.set_xlabel('\n'.join([time_label,'Time [UT]']))
    ax.set_xlim(sTime,eTime)

    #Label Flare classes
    trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    classes = ['A', 'B', 'C', 'M', 'X', '']
    decades = [  8,   7,   6,   5,   4,  3]

    size    = matplotlib.rcParams['ytick.labelsize']
    for cls,dec in zip(classes,decades):
        ax.text(1.01,10**(-dec),cls,transform=trans,fontdict={'size':size},va='center')
        ax.axhline(10**(-dec),ls='--',color='0.8')

    #Format the y-axis
    ax.set_ylabel(r'W m$^{-2}$')
    ax.set_yscale('log')
    ax.set_ylim(1e-9,1e-2)

    ax.grid()
    if legendSize is None:
        legendSize = matplotlib.rcParams['legend.fontsize']
    ax.legend(prop={'size':legendSize},numpoints=1,loc=legendLoc)

    file_keys = list(goes_data['metadata'].keys()) 
    file_keys.remove('variables')
    file_keys.sort()
    md      = goes_data['metadata'][file_keys[-1]]
    title   = ' '.join([md['institution'],md['satellite_id'],'-',md['instrument']])
#    ax.set_title(title)

def __split_sci(value):
    """Split scientific notation into (coefficient,power).
    This is a private function that currently only works on scalars.

    Parameters
    ----------
    value :
        numerical value

    Returns
    -------
    coefficient : float

    Written by Nathaniel Frissell 2014 Sept 07

    """
    s   = '{0:e}'.format(value)
    s   = s.split('e')
    return (float(s[0]),float(s[1]))


def classify_flare(value):
    """Convert GOES X-Ray flux into a string flare classification.
    You should use the 1-8 Angstrom band for classification [1] 
    (B_AVG in the NOAA data files).

    A 0.001 W/m**2 measurement in the 1-8 Angstrom band is classified as an X10 flare..

    This function currently only works on scalars.

    Parameters
    ----------
    value :
        numerical value of the GOES 1-8 Angstrom band X-Ray Flux in W/m^2.

    Returns
    -------
    flare_class : string
        class of solar flare

    References
    ----------
    [1] http://www.spaceweatherlive.com/en/help/the-classification-of-solar-flares

    Example
    -------
        flare_class = classify_flare(0.001)

    Written by Nathaniel Frissell 2014 Sept 07

    """
    coef, power = __split_sci(value)

    if power < -7:
        letter  = 'A'
        coef    = value / 1e-8
    elif power >= -7 and power < -6:
        letter  = 'B'
    elif power >= -6 and power < -5:
        letter  = 'C'
    elif power >= -5 and power < -4:
        letter  = 'M'
    elif power >= -4:
        letter  = 'X'
        coef    = value / 1.e-4

    flare_class = '{0}{1:.1f}'.format(letter,coef)
    return flare_class


def flare_value(flare_class):
    """Convert a string solar flare class [1] into the lower bound in W/m**2 of the 
    1-8 Angstrom X-Ray Band for the GOES Spacecraft.

    An 'X10' flare = 0.001 W/m**2.

    This function currently only works on scalars.

    Parameters
    ----------
    flare_class : string
        class of solar flare (e.g. 'X10')

    Returns
    -------
    value : float
        numerical value of the GOES 1-8 Angstrom band X-Ray Flux in W/m**2.

    References
    ----------
    [1] See http://www.spaceweatherlive.com/en/help/the-classification-of-solar-flares

    Example
    -------
        value = flare_value('X10')

    Written by Nathaniel Frissell 2014 Sept 07

    """
    flare_dict  = {'A':-8, 'B':-7, 'C':-6, 'M':-5, 'X':-4} 
    letter      = flare_class[0]
    power       = flare_dict[letter.upper()]
    coef        = float(flare_class[1:])
    value       = coef * 10.**power
    return value


def find_flares(goes_data,window_minutes=60,min_class='X1',sTime=None,eTime=None,tmp_dir='data'):
    """Find flares of a minimum class in a GOES data dict created by read_goes().
    This works with 1-minute averaged GOES data.

    Classifications are based on the 1-8 Angstrom X-Ray Band for the GOES Spacecraft.[1]

    Parameters
    ----------
    goes_data : dict
        GOES data dict created by read_goes()
    window_minutes : Optional[int]
        Window size to look for peaks in minutes.
        I.E., if window_minutes=60, then no more than 1 flare will be found 
        inside of a 60 minute window.
    min_class : Optional[str]
        Only flares >= to this class will be reported. Use a
        format such as 'M2.3', 'X1', etc.
    sTime : Optional[datetime.datetime]
        Only report flares at or after this time.  If None, the earliest
        available time in goes_data will be used.
    eTime : Optional[datetime.datetime]
        Only report flares before this time.  If None, the last
        available time in goes_data will be used.

    Returns
    -------
    flares : Pandas dataframe listing:
        * time of flares
        * GOES 1-8 Angstrom band x-ray flux
        * Classification of flare

    References
    ----------
    [1] See http://www.spaceweatherlive.com/en/help/the-classification-of-solar-flares

    Example
    -------
        sTime       = datetime.datetime(2014,1,1)
        eTime       = datetime.datetime(2014,6,30)
        sat_nr      = 15 # GOES15
        goes_data   = read_goes(sTime,eTime,sat_nr)
        flares = find_flares(goes_data,window_minutes=60,min_class='X1')

    Written by Nathaniel Frissell 2014 Sept 09

    """
    df  = goes_data['xray']

    if sTime is None: sTime = df.index.min()
    if eTime is None: eTime = df.index.max()

    # Figure out when big solar flares are.
    time_delta      = datetime.timedelta(minutes=window_minutes)
    time_delta_half = datetime.timedelta( minutes=(window_minutes/2.) )

    window_center = [sTime + time_delta_half ]
    while window_center[-1] < eTime:
        window_center.append(window_center[-1] + time_delta)

    b_avg = df['B_AVG']

    keys = []
    for win in window_center:
        sWin = win - time_delta_half
        eWin = win + time_delta_half

        # An all-NaN window (eclipse, data gap, flagged-bad stretch) makes idxmax
        # return pd.NaT, and `NaT is np.nan` is False -- so the old identity check let
        # NaT through into `keys` and the b_avg[keys] lookup below raised
        # "KeyError: '[NaT] not in index'". Test for emptiness up front, which also
        # avoids the deprecated all-NA idxmax call.
        win_vals = b_avg[sWin:eWin]
        if len(win_vals) == 0 or win_vals.isna().all():
            continue
        try:
            idx_max = win_vals.idxmax()
        except (ValueError,TypeError):
            continue
        if pd.isna(idx_max):
            continue
        keys.append(idx_max)
        
    df_win      = pd.DataFrame({'B_AVG':b_avg[keys]})

    flares      = df_win[df_win['B_AVG'] >= flare_value(min_class)]

    # Remove flares that are really window edges instead of local maxima.
    drop_list = []
    for inx_0,key_0 in enumerate(flares.index):
        if inx_0 == len(flares.index)-1: break

        inx_1   = inx_0 + 1
        key_1   = flares.index[inx_1]

        flare_0 = np.unique(flares['B_AVG'][key_0])
        flare_1 = np.unique(flares['B_AVG'][key_1])

        arg_min = np.argmin([flare_0,flare_1])
        key_min = [key_0,key_1][arg_min]

        vals_between = b_avg[key_0:key_1]

        if np.unique(flares['B_AVG'][key_min]) <= vals_between.min():
            drop_list.append(key_min)

    if drop_list != []:
        flares  = flares.drop(drop_list)
    flares  = flares.copy()
    flares['class'] = list(map(classify_flare,flares['B_AVG']))

    return flares


if __name__ == '__main__':

    # Flare Classification Test ####################################################
    print('')
    print('Flare classification test.')
    flares = ['A5.5', 'B4.0', 'X11.1']
    values = [5.5e-8, 4.0e-7, 11.1e-4]

    test_results = []
    for flare,value in zip(flares,values):
        print(('  Testing classify_flare() with {0} ({1:.1e} W/m**2) flare...'.format(flare,value)))

        test_flare = classify_flare(value)
        print(('    classify_flare({0:.1e}) = {1}'.format(value,test_flare)))
        test_results.append(flare == test_flare)

    if np.all(test_results):
        print('CONGRATULATIONS: Test passed for classify_flare()!')
    else:
        print('WARNING: classify_flare() failed self-test.')

    print('')
    test_results = []
    for flare,value in zip(flares,values):
        print(('  Testing flare_value() with {0} ({1:.1e} W/m**2) flare...'.format(flare,value)))
        test_value = flare_value(flare)
        print(('    flare_value({0}) = {1:.1e}'.format(test_flare,value)))
        test_results.append(value == test_value)

    if np.all(test_results):
        print('CONGRATULATIONS: Test passed for flare_value()!')
    else:
        print('WARNING: flare_value() failed self-test.')
    print('')

    # Flare finding and plotting test. #############################################
    sTime       = datetime.datetime(2014,1,1)
    eTime       = datetime.datetime(2014,6,30)
    sat_nr      = 15

    goes_data   = read_goes(sTime,eTime,sat_nr)

    output_dir  = 'data/goes'
    try:
        os.makedirs(output_dir)
    except:
        pass

    flares      = find_flares(goes_data)

    with open(os.path.join(output_dir,'flares.txt'),'w') as fl:
        fl.write(flares.to_string())

    for key,flare in flares.iterrows():
        filename = key.strftime('goes_%Y%m%d_%H%M.png')
        filepath = os.path.join(output_dir,filename)

        fig     = plt.figure()
        ax      = fig.add_subplot(111)
        label   = '{0} Class Flare @ {1}'.format(flare['class'],key.strftime('%H%M UT'))
        ax.plot(key,flare['B_AVG'],'o',label=label)

        plot_sTime  = key - datetime.timedelta(hours=12)
        plot_eTime  = key + datetime.timedelta(hours=12)
        goes_plot(goes_data,ax=ax,sTime=plot_sTime,eTime=plot_eTime)

        fig.savefig(filepath,bbox_inches='tight')
        fig.clf()

    flares_str = """
Thank you for testing the goes.py module.  If everything worked, you should find a 
set of plots for all x-class flares from 1Jan2014 - 30Jun2014 in your DAVIT_TMPDIR/goes.
A list of the flares is given below.  This should match the flares.txt file in the same
directory as your plots.

                     B_AVG     class
2014-01-07 18:32:00  0.000125  X1.2
2014-02-25 00:49:00  0.000497  X5.0
2014-03-29 17:48:00  0.000101  X1.0
2014-04-25 00:27:00  0.000139  X1.4
2014-06-10 11:42:00  0.000222  X2.2
2014-06-10 12:52:00  0.000155  X1.5
2014-06-11 09:06:00  0.000100  X1.0
"""

    print(flares_str)
    print(('Your DAVIT_TMPDIR/goes: {0}'.format(output_dir)))
