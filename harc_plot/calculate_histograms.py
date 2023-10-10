import os
import datetime
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import json

import tqdm

from .timeutils import daterange, strip_time
from . import gen_lib as gl

def calc_histogram(frame,attrs):
    xkey    = attrs['xkey']
    xlim    = attrs['xlim']
    dx      = attrs['dx']
    ykey    = attrs['ykey']
    ylim    = attrs['ylim']
    dy      = attrs['dy']

    xbins   = gl.get_bins(xlim,dx)[:-1]
    ybins   = gl.get_bins(ylim,dy)

    if len(frame) > 2:
       hist, xb, yb = np.histogram2d(frame[xkey], frame[ykey], bins=[xbins, ybins])
    else:
        xb      = xbins
        yb      = ybins
        hist    = np.zeros((len(xb)-1,len(yb)-1))

    crds    = {}
    crds['ut_sTime']    = attrs['sTime']
    crds['freq_MHz']    = attrs['band']
    crds[xkey]          = xb[:-1]
    crds[ykey]          = yb[:-1]
    
    attrs   = attrs.copy()
    for key,val in attrs.items():
        attrs[key] = str(val)
    da = xr.DataArray(hist,crds,attrs=attrs,dims=[xkey,ykey])
    return da 

def main(run_dct):
    """
    data_sources: list, i.e. [1,2]
        0: dxcluster
        1: WSPRNet
        2: RBN

    loc_sources: list, i.e. ['P','Q']
        P: user Provided
        Q: QRZ.com or HAMCALL
        E: Estimated using prefix
    """

    # Get Variables from run_dct
    sDate               = run_dct['sDate']
    eDate               = run_dct['eDate']
    params              = run_dct['params']
    xkeys               = run_dct['xkeys']
    rgc_lim             = run_dct['rgc_lim']
    filter_region       = run_dct['filter_region']
    filter_region_kind  = run_dct['filter_region_kind']
    band_obj            = run_dct['band_obj']
    xb_size_min         = run_dct['xb_size_min']
    yb_size_km          = run_dct['yb_size_km']
    base_dir            = run_dct.get('base_dir')
    output_dir          = run_dct.get('output_dir')
    reprocess           = run_dct.get('reprocess',False)
    loc_sources         = run_dct.get('loc_sources')
    data_sources        = run_dct.get('data_sources',[1,2])

    # Define path for saving NetCDF Files
    if output_dir is None:
        tmp = []
        if filter_region is not None:
            tmp.append('{!s}'.format(filter_region))
            tmp.append('{!s}'.format(filter_region_kind))
        tmp.append('{:.0f}-{:.0f}km'.format(rgc_lim[0],rgc_lim[1]))
        tmp.append('dx{:.0f}min'.format(xb_size_min))
        tmp.append('dy{:.0f}km'.format(yb_size_km))
        ncs_path = os.path.join(base_dir,'_'.join(tmp))
    else:
        ncs_path = output_dir

    gl.prep_output({0:ncs_path},clear=reprocess)

    # Loop through dates
    dates       = list(daterange(sDate, eDate))
    if strip_time(sDate) != strip_time(eDate):
        dates   = dates[:-1]

    for dt in tqdm.tqdm(dates,dynamic_ncols=True):
        nc_name = dt.strftime('%Y%m%d') + '.data.nc'
        nc_path = os.path.join(ncs_path,nc_name)

        # Skip processing if file already exists
        if os.path.exists(nc_path+'.bz2'):
            continue

        # Load spots from CSVs
        load_dates = [dt,dt+pd.Timedelta('1D')]

        df  = pd.DataFrame()
        for ld_inx,load_date in enumerate(load_dates):
            ld_str  = load_date.strftime("%Y-%m-%d") 
            dft = gl.load_spots_csv(ld_str,data_sources=data_sources,
                            rgc_lim=rgc_lim,loc_sources=loc_sources,
                            filter_region=filter_region,filter_region_kind=filter_region_kind)

            if dft is None:
                tqdm.tqdm.write('No data for {!s}'.format(ld_str))
                continue
            for xkey in xkeys:
                dft[xkey]    = ld_inx*24 + dft[xkey]

            df  = pd.concat([df,dft],ignore_index=True)

        # Set Up Data Storage Containers
        data_das = {}
        for xkey in xkeys:
            data_das[xkey] = {}
            for param in params:
                data_das[xkey][param] = []

        map_das = {}
        for xkey in xkeys:
            map_das[xkey] = [] 

        src_cnts    = {}
        for xkey in xkeys:
            if xkey not in src_cnts:
                src_cnts[xkey] = {}
            for band_inx, (band_key,band) in enumerate(band_obj.band_dict.items()):
                if band_key not in src_cnts[xkey]:
                    src_cnts[xkey][band_key]    = {}

        for band_inx, (band_key,band) in enumerate(band_obj.band_dict.items()):
            if len(df) == 0:
                continue

            frame   = df.loc[df["band"] == band.get('meters')].copy()

            # Create attrs diction to save with xarray DataArray
            attrs  = OrderedDict()
            attrs['sTime']              = dt
            attrs['param']              = param
            attrs['filter_region']      = filter_region
            attrs['filter_region_kind'] = filter_region_kind
            attrs['band']               = band_key
            attrs['band_name']          = band['name']
            attrs['band_fname']         = band['freq_name']

            for xkey in xkeys:
                src_cnts[xkey][band_key]    = gl.count_sources(frame)
                for param in params:
                    # Compute General Data
                    attrs['xkey']               = xkey
                    attrs['param']              = param
                    attrs['dx']                 = xb_size_min/60.
                    attrs['xlim']               = (0,24)
                    attrs['ykey']               = 'dist_Km'
                    attrs['ylim']               = rgc_lim
                    attrs['dy']                 = yb_size_km
                    result                      = calc_histogram(frame,attrs)
                    data_da_result              = result
                    data_das[xkey][param].append(result)

                # Compute Map
                time_bins                   = np.array(data_da_result.coords[xkey])
                map_attrs                   = attrs.copy()
                del map_attrs['param']
                map_attrs['xkey']           = 'md_long'
                map_attrs['xlim']           = (-180,180)
                map_attrs['dx']             = 1
                map_attrs['ykey']           = 'md_lat'
                map_attrs['ylim']           = (-90,90)
                map_attrs['dy']             = 1

                map_tmp = []
                for tb_inx,tb_0 in enumerate(time_bins):
                    tb_1                = time_bins[tb_inx] + xb_size_min/60.
                    tf                  = np.logical_and(frame[xkey] >= tb_0, frame[xkey] < tb_1)
                    tb_frame            = frame[tf].copy()
                    result              = calc_histogram(tb_frame,map_attrs)
                    result.coords[xkey] = tb_0
                    map_tmp.append(result)
                map_tmp_da      = xr.concat(map_tmp,dim=xkey)
                map_das[xkey].append(map_tmp_da)

        # Continue if no data.
        if len(map_das[xkeys[0]]) == 0:
            continue

        # Maps - Concatenate all bands into single DataArray
        for xkey in xkeys:
            map_das[xkey]   = xr.concat(map_das[xkey],dim='freq_MHz')

        map_dss = OrderedDict()
        for xkey in xkeys:
            map_ds                  = xr.Dataset()
            map_ds['spot_density']  = map_das[xkey]
            map_dss[xkey]           = map_ds

        # Time Series - Concatenate all bands into single DataArray
        for xkey in xkeys:
            for param in params:
                data_da = data_das[xkey][param] = xr.concat(data_das[xkey][param],dim='freq_MHz')

        data_dss    = OrderedDict()
        for xkey in xkeys:
            data_ds = xr.Dataset()
            for param in params:
                d_ds                    = data_das[xkey][param] 
                src_cnt                 = src_cnts[xkey]
                d_ds.attrs['src_cnt']   = pd.DataFrame(src_cnt).to_json()
                data_ds[param]          = d_ds 
            data_dss[xkey] = data_ds


        # Save to data file.
        first   = True
        for xkey,map_ds in map_dss.items():
            if first:
                mode    = 'w'
                first   = False
            else:
                mode    = 'a'

            group = 'map/{!s}'.format(xkey)
            map_ds.to_netcdf(nc_path,mode=mode,group=group)

        for xkey,data_ds in data_dss.items():
            group = 'time_series/{!s}'.format(xkey)
            data_ds.to_netcdf(nc_path,mode='a',group=group)
        
        mbz2    = gl.MyBz2(nc_path)
        mbz2.compress()
    return ncs_path
