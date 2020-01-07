import os
import glob
import datetime
import dateutil
from collections import OrderedDict

import numpy as np
import pandas as pd
import xarray as xr

import tqdm

from . import gen_lib as gl

class KeyParamStore(object):
    def __init__(self,xkeys,params,prefix):
        self.xkeys      = xkeys
        self.params     = params
        self.prefix     = prefix
        self.data_das   = self._create_dict(xkeys,params)

    def _create_dict(self,xkeys,params):
        data_das = {}
        for xkey in xkeys:
            data_das[xkey] = {}
            for param in params:
                data_das[xkey][param] = []
        return data_das

    def load_nc(self,nc):
        mbz2    = gl.MyBz2(nc)
        mbz2.uncompress()
        for xkey in self.xkeys:
            group   = '{!s}/{!s}'.format(self.prefix,xkey)
            for param in self.params:
                with xr.open_dataset(mbz2.unc_name,group=group) as fl:
                    ds = fl.load()

                if self.prefix == 'map':
                    da00        = ds[param]
                    da00_attrs  = da00.attrs
        
                    da0         = da00.sum(xkey)
                    da0.attrs   = da00_attrs
                    self.data_das[xkey][param].append(da0)
                elif self.prefix == 'time_series':
                    self.data_das[xkey][param].append(ds[param])

#                if self.prefix == 'time_series':
#                    if self.data_das[xkey][param] is None:
#                        self.data_das[xkey][param] = []
#                    self.data_das[xkey][param].append(ds[param])
#
#                elif self.prefix == 'map':
#                    ds00        = ds[param]
#                    ds00_attrs  = ds00.attrs
#        
#                    ds0         = ds00.sum(xkey)
#                    ds0.attrs   = ds00_attrs
#
#                    if self.data_das[xkey][param] is None:
#                        self.data_das[xkey][param] = ds0
#                    else:
#                        ds1     = self.data_das[xkey][param]
#                        import ipdb; ipdb.set_trace()

        mbz2.remove()
        return self.data_das

    def concat(self,dim='ut_sTime'):
        for xkey in self.xkeys:
            for param in self.params:
                self.data_das[xkey][param] = xr.concat(self.data_das[xkey][param],dim=dim)
        return self.data_das

    def compute_stats(self,stats,dim='ut_sTime'):
        def pstat(param,stat):
            return '_'.join([param,stat])

        param_stats = []
        for param in self.params:
            for stat in stats:
                param_stats.append(pstat(param,stat))

        stats_dss   = self._create_dict(self.xkeys,[])

        for xkey in self.xkeys:
            stats_ds    = xr.Dataset()
            for param in self.params:
                for stat in stats:
                    data_da     = self.data_das[xkey][param]
                    stat_da     = eval("data_da.{!s}(dim='{!s}',keep_attrs=True)".format(stat,dim))
                    stat_da.attrs.update({'stat':stat})
                    stat_da.name    = pstat(param,stat)
                    stats_ds[pstat(param,stat)] = stat_da
            stats_dss[xkey] = stats_ds
        self.stats_dss = stats_dss
        return stats_dss

    def stats_to_nc(self,nc_path):
        for xkey,stats_ds in self.stats_dss.items():
            if os.path.exists(nc_path):
                mode = 'a'
            else:
                mode = 'w'

            group = '{!s}/{!s}'.format(self.prefix,xkey)
            stats_ds.to_netcdf(nc_path,mode=mode,group=group)

def main(run_dct):
    xkeys   = run_dct['xkeys']
    params  = run_dct['params']
    src_dir = run_dct['src_dir']
    stats   = run_dct['stats']

    # Set Up Data Storage Containers
    mps = KeyParamStore(xkeys,['spot_density'],'map')
    kps = KeyParamStore(xkeys,params,'time_series')

    ncs = glob.glob(os.path.join(src_dir,'*.data.nc.bz2'))
    ncs.sort()

    for nc in ncs:
        print(nc)
        mps.load_nc(nc)
        kps.load_nc(nc)

    mps.concat()
    kps.concat()

    mps.compute_stats(['sum'])
    kps.compute_stats(stats)

    stats_nc    = os.path.join(src_dir,'stats.nc')
    if os.path.exists(stats_nc+'.bz2'):
        os.remove(stats_nc+'.bz2')

    mps.stats_to_nc(stats_nc)
    kps.stats_to_nc(stats_nc)

    mbz2    = gl.MyBz2(stats_nc)
    mbz2.compress()

    print('Done computing statistics!!')
