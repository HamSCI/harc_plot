#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import harc_plot


#data_sources: list, i.e. [1,2]
#    0: dxcluster
#    1: WSPRNet
#    2: RBN

dsd = data_src_dict = {}
dsd['WSPRNet']      = {'data_sources':[1]}
dsd['RBN']          = {'data_sources':[2]}
dsd['WSPRNet_RBN']  = {'data_sources':[1,2]}


def main(data_source_name):
    region          = 'World'
    run_name        = '-'.join([region,data_source_name])
    data_dir        = os.path.join('data/solarcycle_3hr_250km/histograms',run_name)
    plot_dir        = os.path.join('output/galleries/solarcycle_3hr_250km',run_name)
    params          = ['spot_density']
    xkeys           = ['ut_hrs','slt_mid']
    sTime           = datetime.datetime(2009,1,1)
    eTime           = datetime.datetime(2013,1,1)
    rgc_lim         = (0, 10000)

    #geo_env     = harc_plot.GeospaceEnv()
    geo_env = None

    # Create histogram NetCDF Files ################################################
    rd  = {}
    rd['sDate']                 = sTime
    rd['eDate']                 = eTime
    rd['params']                = params
    rd['xkeys']                 = xkeys
    rd['rgc_lim']               = rgc_lim
    rd['filter_region']         = region
    rd['filter_region_kind']    = 'mids'
    rd['xb_size_min']           = 3*60.
    rd['yb_size_km']            = 250.
    rd['loc_sources']           = None
    rd['data_sources']          = dsd[data_source_name]['data_sources']
    rd['reprocess']             = False
    rd['output_dir']            = data_dir
    rd['band_obj']              = harc_plot.gl.BandData()
    harc_plot.calculate_histograms.main(rd)

if __name__ == '__main__':
    dsns = data_src_names = []
    dsns.append('WSPRNet')
    dsns.append('RBN')
    dsns.append('WSPRNet_RBN')

    for dsn in data_src_names:
        print('Calculating histograms for: {!s}'.format(dsn))
        main(dsn)
