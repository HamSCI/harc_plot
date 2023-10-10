#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import dateutil.parser
import argparse
import harc_plot


#data_sources: list, i.e. [1,2]
#    0: dxcluster
#    1: WSPRNet
#    2: RBN

dsd = data_src_dict = {}
dsd['WSPRNet']      = {'data_sources':[1]}
dsd['RBN']          = {'data_sources':[2]}
dsd['WSPRNet_RBN']  = {'data_sources':[1,2]}


def main(start='2015-01-01',stop='2015-02-01',
        data_source='WSPRNet_RBN',test_mode=False):

    region          = 'World'
    rgc_lim         = (0, 10000)
    xkeys           = ['slt_mid']
    params          = ['spot_density']

    sTime           = dateutil.parser.isoparse(start)
    eTime           = dateutil.parser.isoparse(stop)

    if test_mode:
#        sTime       = datetime.datetime(2015,1,1)
#        eTime       = datetime.datetime(2015,2,1)
        run_name    = '-'.join([region,data_source,'test'])
    else:
#        sTime       = datetime.datetime(2009,1,1)
#        eTime       = datetime.datetime(2020,1,1)
        run_name    = '-'.join([region,data_source])

    data_dir        = os.path.join('data/solarcycle_3hr_250km/histograms',run_name)
    plot_dir        = os.path.join('output/galleries/solarcycle_3hr_250km',run_name)

    #geo_env        = harc_plot.GeospaceEnv()
    geo_env         = None

    # Create histogram NetCDF Files ################################################
    rd  = {}
    rd['sDate']                 = sTime
    rd['eDate']                 = eTime
    rd['params']                = params
    rd['xkeys']                 = xkeys
    rd['rgc_lim']               = rgc_lim
    rd['filter_region']         = region
    rd['filter_region_kind']    = 'mids'
    rd['xb_size_min']           = 6*60.
    rd['yb_size_km']            = 250.
    rd['loc_sources']           = None
    rd['data_sources']          = dsd[data_source]['data_sources']
    rd['reprocess']             = True
    rd['output_dir']            = data_dir
    rd['band_obj']              = harc_plot.gl.BandData()
    harc_plot.calculate_histograms.main(rd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_sources',default=['WSPRNet','RBN','WSPRNet_RBN'],help='[WSPRNet, RBN, WSPRNet_RBN]')
    parser.add_argument('--start',default='2015-01-01')
    parser.add_argument('--stop', default='2015-02-01')
    args = parser.parse_args()

    for data_source in args.data_sources:
        rd  = {}
        rd['data_source']   = data_source
        rd['start']         = args.start
        rd['stop']          = args.stop
        rd['test_mode']     = True
        print(rd)
        main(**rd)
