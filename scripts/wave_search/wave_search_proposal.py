#!/usr/bin/env python3
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import harc_plot

#run_name    = 'Europe'
#run_name    = 'World'
#run_name    = 'US'
run_name    = 'Atlantic_Ocean'
data_dir    = os.path.join('data/wave_search',run_name)
plot_dir    = os.path.join('output/galleries/wave_search',run_name)
params      = ['spot_density']
xkeys       = ['ut_hrs']
#sTime       = datetime.datetime(2017,11,1)
#eTime       = datetime.datetime(2017,12,31)
#sTime       = datetime.datetime(2017,11,2)
#eTime       = datetime.datetime(2017,11,5)

sTime       = datetime.datetime(2017,1,1)
eTime       = datetime.datetime(2018,1,1)

region      = run_name
#rgc_lim     = (0.,3000)
rgc_lim     = (0.,10000)

geo_env     = harc_plot.GeospaceEnv()

# Create histogram NetCDF Files ################################################
rd  = {}
rd['sDate']                 = sTime
rd['eDate']                 = eTime
rd['params']                = params
rd['xkeys']                 = xkeys
rd['rgc_lim']               = rgc_lim
rd['filter_region']         = run_name
rd['filter_region_kind']    = 'mids'
rd['xb_size_min']           = 2.
rd['yb_size_km']            = 25.
rd['reprocess']             = True
rd['output_dir']            = data_dir
rd['band_obj']              = harc_plot.gl.BandData()
harc_plot.calculate_histograms.main(rd)

## Calculate Wave Spectra from Histograms #######################################
#rd = {}
#rd['src_dir']               = data_dir
#rd['params']                = params
#rd['xkeys']                 = xkeys
#harc_plot.waves_histograms.main(rd)
#

# Visualization ################################################################
### Visualize Observations
rd = {}
rd['srcs']                  = os.path.join(data_dir,'*.data.nc.bz2')
rd['baseout_dir']           = plot_dir
rd['sTime']                 = sTime
rd['eTime']                 = eTime
rd['plot_region']           = region
rd['geospace_env']          = geo_env
rd['band_keys']             = [28, 21, 14, 7, 3, 1]
#rd['band_keys']             = [14, 7]
#harc_plot.visualize_histograms.main(rd)

harc_plot.visualize_histograms.plot_dailies(rd)

#harc_plot.visualize_waves.main(rd)
import ipdb; ipdb.set_trace()
