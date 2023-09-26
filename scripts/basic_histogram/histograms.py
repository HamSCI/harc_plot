#!/usr/bin/env python
"""
Script covering the entire histogram workflow process.
"""
import os
import datetime
import harc_plot

#run_name    = 'Europe'
run_name    = 'World'
data_dir    = os.path.join('data/histograms',run_name)
plot_dir    = os.path.join('output/galleries/histograms',run_name)
params      = ['spot_density']
xkeys       = ['ut_hrs','slt_mid']
sTime       = datetime.datetime(2017,7,1)
eTime       = datetime.datetime(2017,7,2)
region      = run_name
rgc_lim     = (0, 10000)

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
rd['xb_size_min']           = 10.
rd['yb_size_km']            = 250.
rd['reprocess']             = True
rd['output_dir']            = data_dir
rd['band_obj']              = harc_plot.gl.BandData()
harc_plot.calculate_histograms.main(rd)

## Calculate Statistics from Histograms #########################################
#rd = {}
#rd['src_dir']               = data_dir
#rd['params']                = params
#rd['xkeys']                 = xkeys
#rd['stats']                 = ['sum','mean','median','std']
#harc_plot.statistics_histograms.main(rd)
#
## Baseline daily observations against statistics ###############################
#rd = {}
#rd['src_dir']               = data_dir
#rd['xkeys']                 = xkeys
#rd['stats']                 = ['pct_err','z_score']
#harc_plot.baseline_histograms.main(rd)

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
harc_plot.visualize_histograms.main(rd)
harc_plot.visualize_histograms.plot_dailies(rd)

#### Visualize Baselines
#rd['srcs']                  = os.path.join(data_dir,'*.baseline_compare.nc.bz2')
#harc_plot.visualize_histograms.main(rd)
#harc_plot.visualize_histograms.plot_dailies(rd)
#
#### Visualize Statistics
#rd = {}
#rd['srcs']                  = os.path.join(data_dir,'stats.nc.bz2')
#rd['baseout_dir']           = plot_dir
#harc_plot.visualize_histograms_simple.main(rd)
