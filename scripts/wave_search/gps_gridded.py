#!/usr/bin/env python3

import datetime
import numpy as np
import gps_tec_plot


sDate = datetime.datetime(2017,11,3)
eDate = datetime.datetime(2017,11,3,0,10)
tec_obj = gps_tec_plot.TecPlotter(sDate)
tec_obj.grid_data(dlat=10.,dlon=10.,sDate=sDate,eDate=eDate)
tec_obj.plot_day_gridded()

import ipdb; ipdb.set_trace()
tec_obj.plot_day()
