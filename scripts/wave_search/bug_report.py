#!/usr/bin/env python3
from matplotlib import pyplot as plt
import numpy as np
import datetime

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import ipdb; ipdb.set_trace()

fig = plt.figure()
ax  = fig.add_subplot(111)

times   = [datetime.datetime(2021,1,1)]
while len(times) < 721:
    times.append(times[-1]+datetime.timedelta(seconds=120))

y_vals  = np.arange(101)

time_axis, y_axis   = np.meshgrid(times,y_vals)
shape               = (len(y_vals)-1,len(times)-1)
z_data              = np.arange(shape[0]*shape[1])
z_data.shape        = shape

im = ax.pcolormesh(time_axis, y_axis, z_data)
fig.savefig('bug_report.png')
