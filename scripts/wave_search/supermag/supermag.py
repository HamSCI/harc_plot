#!/usr/bin/env python3
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from harc_plot import gen_lib as gl

import supermag_api as sm

#mpl.rcParams['font.size']      = 18
#mpl.rcParams['font.weight']    = 'bold'
#mpl.rcParams['figure.figsize'] = np.array([15, 8])
mpl.rcParams['axes.grid']      = True
mpl.rcParams['grid.linestyle'] = ':'
mpl.rcParams['axes.xmargin']   = 0

sDate = datetime.datetime(2017,11,3)
eDate = datetime.datetime(2017,11,4)

logon   = 'w2naf'
start   = sDate.strftime('%Y-%m-%dT%H%M')
extent  = (eDate-sDate).total_seconds()
flags   = ['SME','SML','SMU','smer']

sm_result       = (sm.SuperMAGGetIndices(logon,start,extent,','.join(flags)))
sm_data		= sm_result[1]
sm_data['tval'] = pd.to_datetime(sm_data['tval'],unit='s')
sm_data         = sm_data.set_index('tval')

SMEr	= np.array(sm_data['SMEr'].to_list())

# Plotting #####################################################################
output      = 'output'
date_str    = sDate.strftime('%Y%m%d.%H%M-') + eDate.strftime('%Y%m%d.%H%M')
fname       = 'supermag-{!s}.png'.format(date_str)
fpath       = os.path.join(output,fname)
gl.prep_output({0:output})

axs_adjust = []

fig = plt.figure(figsize=(15,8))
ax  = fig.add_subplot(2,1,1)
axs_adjust.append(ax)

keys	= ['SME','SML','SMU']
for key in keys:
    ax.plot(sm_data.index,sm_data[key],label=key)

ax.legend(loc='upper right')

ax.set_xlabel('Time [UT]')
ax.set_ylabel('nT')

ax  	= fig.add_subplot(2,1,2)
ax_set	= ax
xx  	= sm_data.index.to_list()
xx  	= xx + [xx[-1] + datetime.timedelta(seconds=1)]
yy  	= np.arange(25)
cc  	= SMEr.T
pcl 	= ax.pcolormesh(xx,yy,cc,cmap='viridis')
cbar	= plt.colorbar(pcl,label='SMEr [nT]',pad=0.01,aspect=10)

ax.set_yticks(np.arange(0,25,4))

ax.set_xlabel('Time [UT]')
ax.set_ylabel('MLT')

fig.tight_layout()

for ax in axs_adjust:
    gl.adjust_axes(ax,ax_set)

print('Plotting: {!s}'.format(fpath))
fig.savefig(fpath,bbox_inches='tight')

import ipdb; ipdb.set_trace()
