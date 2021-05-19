#!/usr/bin/env python3
import matplotlib as mpl
mpl.use('agg')
from matplotlib import pyplot as plt
import numpy as np
import datetime

from pyhwm2014 import HWM142D, HWM142DPlot

prms    = {}

prm = 'Uwind'
tmp = {}
tmp['label']    = r'Zonal (U), m/s'
tmp['vmin']     = -250.
tmp['vmax']     =  250.
prms[prm]       = tmp

prm = 'Vwind'
tmp = {}
tmp['label']    = r'Meridional (U), m/s'
tmp['vmin']     = -100.
tmp['vmax']     =  100.
prms[prm]       = tmp

def hwm_plot2D_ax(prm, hwmObj, ax, cmap = None, title = None,
        cbar_label=None, **kwargs ):

        prmd    = prms[prm]

        xlim    = kwargs.get('xlim')
        ylim    = kwargs.get('ylim')
        xlabel  = kwargs.get('xlabel',r'Geog. Lat. ($^o$)')
        ylabel  = kwargs.get('ylabel',r'Altitude (km)')
        cmap    = kwargs.get('cmap', plt.cm.RdBu_r)

        cbar_label  = kwargs.get('cbar_label',prmd.get('label'))

        vmin    = kwargs.get('vmin',prmd.get('vmin',-250.))
        vmax    = kwargs.get('vmax',prmd.get('vmax', 250.))

        xVal    = hwmObj.glatbins
        yVal    = hwmObj.altbins

        X, Y    = np.meshgrid( xVal, yVal )
        X       = X.T
        Y       = Y.T

        zVal    = getattr(hwmObj,prm)
        C       = zVal.T

        ipn     = ax.pcolor( X, Y, C[:-1,:-1], cmap=cmap, edgecolors='None', 
                norm=mpl.colors.Normalize( vmax=vmax, vmin=vmin ))
        ax.set_xlim( xlim )
        ax.set_ylim( ylim )
        ax.set_title( title )
        ax.set_xlabel( xlabel )
        ax.set_ylabel( ylabel )

        date = datetime.datetime(hwmObj.year,1,1)       \
             + datetime.timedelta(days=hwmObj.doy-1)    \
             + datetime.timedelta(hours=hwmObj.ut)

        date_str    = date.strftime('%Y %b %d %H%M UT')

        title = 'HWM14 {!s} GLON: {:0.1f}'.format(date_str,hwmObj.glon)+'$^{\circ}$'
        ax.set_title(title)

        cbpn = plt.colorbar( ipn,ax=ax )
        cbpn.set_label( cbar_label )

def plot_winds(rd):
    dates   = rd.get('dates',[datetime.datetime.now()])
    altlim  = rd.get('altlim', [90., 200.])
    altstp  = rd.get('altstp',2.)
    glatlim = rd.get('glatlim',[0., 90])
    glatstp = rd.get('glatstp',2.)
    glon    = rd.get('glon', -105.)

    nrows   = len(dates)

    fig = plt.figure(figsize=(15,nrows*5.))

    for row_inx,date in enumerate(dates):
        year    = date.year
        doy     = date.timetuple().tm_yday
        ut      = date.hour + date.minute/60.

        hwmObj = HWM142D(
            year    = year,
            day     = doy,
            ut      = ut,
            glon    = glon,
            glatlim = glatlim,
            glatstp = glatstp,
            altlim  = altlim,
            altstp  = altstp,
            ap      = [-1, 35],
            f107    = -1,
            f107a   = -1,
            option  =  2,
            stl     = -1,
            verbose = False
        )

        ax_inx = nrows*row_inx + 1
        ax  = fig.add_subplot(nrows,2,ax_inx)
        prm = 'Uwind'
        hwm_plot2D_ax(prm, hwmObj, ax)

        ax  = fig.add_subplot(nrows,2,ax_inx+1)
        prm = 'Vwind'
        hwm_plot2D_ax(prm, hwmObj, ax)

    fig.tight_layout()
    fig.savefig('output_1.png',bbox_inches='tight')

if __name__ == '__main__':
    dates   = []
    dates.append(datetime.datetime(2017,11, 3,10))
    dates.append(datetime.datetime(2017, 5,16,10))
    rd = {}
    rd['dates'] = dates

    plot_winds(rd)
