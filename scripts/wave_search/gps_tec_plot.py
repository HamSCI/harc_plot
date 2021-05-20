#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import datetime
from scipy.interpolate import griddata
from optparse import OptionParser

import cartopy.crs as ccrs
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.feature.nightshade import Nightshade


from harc_plot import gen_lib as gl

# -d 307 -y 2017 -s /Users/ajc/PROJECTS/BSION/NATHANIEL -o /Users/ajc/PROJECTS/BSION/NATHANIEL/PLOTS -w 241 -e 30 -n 1 -a -.2 -b .2 -f 1 -r 0 -g 0 -k 1

# tecs
# 0: Decimal Day
# 1: lat
# 2: lon
# 3: tid
# 4: tec

def set_foregroundcolor(ax, color):
    '''For the specified axes, sets the color of the frame, major ticks,                                                             
         tick labels, axis labels, title and legend                                                                                   
     '''
    for tl in ax.get_xticklines() + ax.get_yticklines():
        tl.set_color(color)
    for spine in ax.spines:
        ax.spines[spine].set_edgecolor(color)
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_color(color)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_color(color)
    ax.axes.xaxis.label.set_color(color)
    ax.axes.yaxis.label.set_color(color)
    ax.axes.xaxis.get_offset_text().set_color(color)
    ax.axes.yaxis.get_offset_text().set_color(color)
    ax.axes.title.set_color(color)
    lh = ax.get_legend()
    if lh != None:
        lh.get_title().set_color(color)
        lh.legendPatch.set_edgecolor('none')
        labels = lh.get_texts()
        for lab in labels:
            lab.set_color(color)
    for tl in ax.get_xticklabels():
        tl.set_color(color)
    for tl in ax.get_yticklabels():
        tl.set_color(color)
 
 
def set_backgroundcolor(ax, color):
    '''Sets the background color of the current axes (and legend).                                                                   
         Use 'None' (with quotes) for transparent. To get transparent                                                                 
         background on saved figures, use:                                                                                            
         pp.savefig("fig1.svg", transparent=True)                                                                                     
    '''
    ax.patch.set_facecolor(color)
    lh = ax.get_legend()
    if lh != None:
        lh.legendPatch.set_facecolor(color)

class TecPlotter(object):
    def __init__(self,date,ww=241,ee=30,nn=1,dt=60.,wgec=False,
            prefix  = 'data/gps_tec_haystack', prefixb = 'output-gpsTec'):
        """
        ww      = 241           # Set window length in 15 second steps - must be odd - default is 241 (or 1 hour)
        ee      = 30            # cutoff elevation, default=30
        nn      = 1             # polynomial order of fit, default=1
        dt      = 60.0
        prefix  = 'data/gps_tec_haystack'        # Data directory
        prefixb = 'output-gpsTec'      # Output directory
        """

        date    = datetime.datetime(date.year,date.month,date.day)
        year    = date.year
        day     = date.utctimetuple().tm_yday

        str1    = "%03d_%d_w%03d_n%02d_e%02d"%(day,year,ww,nn,ee)
        str2    = "w%03d_n%02d_e%02d"%(ww,nn,ee)
        prefixo = "%s/%s"%(prefixb,str2)
        h       = h5py.File("%s/tid_%s.h5"%(prefix,str1),"r")

        if wgec:
            gfo         = file("%s/gec_%s_us.dat"%(prefixb,str1),"w")
            gfo_left    = file("%s/gec_left_%s_us.dat"%(prefixb,str1),"w")
            gfo_right   = file("%s/gec_right_%s_us.dat"%(prefixb,str1),"w")
            gfo_mid     = file("%s/gec_mid_%s_us.dat"%(prefixb,str1),"w")

        ut0     = (datetime.datetime(year,1,1,0,0) - datetime.datetime(1970,1,1,0,0)).total_seconds() + (day-1.0)*24.0*3600
        tecs    = h["tecs"][()]

        # Select the day of interest
        gidx    = np.where((tecs[:,0]>(day-1.0)) & (tecs[:,0] < day))[0]
        tecs    = tecs[gidx,:]
        tod     = (tecs[:,0]-(day-1.0))*24.0*3600.0

        print('min(tecs) = {!s}'.format(np.min(tecs[:,0])))
        print('max(tecs) = {!s}'.format(np.max(tecs[:,0])))

        h.close()

        self.tecs       = tecs
        self.tod        = tod
        self.date       = date
        self.day        = day
        self.ut0        = ut0
        self.ww         = ww
        self.ee         = ee
        self.dt         = dt
        self.wgec       = wgec
        self.prefixb    = prefixb
        self.prefixo    = prefixo

    def date2inx(self,date):
        """
        Convert python datetime to tec vector time index.
        """
        td  = date-self.date
        inx = int(td.total_seconds()/self.dt)
        return inx

    def plot_day(self,sDate=None,eDate=None,tdelta=None,**kwargs):
        """
        Plots entire day of TEC data.
        sDate:  datetime.datetime start time (defaults to self.date)
        eDate:  datetime.datetime end time (defaults to self.date+datetime.datetime(hours=24)
        tdelta: datetime.timedelta (defaults to datetime.timedelta(seconds=self.dt)
        """
        if sDate is None:
            sDate   = self.date
        if eDate is None:
            eDate   = self.date + datetime.timedelta(hours=24)
        if tdelta is None:
            tdelta  = datetime.timedelta(seconds=self.dt)

        dates   = [sDate]
        while dates[-1] < eDate:
            dates.append(dates[-1]+tdelta)

        for plotDate in dates:
            self.plot_tec_fig(plotDate,**kwargs)

    def plot_tec_fig(self,plotDate,fname=None,rr=0,**kwargs):

        fig = plt.figure(figsize=(15,7.5))
        ax  = self.plot_tec_ax(plotDate,fig=fig,rr=rr,**kwargs)

        if fname is None:
            pref        = self.prefixo
            dname       = "%s/%s"%(pref,plotDate.strftime('%Y-%m-%dT%H-00-00'))
            utcstr      = plotDate.strftime('UTC %Y-%m-%d %H:%M:%S')
            date_str    = plotDate.strftime('%Y%m%d.%H%M.%SUT')
            print(utcstr)
            os.system("mkdir -p %s"%(dname))
            if rr==1:
                fname = "{!s}/tec_n3plot_{!s}.png".format(dname,date_str)
            if rr==0:
                fname = "{!s}/tid_n3plot_{!s}.png".format(dname,date_str)

        fig.savefig(fname,bbox_inches='tight')
        print('   --> {!s}'.format(fname))
        plt.close(fig)

    def plot_tec_ax(self,plotDate,fig=None,ax=None,vx=-0.2,vy=0.2,rr=0,zz=10,
            projection=ccrs.PlateCarree(),maplim_region='World',wgec=False):
        """
        vx      = -0.2          # if f=1, plotting color scale min, default=5
        vy      =  0.2          # if f=1, plotting color scale max, default=50
        figs    = True          # generate figures (0=no 1= yes) default=0
        rr      = 0             # if f=1, plot TEC (=1) or TID (=0), default=1
        wgec    = False         # generate a data output file for global and US regional mean TEC (0=no 1=yes), default=0
        zz      = 10.0          # if f=1, point size of the scatter plot, default=10
        """

        inx                 = self.date2inx(plotDate)

        tecs                = self.tecs
        day                 = self.day
        ut0                 = self.ut0
        dt                  = self.dt
        tod                 = self.tod

        pidx                = np.where( (tod  > dt*inx) & (tod < (dt*inx+dt)) )[0]           # Time of Day Index
        pidx0               = np.where( (tod  > dt*inx-dt/2) & (tod < (dt*inx+dt/2)) )[0]    # Time of Day Index +/- dt/2
        pidus               = np.where( (tod  > dt*inx-dt/2) & (tod < (dt*inx+dt/2)) & (abs(tecs[:,2]+ 95)<25)  & (abs(tecs[:,1]-40)<10))[0]
        pidus_left          = np.where( (tod  > dt*inx-dt/2) & (tod < (dt*inx+dt/2)) & (abs(tecs[:,2]+ 80)<2.5) & (abs(tecs[:,1]-40)<10))[0]
        pidus_mid           = np.where( (tod  > dt*inx-dt/2) & (tod < (dt*inx+dt/2)) & (abs(tecs[:,2]+ 95)<2.5) & (abs(tecs[:,1]-40)<10))[0]
        pidus_right         = np.where( (tod  > dt*inx-dt/2) & (tod < (dt*inx+dt/2)) & (abs(tecs[:,2]+110)<2.5) & (abs(tecs[:,1]-40)<10))[0] 
        gec                 = np.median(tecs[pidx0,4])
        gecus               = np.median(tecs[pidus,4])
        gtidus              = np.median(tecs[pidus,3])
        gecus_std           = np.std(tecs[pidus,4])
        gtidus_std          = np.std(tecs[pidus,3])
        gecus_mid           = np.median(tecs[pidus_mid,4])
        gtidus_mid          = np.median(tecs[pidus_mid,3])
        gecus_std_mid       = np.std(tecs[pidus_mid,4])
        gtidus_std_mid      = np.std(tecs[pidus_mid,3])
        gecus_left          = np.median(tecs[pidus_left,4])
        gtidus_left         = np.median(tecs[pidus_left,3])
        gecus_std_left      = np.std(tecs[pidus_left,4])
        gtidus_std_left     = np.std(tecs[pidus_left,3])
        gecus_right         = np.median(tecs[pidus_right,4])
        gtidus_right        = np.median(tecs[pidus_right,3])
        gecus_std_right     = np.std(tecs[pidus_right,4])
        gtidus_std_right    = np.std(tecs[pidus_right,3]) 

        if wgec:
            gfo.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus,gtidus,gtidus_std,len(pidx0),len(pidus)))
            gfo_left.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus_left,gtidus_left,gtidus_std_left,len(pidx0),len(pidus_left)))
            gfo_mid.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus_mid,gtidus_mid,gtidus_std_mid,len(pidx0),len(pidus_mid)))
            gfo_right.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus_right,gtidus_right,gtidus_std_right,len(pidx0),len(pidus_right)))


        if fig is None:
            fig = plt.figure(figsize=(15,7.5))

        if ax is None:
            ax  = fig.add_subplot(1,1,1, projection=projection)

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

        ax.coastlines()
        ax.gridlines(draw_labels=True)

        x, y = tecs[pidx,2],tecs[pidx,1]
        if rr==1: # TEC
           mpbl = ax.scatter(x,y,c=tecs[pidx,4],edgecolors='none',vmin=vx,vmax=vy,s=zz,marker='s')
        if rr==0: # TID
           mpbl = ax.scatter(x,y,c=tecs[pidx,3],edgecolors='none',vmin=vx,vmax=vy,s=zz)

        ax.add_feature(Nightshade(plotDate, alpha=0.2))
        ax.set_title(plotDate.strftime('UTC %Y-%m-%d %H:%M:%S'))

        cbar = plt.colorbar(mpbl,shrink=0.8,pad=0.075)
        cbar.set_label('$\Delta$TECu',size='medium')
        cbar.ax.tick_params(labelsize='medium')


if __name__ == '__main__':
    sDate = datetime.datetime(2017,11,3)
    eDate = datetime.datetime(2017,11,4)
    tec_obj = TecPlotter(sDate)
    tec_obj.plot_day()
