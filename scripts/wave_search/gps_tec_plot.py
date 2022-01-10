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

import tqdm

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
#    def __init__(self,date,ww=121,ee=15,nn=1,dt=60.,wgec=False,
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

    def grid_data(self,dlat=1.,dlon=1.,sDate=None,eDate=None,tdelta=None):
        """
        Grid GPS data onto a regular grid.
        """
        dates   = self.get_dates(sDate=sDate,eDate=eDate,tdelta=tdelta)
        lats    = np.arange(-90.,90.,dlat)
        lons    = np.arange(-180.,180.,dlon)

        mean_dtecs  = np.zeros((lats.size,lons.size,len(dates)))
        std_dtecs   = np.zeros((lats.size,lons.size,len(dates)))
        n_dtecs     = np.zeros((lats.size,lons.size,len(dates)))


        tecs    = self.tecs
        day     = self.day
        ut0     = self.ut0
        dt      = self.dt
        tod     = self.tod

        for date_inx,date in tqdm.tqdm(enumerate(dates),dynamic_ncols=True,total=len(dates)):
            inx                 = self.date2inx(date)

            pidx                = np.where( (tod  > dt*inx) & (tod < (dt*inx+dt)) )[0]           # Time of Day Index

            tec_lats    = tecs[pidx,1]
            tec_lons    = tecs[pidx,2]
            dtecs       = tecs[pidx,3]

            for lat_inx,lat in enumerate(lats):
                tf  = np.logical_and(tec_lats >= lat, tec_lats < lat + dlat)
                tec_lats_1  = tec_lats[tf]
                tec_lons_1  = tec_lons[tf]
                dtecs_1     = dtecs[tf]

                for lon_inx,lon in enumerate(lons):
                    tf  = np.logical_and(tec_lons_1 >= lon, tec_lons_1 < lon + dlon)
                    tec_lats_2  = tec_lats_1[tf]
                    tec_lons_2  = tec_lons_1[tf]
                    dtecs_2     = dtecs_1[tf]

                    if len(dtecs_2) == 0:
                        continue
                    mean_dtecs[lat_inx,lon_inx,date_inx]    = np.nanmean(dtecs_2)
                    std_dtecs[lat_inx,lon_inx,date_inx]     = np.nanstd(dtecs_2)
                    n_dtecs[lat_inx,lon_inx,date_inx]       = len(dtecs_2)

        self.gridded_dates  = dates
        self.gridded_lats   = lats
        self.gridded_lons   = lons
        self.mean_dtecs     = mean_dtecs
        self.std_dtecs      = std_dtecs
        self.n_dtecs        = n_dtecs

    def plot_keogram(self,ax,lat_0,lat_1,lon_0,lon_1,sDate=None,eDate=None,tdelta=None,keotype='lat',
           map_ax=None,vx=-0.2,vy=0.2,rr=0,zz=10,cmap='jet'):
        """
        Plot a keogram of GNSS TEC Data
        """
        dates   = self.get_dates(sDate=sDate,eDate=eDate,tdelta=tdelta)

        tecs    = self.tecs
        day     = self.day
        ut0     = self.ut0
        dt      = self.dt
        tod     = self.tod

#        keo_xx      = []
#        keo_yy      = np.array([])
#        keo_dtecs   = np.array([])
#        keo_lats    = np.array([])
#        keo_lons    = np.array([])
        for date_inx,date in tqdm.tqdm(enumerate(dates),dynamic_ncols=True,total=len(dates),desc='{!s} GNSS Keogram'.format(keotype)):
            inx                 = self.date2inx(date)

            pidx                = np.where( (tod  > dt*inx) & (tod < (dt*inx+dt)) )[0]           # Time of Day Index
            tec_lats    = tecs[pidx,1]
            tec_lons    = tecs[pidx,2]
            dtecs       = tecs[pidx,3]

            # Filter out unneeded lats/lons.
            tf          = np.logical_and(tec_lats >= lat_0, tec_lats < lat_1)
            tec_lats    = tec_lats[tf]
            tec_lons    = tec_lons[tf]
            dtecs       = dtecs[tf]

            tf          = np.logical_and(tec_lons >= lon_0, tec_lons < lon_1)
            tec_lats    = tec_lats[tf]
            tec_lons    = tec_lons[tf]
            dtecs       = dtecs[tf]

            xx          = [date]*len(dtecs)
            if keotype == 'lat':
                yy          = tec_lats
            elif keotype == 'lon':
                yy          = tec_lons

            mpbl        = ax.scatter(xx,yy,c=dtecs,vmin=vx,vmax=vy,cmap=cmap,edgecolors='none',rasterized=True)

            if map_ax is not None:
                map_ax.scatter(tec_lons,tec_lats,c=dtecs,vmin=vx,vmax=vy,cmap=cmap,edgecolors='none',rasterized=True)

#            keo_xx      = keo_xx + xx
#            keo_yy      = np.concatenate((keo_yy,yy))
#            keo_dtecs   = np.concatenate((keo_dtecs,dtecs))
#            keo_lats    = np.concatenate((keo_lats,tec_lats))
#            keo_lons    = np.concatenate((keo_lons,tec_lons))
#
#        mpbl        = ax.scatter(keo_xx,keo_yy,c=keo_dtecs,vmin=vx,vmax=vy,cmap=cmap,edgecolors='none',rasterized=True)
#
#        if map_ax is not None:
#            map_ax.scatter(keo_lons,keo_lats,c=keo_dtecs,vmin=vx,vmax=vy,cmap=cmap,edgecolors='none',rasterized=True)

        sDate_str = sDate.strftime('%d %b %Y')
        if keotype == 'lat':
            ax.set_ylabel('Latitude [deg]')
            ax.set_ylim(lat_0,lat_1)
            title = ' - '.join([sDate_str,'Lon Lim: {:.0f} to {:.0f}'.format(lon_0,lon_1)])
            ax.set_title(title)
        elif keotype == 'lon':
            ax.set_ylabel('Longitude [deg]')
            ax.set_ylim(lon_0,lon_1)
            title = ' - '.join([sDate_str,'Lat Lim: {:.0f} to {:.0f}'.format(lat_0,lat_1)])
            ax.set_title(title)

        ax.set_xlim(sDate,eDate)
        ax.set_xlabel('Time [UT]')

        cbar = plt.colorbar(mpbl,shrink=0.8,pad=0.075,cax=None,extend='both')
        cbar.set_label('$\Delta$TECu')#,size='medium')
        return {'ax':ax,'cbar':cbar,'title':title}

    def date2inx(self,date):
        """
        Convert python datetime to tec vector time index.
        """
        td  = date-self.date
        inx = int(td.total_seconds()/self.dt)
        return inx

    def get_dates(self,sDate=None,eDate=None,tdelta=None):
        """
        Get available dates of TEC data within a range.

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

        return dates

    def plot_day(self,sDate=None,eDate=None,tdelta=None,**kwargs):
        """
        Plots entire day of TEC data.
        sDate:  datetime.datetime start time (defaults to self.date)
        eDate:  datetime.datetime end time (defaults to self.date+datetime.datetime(hours=24)
        tdelta: datetime.timedelta (defaults to datetime.timedelta(seconds=self.dt)
        """

        dates   = self.get_dates(sDate=sDate,eDate=eDate,tdelta=tdelta)
        for plotDate in dates:
            self.plot_tec_fig(plotDate,**kwargs)


    def plot_day_gridded(self,param='mean_dtecs'):
        dates   = self.gridded_dates
        for plotDate in dates:
            self.plot_tec_fig_gridded(plotDate,param)

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

    def plot_tec_fig_gridded(self,plotDate,param='mean_dtecs',fname=None,**kwargs):
        fig = plt.figure(figsize=(15,7.5))
        ax  = self.plot_tec_gridded_ax(plotDate,fig=fig,param=param,**kwargs)

        if fname is None:
            pref        = self.prefixo
            dname       = "%s/%s"%(pref,plotDate.strftime('%Y-%m-%dT%H-00-00'))
            utcstr      = plotDate.strftime('UTC %Y-%m-%d %H:%M:%S')
            date_str    = plotDate.strftime('%Y%m%d.%H%M.%SUT')
            print(utcstr)
            os.system("mkdir -p %s"%(dname))

            fname = "{!s}/{!s}_{!s}.png".format(dname,param,date_str)

        fig.savefig(fname,bbox_inches='tight')
        print('   --> {!s}'.format(fname))
        plt.close(fig)

    def plot_tec_gridded_ax(self,plotDate,fig=None,ax=None,param='mean_dtecs',vx=-0.2,vy=0.2,cmap='jet',
            projection=ccrs.PlateCarree(),maplim_region='World',cax=None,
            title=None,grid=True,
            top_labels=True,bottom_labels=True,right_labels=True,left_labels=True):
        """
        vx      = -0.2          # if f=1, plotting color scale min, default=5
        vy      =  0.2          # if f=1, plotting color scale max, default=50
        figs    = True          # generate figures (0=no 1= yes) default=0
        """

        if fig is None:
            fig = ax.get_figure()

        if ax is None:
            ax  = fig.add_subplot(1,1,1, projection=projection)
        plt.sca(ax)

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

        ax.coastlines()
        if grid is True:
            grd = ax.gridlines()
            grd.bottom_labels   = bottom_labels
            grd.top_labels      = top_labels
            grd.left_labels     = left_labels
            grd.right_labels    = right_labels

        ax.add_feature(Nightshade(plotDate, alpha=0.2))

        lats    = self.gridded_lats
        lons    = self.gridded_lons

        dinx    = np.where(np.array(self.gridded_dates) == plotDate)[0][0]
        vals    = getattr(self,param)[:,:,dinx]

        mpbl    = ax.pcolormesh(lons,lats,vals,vmin=vx,vmax=vy,cmap=cmap)

        if title is None:
            ax.set_title(plotDate.strftime('UTC %Y-%m-%d %H:%M:%S'))
        elif title is not False:
            ax.set_title(title)

        cbar = plt.colorbar(mpbl,shrink=0.8,pad=0.075,cax=cax,extend='both')
#        cbar.set_label('$\Delta$TECu')#,size='medium')
        cbar.set_label(param)#,size='medium')

        return {'ax':ax,'cbar':cbar}

    def plot_tec_ax(self,plotDate,fig=None,ax=None,vx=-0.2,vy=0.2,rr=0,zz=10,cmap='jet',
            projection=ccrs.PlateCarree(),maplim_region='World',cax=None,wgec=False,
            title=None,grid=True,
            top_labels=True,bottom_labels=True,right_labels=True,left_labels=True):
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
            fig = ax.get_figure()

        if ax is None:
            ax  = fig.add_subplot(1,1,1, projection=projection)
        plt.sca(ax)

        ax.set_xlim(gl.regions[maplim_region]['lon_lim'])
        ax.set_ylim(gl.regions[maplim_region]['lat_lim'])

        ax.coastlines()
        if grid is True:
            grd = ax.gridlines()
            grd.bottom_labels   = bottom_labels
            grd.top_labels      = top_labels
            grd.left_labels     = left_labels
            grd.right_labels    = right_labels

        x, y = tecs[pidx,2],tecs[pidx,1]
        if rr==1: # TEC
           mpbl = ax.scatter(x,y,c=tecs[pidx,4],edgecolors='none',vmin=vx,vmax=vy,s=zz,marker='s',cmap=cmap)
        if rr==0: # TID
           mpbl = ax.scatter(x,y,c=tecs[pidx,3],edgecolors='none',vmin=vx,vmax=vy,s=zz,cmap=cmap)

        ax.add_feature(Nightshade(plotDate, alpha=0.2))

        if title is None:
            ax.set_title(plotDate.strftime('UTC %Y-%m-%d %H:%M:%S'))
        elif title is not False:
            ax.set_title(title)

        cbar = plt.colorbar(mpbl,shrink=0.8,pad=0.075,cax=cax,extend='both')
        cbar.set_label('$\Delta$TECu')#,size='medium')
#        cbar.ax.tick_params(labelsize='medium')

        return {'ax':ax,'cbar':cbar}

    def get_tec_vals(self,plotDate,wgec=False,return_tec_values=False):
        """
        wgec    = False         # generate a data output file for global and US regional mean TEC (0=no 1=yes), default=0
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

        ret_dct = {}

        if wgec:
            gfo.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus,gtidus,gtidus_std,len(pidx0),len(pidus)))
            gfo_left.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus_left,gtidus_left,gtidus_std_left,len(pidx0),len(pidus_left)))
            gfo_mid.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus_mid,gtidus_mid,gtidus_std_mid,len(pidx0),len(pidus_mid)))
            gfo_right.write("%d  %1.5f %7.3f %7.3f %9.5f %9.5f  %7.0f %7.0f\n"%(day,inx*dt/3600,gec,gecus_right,gtidus_right,gtidus_std_right,len(pidx0),len(pidus_right)))

        if return_tec_values:
            ret_dct['pidx']             = pidx            
            ret_dct['pidx0']            = pidx0           
            ret_dct['pidus']            = pidus           
            ret_dct['pidus_left']       = pidus_left      
            ret_dct['pidus_mid']        = pidus_mid       
            ret_dct['pidus_right']      = pidus_right     
            ret_dct['tecs']             = tecs

        ret_dct['gec']                = gec             
        ret_dct['gecus']              = gecus           
        ret_dct['gtidus']             = gtidus          
        ret_dct['gecus_std']          = gecus_std       
        ret_dct['gtidus_std']         = gtidus_std      
        ret_dct['gecus_mid']          = gecus_mid       
        ret_dct['gtidus_mid']         = gtidus_mid      
        ret_dct['gecus_std_mid']      = gecus_std_mid   
        ret_dct['gtidus_std_mid']     = gtidus_std_mid  
        ret_dct['gecus_left']         = gecus_left      
        ret_dct['gtidus_left']        = gtidus_left     
        ret_dct['gecus_std_left']     = gecus_std_left  
        ret_dct['gtidus_std_left']    = gtidus_std_left 
        ret_dct['gecus_right']        = gecus_right     
        ret_dct['gtidus_right']       = gtidus_right    
        ret_dct['gecus_std_right']    = gecus_std_right 
        ret_dct['gtidus_std_right']   = gtidus_std_right

        return ret_dct


if __name__ == '__main__':
    sDate = datetime.datetime(2017,11,3)
    eDate = datetime.datetime(2017,11,4)
    tec_obj = TecPlotter(sDate)
    tec_obj.plot_day()
