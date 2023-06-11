#!/usr/bin/env python3
import os
import datetime
import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from collections import OrderedDict
import tqdm

from . import gen_lib as gl

this_name   = os.path.basename(__file__[:-3])
output_dir  = os.path.join('output/galleries',this_name)
gl.prep_output({0:output_dir})

class SymH():
    def __init__(self,years=[2016,2017],
            symh_pattern    = 'data/kyoto_wdc/SYM-{!s}.dat.txt.bz2',
            ssc_pattern     = 'data/obsebre.es/ssc/ssc_{!s}_*.txt',
            output_dir      = 'output'):
        self.years          = years
        self.symh_pattern   = symh_pattern
        self.ssc_pattern    = ssc_pattern
        self.output_dir     = output_dir

        self._load_symh()
        self._load_sscs()
        self._calc_storms_df()
        
    def _load_symh(self):
        years   = self.years
        pattern = self.symh_pattern

        # Load Kyoto SYM-H #####################
        if years is None:
            fpaths          = glob.glob(pattern.format('*'))
        else:
            fpaths          = [pattern.format(x) for x in years]

        df = pd.DataFrame()
        for fpath in fpaths:
            df_tmp      = pd.read_csv(fpath,sep='\s+',header=14,parse_dates=[['DATE','TIME']])
            df_tmp      = df_tmp.set_index('DATE_TIME')
            df_tmp      = df_tmp[['ASY-D','ASY-H','SYM-D','SYM-H']].copy()
            df          = pd.concat([df,df_tmp])

        df.sort_index(inplace=True)
        tf      = df[:] == 99999
        df[tf]  = np.nan
        df.dropna(inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        self.df = df

    def _load_sscs(self):
        """
        Load multiple SSC files from http://www.obsebre.es/en/rapid into a dataframe
        and produce a list only of SSC times.

        Store these as attributes to the class.
        """
        years   = self.years
        pattern = self.ssc_pattern

        # Load SSC Data Downloaded from http://www.obsebre.es/en/rapid #
        if years is None:
            years   = ['*']

        fpaths  = []
        for yr in years:
            fpaths  += glob.glob(pattern.format(yr))
        fpaths.sort()

        df = pd.DataFrame()
        for fpath in fpaths:
            dft = self._load_ssc(fpath)
            df  = pd.concat([df,dft])

        self.df_ssc     = df
        self.ssc_list   = df.index.to_pydatetime().tolist()

    def _load_ssc(self,fpath):
        """
        Load a single SSC file from http://www.obsebre.es/en/rapid
        into a dataframe.
        """

        print(fpath)
        with open(fpath,'r') as fl:
            lines   = fl.readlines()

        to_df   = []
        for line in lines:
            od  = OrderedDict()

            spl = line.replace('\n','').split()
            try:
                dt  = datetime.datetime( *(int(float(x)) for x in spl[:5]) )
            except:
                continue

            od['datetime']  = dt
            to_df.append(od)

        dft = pd.DataFrame(to_df)
        dft = dft.set_index('datetime')
        return dft

    def plot(self,var='SYM-H',figsize=(10,8)):
        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(1,1,1)
        xx  = self.df.index
        yy  = self.df[var]
        ax.plot(xx,yy)
        ax.set_xlabel('UT Time')
        ax.set_ylabel(var)
        fig.tight_layout()
        fname   = '{!s}.png'.format(var)
        fpath   = os.path.join(self.output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')

    def plot_storms(self,figsize=(15,8),xlim=(-1,3),clear=False):
        t_before    = datetime.timedelta(days=xlim[0]) 
        t_after     = datetime.timedelta(days=xlim[1])

        output_dir  = os.path.join(self.output_dir,'storms')
        gl.prep_output({0:output_dir},clear=clear)

        for ssc in self.ssc_list:
            print(ssc)
            ep0 = ssc + t_before
            ep1 = ssc + t_after

            fig = plt.figure(figsize=figsize)
            ax  = fig.add_subplot(1,1,1)
            ax.axvline(0,ls=':',color='k')
            ax.axhline(0,ls=':',color='k')

            tf  = np.logical_and(self.df.index >= ep0, self.df.index < ep1)
            dft = self.df[tf]
            xx  = (dft.index - ssc).total_seconds()/86400.
            yy  = dft['SYM-H']
            ax.plot(xx,yy,marker='.',label='SYM-H')

            ax.set_title(ssc.strftime('%d %b %Y %H%M UT'))
            ax.set_xlabel('Epoch Time')
            ax.set_ylabel('Sym-H [nT]')

            ax.set_xlim(xlim)
            ax.set_ylim(-500,100)

            ax.legend()

            fig.tight_layout()
            ssc_str = ssc.strftime('%Y%m%d.%H%M')
            fname   = '{!s}.png'.format(ssc_str)
            fpath   = os.path.join(output_dir,fname)
            fig.savefig(fpath,bbox_inches='tight')

    def plot_storms_sea(self,figsize=(15,8),xlim=(-1,3),clear=False):
        t_before    = datetime.timedelta(days=xlim[0]) 
        t_after     = datetime.timedelta(days=xlim[1])

        output_dir  = os.path.join(self.output_dir,'storms')
        gl.prep_output({0:output_dir},clear=clear)

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(1,1,1)
        ax.axvline(0,ls=':',color='k')
        ax.axhline(0,ls=':',color='k')

        for ssc in self.ssc_list:
            print(ssc)
            ep0 = ssc + t_before
            ep1 = ssc + t_after


            tf  = np.logical_and(self.df.index >= ep0, self.df.index < ep1)
            dft = self.df[tf]
            xx  = (dft.index - ssc).total_seconds()/86400.
            yy  = dft['SYM-H']
            ax.plot(xx,yy)

            ax.set_xlabel('Epoch Time')
            ax.set_ylabel('Sym-H [nT]')

            ax.set_xlim(xlim)
            ax.set_ylim(-500,100)

        ssc0    = min(self.ssc_list).strftime('%b %Y')
        ssc1    = max(self.ssc_list).strftime('%b %Y')
        title   = '{!s} - {!s} ({!s} Storms)'.format(ssc0,ssc1,len(self.ssc_list))
        ax.set_title(title)

        fig.tight_layout()
        fname   = 'sea.png'
        fpath   = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')

    def _calc_storms_df(self,xlim=(-1,3)):
        self.xlim   = xlim

        t_before    = datetime.timedelta(days=xlim[0]) 
        t_after     = datetime.timedelta(days=xlim[1])

        # Time Vector in Seconds
        time_vec    = np.arange(xlim[0]*1440,xlim[1]*1440) * 60.
        time_ser    = pd.Series(None,time_vec)

        storms_df    = pd.DataFrame()
        for ssc in self.ssc_list:
#            print(ssc)
            ep0 = ssc + t_before
            ep1 = ssc + t_after

            tf          = np.logical_and(self.df.index >= ep0, self.df.index < ep1)
            dft         = self.df[tf]['SYM-H'].copy()
            tmp         = (dft.index - ssc).total_seconds()
            dft.index   = tmp

            # Interpolate onto standard time grid.
            try:
                df_new      = dft.combine_first(time_ser)
            except:
                import ipdb; ipdb.set_trace()
            df_new      = df_new.interpolate().reindex(time_vec)

            data        = np.array([df_new.tolist()])

            this_df     = pd.DataFrame(data=data,index=[ssc],columns=time_vec)
            storms_df   = pd.concat([storms_df,this_df])
        self.storms_df   = storms_df

    def plot_storms_df_sea(self,figsize=(15,8),clear=False):
        xlim        = self.xlim

        output_dir  = os.path.join(self.output_dir,'storms_df')
        gl.prep_output({0:output_dir},clear=clear)

        fig = plt.figure(figsize=figsize)
        ax  = fig.add_subplot(1,1,1)
        ax.axvline(0,ls=':',color='k')
        ax.axhline(0,ls=':',color='k')

        for ssc,data in self.storms_df.iterrows():
            print(ssc)
            xx  = data.index/86400.
            yy  = data
            ax.plot(xx,yy,alpha=0.4)
        
        hdls    = []
        labels  = []

        xx  = data.index/86400.

        yy  = self.storms_df.quantile(0.75)
        hdl = ax.plot(xx,yy,lw=1,color='k')
        hdls.append(hdl[0])
        labels.append('Upper Quartile')

        yy  = self.storms_df.median()
        hdl = ax.plot(xx,yy,lw=3,color='k')
        hdls.append(hdl[0])
        labels.append('Median')

        yy  = self.storms_df.quantile(0.25)
        hdl = ax.plot(xx,yy,lw=1,color='k')
        hdls.append(hdl[0])
        labels.append('Lower Quartile')

        ax.legend(hdls,labels)
        
        ax.set_xlim(xlim)
        ax.set_ylim(-500,100)

        ax.set_xlabel('Epoch Time')
        ax.set_ylabel('Sym-H [nT]')

        ssc0    = self.storms_df.index.min().strftime('%b %Y')
        ssc1    = self.storms_df.index.max().strftime('%b %Y')
        title   = '{!s} - {!s} ({!s} Storms)'.format(ssc0,ssc1,len(self.ssc_list))
        ax.set_title(title)

        fig.tight_layout()
        fname   = 'sea.png'
        fpath   = os.path.join(output_dir,fname)
        fig.savefig(fpath,bbox_inches='tight')

    def get_closest(self,dt):
        df  = self.df
        inx = np.argmin(np.abs(df.index - dt))
        return df.iloc[inx]

def main():
    years = np.arange(2001,2014)
    symh = SymH(years=years,output_dir=output_dir)
#    symh.plot()
#    symh.plot_storms(clear=True)
#    symh.plot_storms_sea(clear=True)
    symh.plot_storms_df_sea(clear=True)
    import ipdb; ipdb.set_trace()


if __name__ == '__main__':
    main()
