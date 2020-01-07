import os
import glob
import datetime
from collections import OrderedDict

import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd

import tqdm

from . import gen_lib as gl
from .geospace_env import GeospaceEnv

class HistogramSelector(object):
    def __init__(self,pattern='*.data.nc.bz2',geospace_env=None,**kwargs):

        self.run_dct            = kwargs
        self.run_dct['pattern'] = pattern

        if geospace_env is None:
            geospace_env    = GeospaceEnv()
        self.geospace_env   = geospace_env

        self.gen_filetable()
        self.select_dates()
        self.select_symh()
        self.select_kp()
        self.apply_selection()
        self.create_links()
        self.write_reports()
        self.plot_summary()

    def gen_filetable(self):
        """
        Create a filetable in a dataframe with all available datafiles.
        """

        input_dir  = self.run_dct.get('input_dir')
        pattern    = self.run_dct.get('pattern')

        files   = glob.glob(os.path.join(input_dir,pattern))
        files.sort()
        
        idx_lst     = [] 
        data_lst    = []
        for fl in files:
            bn      = os.path.basename(fl)
            fl_date = datetime.datetime.strptime(bn[:8],'%Y%m%d')

            dct = OrderedDict()
            dct['input_file']   = fl
            dct['keep']         = True

            data_lst.append(dct)
            idx_lst.append(fl_date)
        ft  = pd.DataFrame(data_lst,index=idx_lst)
        self.filetable = ft

    def apply_selection(self):
        ft      = self.filetable

        keep    = ft['keep']
        reject  = np.logical_not(keep)

        self.filetable          = ft[keep].copy()
        self.filetable_reject   = ft[reject].copy()

    def create_links(self):
        """
        Clear output_directory and make the softlinks to files listed in
        the filetable.
        """

        output_dir = self.run_dct.get('output_dir')

        gl.prep_output({0:output_dir},clear=True)

        for rinx,row in self.filetable.iterrows():
            input_file  = row['input_file']
            input_link  = os.path.relpath(input_file,output_dir)
            bn          = os.path.basename(input_file)
            output_file = os.path.join(output_dir,bn)
            os.symlink(input_link,output_file)

            print('Symlink: {!s} --> {!s}'.format(input_file,output_file))

    def write_reports(self):
        output_dir = self.run_dct.get('output_dir')

        rpts    = []
        rpts.append(('filetable.csv',       self.filetable))
        rpts.append(('filetable_reject.csv',self.filetable_reject))

        hdr = []
        for key,val in self.run_dct.items():
            line    = '# {!s}: {!s}\n'.format(key,val)
            hdr.append(line)
        hdr.sort()

        hdr.append('#\n')

        for fname,ft in rpts:
            fpath   = os.path.join(output_dir,fname)

            with open(fpath,'w') as fl:
                line    = '# {!s}\n'.format(fpath)
                fl.write(line)
                fl.write('#\n')
                for line in hdr:
                    fl.write(line)

            ft.to_csv(fpath,mode='a')

    def plot_summary(self):
        output_dir = self.run_dct.get('output_dir')

        rpts    = []
        rpts.append(('filetable.png',       self.filetable))
        rpts.append(('filetable_reject.png',self.filetable_reject))

        for fname,ft in rpts:
            fpath   = os.path.join(output_dir,fname)
            print('Plotting {!s}...'.format(fpath))

            nx      = 1
            ny      = 3
            plt_nr  = 0

            fig     = plt.figure(figsize=(20,ny*6))

            # Plot Sym-H ###########################
            print('   Sym-H')
            plt_nr  += 1
            ax      = fig.add_subplot(ny,nx,plt_nr)

            df              = self.geospace_env.symh.df.copy()
            df['ut_hrs']    = df.index.map(lambda x: x.hour + x.minute/60.)

            for rinx,row in tqdm.tqdm(ft.iterrows(),total=len(ft)):
                sT  = rinx
                eT  = rinx + pd.Timedelta('1D')
                tf  = np.logical_and(df.index >= sT, df.index < eT)
            
                dft = df[tf]

                xx  = dft['ut_hrs']
                yy  = dft['SYM-H']

                ax.plot(xx,yy)

            ax.axhline(0,color='k',ls=':')
            ax.set_xlim(0,24)
            ax.set_ylim(-600,200)
            ax.set_xlabel('UT Hours')
            ax.set_ylabel('Sym-H [nT]')

            # Plot Kp ##############################
            print('   Kp')
            plt_nr  += 1
            ax      = fig.add_subplot(ny,nx,plt_nr)

            df  = self.geospace_env.omni.df['Kp'].to_frame()
            df['ut_hrs']    = df.index.map(lambda x: x.hour + x.minute/60.)

            for rinx,row in tqdm.tqdm(ft.iterrows(),total=len(ft)):
                sT  = rinx
                eT  = rinx + pd.Timedelta('1D')
                tf  = np.logical_and(df.index >= sT, df.index < eT)
            
                dft = df[tf]

                xx  = dft['ut_hrs']
                yy  = dft['Kp']

                ax.plot(xx,yy,ls='',marker='o')

            ax.set_xlim(0,24)
            ax.set_ylim(0,10)
            ax.set_xlabel('UT Hours')
            ax.set_ylabel('Kp')

            plt_nr  += 1
            ax      = fig.add_subplot(ny,nx,plt_nr)
            ax.plot(np.arange(100))

            fig.tight_layout()

            title   = []
            title.append(fpath)
            if len(ft) > 0:
                dt_str0 = ft.index.min().strftime('%d %b %Y')
                dt_str1 = ft.index.max().strftime('%d %b %Y')
                dt_str  = '{!s} - {!s}'.format(dt_str0,dt_str1)
            else:
                dt_str  = 'no_events'
            n_str   = 'N = {!s}'.format(len(ft))
            line    = '{!s} ({!s})'.format(n_str,dt_str)
            title.append(line)
            fig.text(0.5,1,'\n'.join(title),fontdict={'weight':'bold','size':26},ha='center')

            fig.savefig(fpath,bbox_inches='tight')
            plt.close(fig)

    def select_dates(self):
        """
        Drop dates based on range from filetable.
        Don't put on reject list, because we only want days on to
        be on that list for geophysical reasons.
        """
        sTime       = self.run_dct.get('sTime')
        eTime       = self.run_dct.get('eTime')
        ft          = self.filetable
        ft_raw  = ft.copy()

        if sTime is not None:
            sDate   = datetime.datetime(sTime.year,sTime.month,sTime.day)
            ft      = ft[ft.index >= sDate].copy()

        if eTime is not None:
            eDate   = datetime.datetime(eTime.year,eTime.month,eTime.day)
            ft      = ft[ft.index <= eDate].copy()
        
        self.filetable  = ft

    def select_symh(self,symh_min=None,symh_max=None,**kwargs):
        symh_min    = self.run_dct.get('symh_min')
        symh_max    = self.run_dct.get('symh_max')

        ft      = self.filetable
        symh    = self.geospace_env.symh.df['SYM-H']

        symh_mins = []
        symh_maxs = []
        for rinx,row in ft.iterrows():
            sT  = rinx
            eT  = rinx + pd.Timedelta('1D')
            tf  = np.logical_and(symh.index >= sT, symh.index < eT)

            symh_mins.append(symh[tf].min())
            symh_maxs.append(symh[tf].max())

        ft['symh_min']  = symh_mins
        ft['symh_max']  = symh_maxs

        if symh_min is not None:
            tf  = ft['symh_min'] < symh_min
            ft.loc[tf,'keep']   = False

        if symh_max is not None:
            tf  = ft['symh_max'] >= symh_max
            ft.loc[tf,'keep']   = False

        self.filetable  = ft

    def select_kp(self,kp_min=None,kp_max=None,**kwargs):
        kp_min      = self.run_dct.get('kp_min')
        kp_max      = self.run_dct.get('kp_max')

        ft          = self.filetable
        kp          = self.geospace_env.omni.df['Kp']

        kp_mins = []
        kp_maxs = []
        for rinx,row in ft.iterrows():
            sT  = rinx
            eT  = rinx + pd.Timedelta('1D')
            tf  = np.logical_and(kp.index >= sT, kp.index < eT)

            kp_mins.append(kp[tf].min())
            kp_maxs.append(kp[tf].max())

        ft['kp_min']  = kp_mins
        ft['kp_max']  = kp_maxs

        if kp_min is not None:
            tf  = ft['kp_min'] < kp_min
            ft.loc[tf,'keep']   = False

        if kp_max is not None:
            tf  = ft['kp_max'] >= kp_max
            ft.loc[tf,'keep']   = False

        self.filetable  = ft

def main(run_dct):
    hgs = HistogramSelector(**run_dct)
