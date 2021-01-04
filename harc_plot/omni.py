import datetime
import glob

from collections import OrderedDict
import bz2

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
from matplotlib import pyplot as plt

from .symh import SymH

def to_ut_hr(dt):
    return dt.hour + dt.minute/60. + dt.second/3600.

def date_parser(row):
    year    = row['year']
    doy     = row['doy']
    hr      = row['hr']

    dt  = datetime.datetime(int(year),1,1)+datetime.timedelta(days=(int(doy)-1),hours=int(hr))
    return dt

class Omni():
    def __init__(self):
        self._load_omni()
        self._load_symh()

    def _load_omni(self,omni_csv='data/omni/omni_data_reduced.txt.bz2'):
        """
        Load OMNI data.
        """

        input_file  = omni_csv

        vrs = OrderedDict()
        vrs['Bartels rotation number']          = {'symbol':'bartels'}
        vrs['ID for IMF spacecraft']            = {'symbol':'IMF_spacecraft_ID',        'NaN':99}
        vrs['ID for SW Plasma spacecraft']      = {'symbol':'SW_spacecraft_ID',         'NaN':99}
        vrs['# of points in IMF averages']      = {'symbol':'N_IMF_points',             'NaN':999}
        vrs['# of points in Plasma averag.']    = {'symbol':'N_plasma_points',          'NaN':999}
        vrs['Scalar B, nT']                     = {'symbol':'B_nT',                     'NaN':999.9}
        vrs['Vector B Magnitude,nT']            = {'symbol':'B_vec_nT',                 'NaN':999.9}
        vrs['Lat. Angle of B (GSE)']            = {'symbol':'B_theta_GSE',              'NaN':999.9}
        vrs['Long. Angle of B (GSE)']           = {'symbol':'B_phi_GSE',                'NaN':999.9}
        vrs['BX, nT (GSE, GSM)']                = {'symbol':'Bx_GSE_nT',                'NaN':999.9}
        vrs['BY, nT (GSE)']                     = {'symbol':'By_GSE_nT',                'NaN':999.9}
        vrs['BZ, nT (GSE)']                     = {'symbol':'Bz_GSE_nT',                'NaN':999.9}
        vrs['BY, nT (GSM)']                     = {'symbol':'By_GSM_nT',                'NaN':999.9}
        vrs['BZ, nT (GSM)']                     = {'symbol':'Bz_GSM_nT',                'NaN':999.9}
        vrs['RMS_magnitude, nT']                = {'symbol':'RMS_mag_nT',               'NaN':999.9}
        vrs['RMS_field_vector, nT']             = {'symbol':'RMS_vec_nT',               'NaN':999.9}
        vrs['RMS_BX_GSE, nT']                   = {'symbol':'RMS_BX_GSE_nT',            'NaN':999.9}
        vrs['RMS_BY_GSE, nT']                   = {'symbol':'RMS_BY_GSE_nT',            'NaN':999.9}
        vrs['RMS_BZ_GSE, nT']                   = {'symbol':'RMS_BZ_GSE_nT',            'NaN':999.9}
        vrs['SW Plasma Temperature, K']         = {'symbol':'T_K',                      'NaN':9999999}
        vrs['SW Proton Density, N/cm^3']        = {'symbol':'N_p',                      'NaN':999.9}
        vrs['SW Plasma Speed, km/s']            = {'symbol':'V',                        'NaN':9999}
        vrs['SW Plasma flow long. angle']       = {'symbol':'SW_phi',                   'NaN':999.9}
        vrs['SW Plasma flow lat. angle']        = {'symbol':'SW_theta',                 'NaN':999.9}
        vrs['Alpha/Prot. ratio']                = {'symbol':'alpha_prot_ratio',         'NaN':9.999}
        vrs['sigma-T,K']                        = {'symbol':'sigma_T_K',                'NaN':9999999}
        vrs['sigma-n, N/cm^3)']                 = {'symbol':'sigma_N_p',                'NaN':999.9}
        vrs['sigma-V, km/s']                    = {'symbol':'sigma_V',                  'NaN':9999}
        vrs['sigma-phi V, degrees']             = {'symbol':'sigma_phi',                'NaN':999.9}
        vrs['sigma-theta V, degrees']           = {'symbol':'sigma_theta',              'NaN':999.9}
        vrs['sigma-ratio']                      = {'symbol':'sigma_ratio',              'NaN':9.999}
        vrs['Flow pressure']                    = {'symbol':'P',                        'NaN':99.99}
        vrs['E elecrtric field']                = {'symbol':'E',                        'NaN':999.99}
        vrs['Plasma betta']                     = {'symbol':'beta',                     'NaN':999.99}
        vrs['Alfen mach number']                = {'symbol':'M_alfven',                 'NaN':999.9}
        vrs['Magnetosonic Much num.']           = {'symbol':'M_magsonic',               'NaN':99.9}
        vrs['Kp index']                         = {'symbol':'Kp_x10'}
        vrs['R (Sunspot No.)']                  = {'symbol':'R'}
        vrs['Dst-index, nT']                    = {'symbol':'Dst_nT'}
        vrs['ap_index, nT']                     = {'symbol':'Ap_nT'}
        vrs['f10.7_index']                      = {'symbol':'F10.7',                    'NaN':999.9}
        vrs['AE-index, nT']                     = {'symbol':'AE',                       'NaN':9999}
        vrs['AL-index, nT']                     = {'symbol':'AL',                       'NaN':99999}
        vrs['AU-index, nT']                     = {'symbol':'AU',                       'NaN':99999}
        vrs['pc-index']                         = {'symbol':'Pc',                       'NaN':999.9}
        vrs['Lyman_alpha']                      = {'symbol':'Lyman_alpha'}
        vrs['Proton flux (>1 Mev)']             = {'symbol':'PF_1MeV',                  'NaN':999999.99}
        vrs['Proton flux (>2 Mev)']             = {'symbol':'PF_2MeV',                  'NaN':99999.99}
        vrs['Proton flux (>4 Mev)']             = {'symbol':'PF_4MeV',                  'NaN':99999.99}
        vrs['Proton flux (>10 Mev)']            = {'symbol':'PF_10MeV',                 'NaN':99999.99}
        vrs['Proton flux (>30 Mev)']            = {'symbol':'PF_30MeV',                 'NaN':99999.99}
        vrs['Proton flux (>60 Mev)']            = {'symbol':'PF_60MeV',                 'NaN':99999.99}
        vrs['Flux FLAG']                        = {'symbol':'flux_flag'}


        # Open File
        if input_file[-4:] == '.bz2':
            with bz2.BZ2File(input_file) as fl:
                lines   = fl.readlines()
            lines   = [x.decode() for x in lines]

        else:
            with open(input_file) as fl:
                lines   = fl.readlines()
        
        # Pull out data, ignore headers and footers.
        data    = []
        names   = OrderedDict()
        for line in lines:
            ln  = line.split()
            try:
                if ln[0] == 'YEAR':
                    cols = ln
                    continue

                col0 = int(ln[0])
                if col0 > 1800:
                    data.append(ln)
                else:
                    names[col0] = ' '.join(ln[1:])
            except:
                pass

        # Get a dictionary that only has the variables in the actual datafile.
        columns = ['year','doy','hr']
        nan_dct = OrderedDict()
        for key,val in names.items():
            result = vrs.get(val)
            if result is None:
                symbol = key
            else:
                symbol          = result['symbol']
                nan_dct[symbol] = result.get('NaN')
            columns.append(symbol)

        # Place into dataframe.
        df          = pd.DataFrame(data,columns=columns,dtype=np.float)

        # Parse Dates and remove old date columns
        df.index    = df.apply(date_parser,axis=1)
        del df['year']
        del df['doy']
        del df['hr']

        # Remove Not-a-Numbers
        for key,val in nan_dct.items():
            if val is not None:
                tf  = df[key] == val
                df.loc[tf,key]  = np.nan

        # Scale Kp
        df['Kp']   = df['Kp_x10']/10.
        del df['Kp_x10']

        self.df         = df 

    def _load_symh(self,years=[2016,2017]):
        """
        Create and load SymH data/object.
        """
        symh            = SymH(years)
        self.symh       = symh
        self.df_symasy  = symh.df


    def plot_dst_kp(self,sTime,eTime,ax,xkey='index',xlabels=True,
            dst_param='Dst_nT',dst_lw=1,kp_markersize=10):
        """
        DST and Kp

        dst_param = ['Dst_nT','SYM-H']
        """
        tf  = np.logical_and(self.df.index >= sTime, self.df.index < eTime)
        df  = self.df[tf].copy()

        ut_hrs  = df.index.map(to_ut_hr)

        lines       =[]

        if dst_param == 'Dst_nT':
            if xkey == 'index':
                xx      = df.index
                xlim    = (sTime,eTime)
            else:
                xx      = ut_hrs
                xlim    = (to_ut_hr(sTime), (eTime-sTime).total_seconds()/3600.)

            yy      = df['Dst_nT'].tolist()
            ylabel  = 'Dst [nT]'
        else:
            xx      = self.df_symasy.index
            yy      = self.df_symasy[dst_param].tolist()
            xlim    = (sTime,eTime)
            ylabel  = 'SYM-H [nT]'

        tmp,        = ax.plot(xx,yy,label=ylabel,color='k',lw=dst_lw)
#        ax.fill_between(xx,0,yy,color='0.75')
        lines.append(tmp)
        ax.set_ylabel(ylabel)
        ax.axhline(0,color='k',ls='--')
        ax.set_ylim(-200,100)
        ax.set_xlim(xlim)

        # Kp ###################################
        ax_1        = ax.twinx()
        ax.set_zorder(ax_1.get_zorder()+1)
        ax.patch.set_visible(False)
        low_color   = 'green'
        mid_color   = 'darkorange'
        high_color  = 'red'
        label       = 'Kp'

        if xkey == 'index':
            xvals       = df.index
        else:
            xvals       = np.array(ut_hrs)

        kp          = np.array(df['Kp'].tolist())

        if len(kp) > 0:
            color       = low_color
            markers,stems,base  = ax_1.stem(xvals,kp,linefmt=color)
            markers.set_color(color)
            markers.set_label('Kp Index')
            markers.set_markersize(kp_markersize)
            lines.append(markers)

            tf = np.logical_and(kp >= 4, kp < 5)
            if np.count_nonzero(tf) > 0:
                xx      = xvals[tf]
                yy      = kp[tf]
                color   = mid_color
                markers,stems,base  = ax_1.stem(xx,yy,linefmt=color)
                markers.set_color(color)
                markers.set_markersize(kp_markersize)
                lines.append(markers)

            tf = kp >= 5
            if np.count_nonzero(tf) > 0:
                xx      = xvals[tf]
                yy      = kp[tf]
                color   = high_color
                markers,stems,base  = ax_1.stem(xx,yy,linefmt=color)
                markers.set_color(color)
                markers.set_markersize(kp_markersize)
                lines.append(markers)

        ax_1.set_ylabel('Kp Index')
        ax_1.set_ylim(0,9)
        ax_1.set_yticks(np.arange(10))
        for tk,tl in zip(ax_1.get_yticks(),ax_1.get_yticklabels()):
            if tk < 4:
                color = low_color
            elif tk == 4:
                color = mid_color
            else:
                color = high_color
            tl.set_color(color)

        if not xlabels:
            for xtl in ax.get_xticklabels():
                xtl.set_visible(False)
        plt.sca(ax)
        return [ax,ax_1]

    def plot_f107(self,sTime,eTime,ax,xkey='index',xlabels=True):
        """
        Plot F10.7
        """
        tf  = np.logical_and(self.df.index >= sTime, self.df.index < eTime)
        df  = self.df[tf].copy()

        ut_hrs  = df.index.map(to_ut_hr)

        lines       =[]

        if xkey == 'index':
            xx      = df.index
            xlim    = (sTime,eTime)
        else:
            xx      = ut_hrs
            xlim    = (to_ut_hr(sTime), (eTime-sTime).total_seconds()/3600.)

        yy          = df['F10.7'].tolist()
        ylabel      = 'F10.7 SFI'
        tmp,        = ax.plot(xx,yy,label=ylabel,color='k',lw=1)
        lines.append(tmp)
        ax.set_ylabel(ylabel)
        ax.set_xlim(xlim)

        if not xlabels:
            for xtl in ax.get_xticklabels():
                xtl.set_visible(False)
        plt.sca(ax)
        return

    def get_closest(self,dt):
        df  = self.df
        inx = np.argmin(np.abs(df.index - dt))
        return df.iloc[inx]
