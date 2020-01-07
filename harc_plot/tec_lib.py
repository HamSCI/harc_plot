#!/usr/bin/env python3
import sys, os.path
import glob

import datetime

import numpy as np
import pandas as pd
import h5py
import tqdm

import madrigalWeb.madrigalWeb

from . import gen_lib as gl

def download_tec(sTime,eTime,data_dir='data'):
    gl.prep_output({0:data_dir},clear=False)

    # constants
    user_fullname = 'Nathaniel Frissell'
    user_email = 'nathaniel.a.frissell@njit.edu'
    user_affiliation = 'New Jersey Institute of Technology'

    if len(sys.argv) > 1:
        madrigalUrl = sys.argv[1]
    else:
        madrigalUrl = 'http://madrigal.haystack.mit.edu/madrigal'

    madObj = madrigalWeb.madrigalWeb.MadrigalData(madrigalUrl)

    inst_code   = 8000
    print('--> Instrument Name')
    instList = madObj.getAllInstruments()
    for inst in instList:
        if inst.code == inst_code:
            print((str(inst) + '\n'))
            

    print('--> Experiment Names')
    expList = madObj.getExperiments(inst_code,sTime.year,sTime.month,sTime.day,0,0,0,eTime.year,eTime.month,eTime.day,0,0,0)
    for exp in expList:
        print((str(exp) + '\n'))

    print('--> Download Data Files')
    fileList    = []
    for exp in expList:
        fls  = madObj.getExperimentFiles(exp.id)
        fileList += fls

    for thisFile in fileList:
        if thisFile.category == 1:
            print((str(thisFile) + '\n'))
            fname   = os.path.basename(thisFile.name)+'.hd5'
            fpath   = os.path.join(data_dir,fname)
            result  = madObj.downloadFile(thisFile.name, fpath,
                               user_fullname, user_email, user_affiliation, "hdf5")
        
def load_tec(sTime,eTime,data_dir,region=None):
    files   = glob.glob(os.path.join(data_dir,'*.hd5'))

    df  = pd.DataFrame()
    for file_name in tqdm.tqdm(files):
        print('Loading {!s}...'.format(file_name))
        fl      = h5py.File(file_name,'r+')
        data    = fl['Data']['Table Layout'].value
        dft     = pd.DataFrame(data)

        dft     = dft.rename({'min':'minute','sec':'second'},axis='columns')
        dt_keys = ['year', 'month', 'day', 'hour', 'minute', 'second']
        dft['datetime'] = pd.to_datetime(dft[dt_keys])

        if region is not None:
            rg              = gl.regions.get(region)
            lat_0, lat_1    = rg['lat_lim']
            lon_0, lon_1    = rg['lon_lim']
            
            tf_lat  = np.logical_and(dft.gdlat >= lat_0, dft.gdlat < lat_1) 
            tf_lon  = np.logical_and(dft.glon  >= lon_0, dft.glon  < lon_1) 
            tf      = np.logical_and(tf_lat,tf_lon)
            dft     = dft[tf]

        keypers = []
        keypers.append('datetime')
        keypers.append('gdlat')
        keypers.append('glon')
        keypers.append('tec')
        keypers.append('dtec')
#        keypers.append('recno')
#        keypers.append('kindat')
#        keypers.append('kinst')
#        keypers.append('gdalt')
        
        dft = dft[keypers].copy()
        df  = df.append(dft,ignore_index=True)

    print('TEC Date Filter...')
    tf  = np.logical_and(df['datetime'] >= sTime,
                         df['datetime'] <  eTime)
    df  = df[tf].copy()

    return df
        
if __name__ == '__main__':
    data_dir  = 'data/tec'

    output_dir  = 'output/galleries/tec'
    gl.prep_output({0:output_dir},clear=False)

    sTime       = datetime.datetime(2017,9,4)
    eTime       = datetime.datetime(2017,9,5)
    download    = False

    if download:
        download_tec(sTime,eTime,data_dir=data_dir)

    df = load_tec(sTime,eTime,data_dir)

    import ipdb; ipdb.set_trace()
