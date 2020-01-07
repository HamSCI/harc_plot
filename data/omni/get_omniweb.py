#!/usr/bin/env python3
import os
import datetime
from collections import OrderedDict

def download_omni(sDate,eDate,res='hour',output_file='omni_data.txt'):
    """
    Download Plasma and Geomagnetic Indices Data from NASA OmniWeb.
    This script allows for automated downloading of data from:
    https://omniweb.gsfc.nasa.gov/form/dx1.html

    sDate:  datetime.datetime
    eDate:  datetime.datetime
    res:    'hour', 'daily', or '27day'
    """

    spacecraft  = 'omni2'

    vrs = OrderedDict()

#    vrs['3']    = 'Bartels Rotation Number'
#    vrs['4']    = 'IMF Spacecraft ID'
#    vrs['5']    = 'Plasma Spacecraft ID'
#    
#    vrs['6']    = 'Fine Scale Points in IMF Avgs'
#    vrs['7']    = 'Fine Scale Points in Plasma Avgs '
    
    # Magnetic Field
#    vrs['8']    = 'IMF Magnitude Avg, nT'
#    vrs['9']    = 'Magnitude, Avg IMF Vr, nT'
#    vrs['10']   = 'Lat. of Avg. IMF, deg.'
#    vrs['11']   = 'Long. of Avg. IMF, deg. '
#    vrs['12']   = 'Bx, GSE/GSM, nT'
#    vrs['13']   = 'By, GSE, nT '
#    vrs['14']   = 'Bz, GSE, nT'
    
#    vrs['15']   = 'By, GSM, nT'
#    vrs['16']   = 'Bz, GSM, nT'
#    vrs['17']   = 'Sigma in IMF Magnitude Avg.'
#    vrs['18']   = 'Sigma in IMF Vector Avg'
#    vrs['19']   = 'Sigma Bx, nT'
#    vrs['20']   = 'Sigma By, nT'
#    vrs['21']   = 'Sigma Bz, nT'
#    
#    # Plasma
#    vrs['22']   = 'Proton Temperature, K'
#    vrs['23']   = 'Proton Density, n/cc'
#    vrs['24']   = 'Flow Speed, km/sec'
#    vrs['25']   = 'Flow Longitude, deg.'
#    vrs['26']   = 'Flow Latitude, deg.'
#    vrs['27']   = 'Alpha/Proton Density Ratio'
#    
#    vrs['29']   = 'Sigma-T'
#    vrs['30']   = 'Sigma-Np'
#    vrs['31']   = 'Sigma-V'
#    vrs['32']   = 'Sigma-Flow-Longitude'
#    vrs['33']   = 'Sigma-Flow-Latitude'
#    vrs['34']   = 'Sigma-Alpha/Proton Ratio'
#    
#    # Derived Parameters
#    vrs['28']   = 'Flow Pressure, nPa'
#    vrs['35']   = 'Ey - Electric Field, mV/m'
#    vrs['36']   = 'Plasma Beta'
#    
#    vrs['37']   = 'Alfven Mach Number'
#    vrs['54']   = 'Magnetosonic Mach Number'
    
    # Indices
    vrs['38']   = 'Kp*10 Index'
    vrs['39']   = 'R Sunspot Number (new version)'
    vrs['40']   = 'Dst Index, nT '
    vrs['49']   = 'ap index, nT'
    vrs['50']   = 'Solar index F10.7'
    
    vrs['41']   = 'AE Index, nT '
    vrs['52']   = 'AL Index, nT'
    vrs['53']   = 'AU Index, nT'
#    vrs['51']   = 'Polar Cap (PCN) index from Thule '
#    vrs['55']   = 'Lyman Alpha Solar Index'
#    
#    # Particles
#    vrs['42']   = 'Proton Flux* > 1 MeV'
#    vrs['43']   = 'Proton Flux* > 2 MeV'
#    vrs['44']   = 'Proton Flux* > 4 MeV '
#    vrs['45']   = 'Proton Flux* > 10 MeV'
#    
#    vrs['46']   = 'Proton Flux* > 30 MeV'
#    vrs['47']   = 'Proton Flux* > 60 MeV'
#    vrs['48']   = 'Magnetospheric Flux Flag'

    sDate_str   = sDate.strftime('%Y%m%d')
    eDate_str   = eDate.strftime('%Y%m%d')

    codes   = ['vars={!s}'.format(x) for x in vrs.keys()]
    var_str = '&'.join(codes)
    cmd     = 'wget --post-data "activity=retrieve&res={!s}&spacecraft={!s}&start_date={!s}&end_date={!s}&{!s}&scale=Linear&ymin=&ymax=&view=0&charsize=&xstyle=0&ystyle=0&symbol=0&symsize=&linestyle=solid&table=0&imagex=640&imagey=480&color=&back=" https://omniweb.sci.gsfc.nasa.gov/cgi/nx1.cgi -O {!s}'.format(
                    res,spacecraft,sDate_str,eDate_str,var_str,output_file)
    os.system(cmd)

if __name__ == '__main__':
    sDate       = datetime.datetime(2000,1,1)
    eDate       = datetime.datetime(2018,9,30)
    output_file = 'omni_data.txt'
    
    download_omni(sDate,eDate,output_file=output_file)
