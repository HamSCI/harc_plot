#!/usr/bin/env python3

import os
import datetime

import numpy as np
import pandas as pd


#    """
#    Load spots from CSV file and filter for network/location source quality.
#    Also provide range and regional filtering, compute midpoints, ut_hrs,
#    and slt_mid.
#
#    data_sources: list, i.e. [1,2]
#    ¦   0: dxcluster
#    ¦   1: WSPRNet
#    ¦   2: RBN
#
#    loc_sources: list, i.e. ['P','Q']
#    ¦   P: user Provided
#    ¦   Q: QRZ.com or HAMCALL
#    ¦   E: Estimated using prefix
#    """

csv_path = 'data/spot_csvs/2009-11-01.csv.bz2'
df  	 = pd.read_csv(csv_path,parse_dates=['occurred'])

tf = df['source'] == 2
df_rbn = df[tf].copy()

import ipdb; ipdb.set_trace()
