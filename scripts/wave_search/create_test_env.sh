#!/bin/bash
MPL=3.4.1
ENV=mpl${MPL}

conda remove -y --name $ENV --all
conda create -y -n $ENV python=3.7.7
source ~/anaconda3/bin/activate $ENV

conda install -y -c conda-forge matplotlib=${MPL}
conda install -y pandas scipy tqdm netCDF4 xarray
conda install -y -c conda-forge cartopy ipdb

#cd /usr/local/MATLAB/R2019b/extern/engines/python
#python3 setup.py install

cd ~/code/pyDARNio
pip install -e .

cd ~/code/pydarn
pip install -e .

conda list
