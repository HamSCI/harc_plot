# HamSCI Amateur Radio Communications (HARC) Database Plotting Toolkit

Tools for loading and plotting HF amateur radio propagation spot data — from
RBN, WSPRNet, and PSKReporter — as distributed via the MIT Haystack CEDAR
[Madrigal database](http://cedar.openmadrigal.org/).

## Requirements

* Linux or macOS (tested on Ubuntu 22.04+). Windows is not supported directly;
  WSL2 works.
* Python 3.10 or newer.
* Conda recommended (miniconda or Anaconda) for managing the scientific Python
  stack.

Python dependencies are listed in `requirements.txt`. The main ones:
`numpy`, `pandas`, `xarray`, `netCDF4`, `tables` (pytables), `h5py`,
`matplotlib`, `cartopy`, `madrigalWeb`.

## Install

```bash
# Create and activate a conda environment
conda create -n harc_plot -c conda-forge python=3.11 cartopy pytables netCDF4 xarray pandas matplotlib tqdm
conda activate harc_plot

# Clone and install
git clone https://github.com/HamSCI/harc_plot.git
cd harc_plot
pip install -e .
```

`cartopy` is installed via conda because its binary dependencies (PROJ, GEOS)
are awkward to get from pip.

## Data input

The toolkit reads spot data in two formats:

1. **Madrigal HDF5 (preferred)** — files named `rsdYYYY-MM-DD.01.hdf5`, placed
   in `data/madrigal/`. Downloadable from the Madrigal database via
   `madrigalWeb`. A sample (`data/madrigal/rsd2015-06-10.01.hdf5`, ~39 MB,
   covering all three networks) ships with the repository for testing.
2. **Legacy CSV** — files named `YYYY-MM-DD.csv.bz2` placed in `data/spot_csvs/`
   (pre-Madrigal HamSCI pipeline).

`harc_plot.gen_lib.load_spots_csv()` auto-detects which format is present.

### A note about Madrigal timestamps

A subset of Madrigal ham-radio files (empirically: most of 2023 and 2024,
other years clean) have their `ut1_unix`/`ut2_unix` columns shifted by a
local-timezone offset. The HDF5 loader (`load_madrigal_hdf5()`) detects this
per file by comparing `ut1_unix` against the authoritative `y/m/d/h/m/s` byte
columns and corrects or falls back as needed. You will see a `RuntimeWarning`
on any affected file.

## Test installation

From the `harc_plot` directory:

```bash
cd scripts/basic_histogram
./histograms.py
```

This loads the bundled sample (`data/madrigal/rsd2015-06-10.01.hdf5`), bins it
into 2D histograms, and produces four plots:

```
output/galleries/histograms/World/ut_hrs/20150610.0000UT-20150611.0000UT.data.nc.ut_hrs.spot_density.png
output/galleries/histograms/World/slt_mid/20150610.0000UT-20150611.0000UT.data.nc.slt_mid.spot_density.png
output/galleries/histograms/World/ut_hrs/dailies/20150610.0000UT-20150611.0000UT.data.nc.ut_hrs.spot_density.png
output/galleries/histograms/World/slt_mid/dailies/20150610.0000UT-20150611.0000UT.data.nc.slt_mid.spot_density.png
```

![Example Plot](example_plots/20150610.0000UT-20150611.0000UT.data.nc.ut_hrs.spot_density.png)
