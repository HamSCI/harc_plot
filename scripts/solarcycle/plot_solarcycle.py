#!/usr/bin/env python3
"""
Aggregate per-day histogram NetCDFs produced by batch_histograms.py into a
single multi-year solar-cycle panel plot.

For each --network, reads every YYYYMMDD.data.nc.bz2 under the matching
config-keyed input dir, concatenates into an xarray Dataset, and renders
time-series + map panels via harc_plot.visualize_histograms.ncLoader.

The loaded-and-concatenated Dataset is pickled to a cache file so iterating
on plot styling does not re-read all ~4000 daily NetCDFs every time.

Typical usage:

    ./plot_solarcycle.py --network WSPRNet
    ./plot_solarcycle.py --network RBN --start 2009-02-25
    ./plot_solarcycle.py --network PSKReporter --reset-cache
"""
from __future__ import annotations

import argparse
import bz2
import datetime
import hashlib
import os
import pickle
import sys

import dateutil.parser
import harc_plot
from harc_plot import visualize_histograms as vh


NETWORK_DEFAULT_START = {
    'WSPRNet':     '2015-01-01',  # First WSPRNet data in Madrigal
    'RBN':         '2009-02-25',  # Full RBN archive span
    'PSKReporter': '2015-01-01',  # First PSKReporter data in Madrigal
    'All':         '2015-01-01',  # Limited by network with latest start
}


def _hash_rd(rd):
    """Short deterministic hash of the run dict for cache keying."""
    key = repr(sorted((k, repr(v)) for k, v in rd.items())).encode()
    return hashlib.sha1(key).hexdigest()[:12]


def _load_nc_cache(rd, cache_dir, reset_cache=False):
    """Load-and-concatenate NetCDFs, caching the xarray result to a pickle."""
    h = _hash_rd(rd)
    fname = f"{rd['sTime']:%Y%m%d}-{rd['eTime']:%Y%m%d}.{h}.p.bz2"
    fpath = os.path.join(cache_dir, fname)
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(fpath) and reset_cache:
        os.remove(fpath)

    if os.path.exists(fpath):
        print(f'Loading cached concatenation: {fpath}', file=sys.stderr)
        with bz2.BZ2File(fpath, 'rb') as fh:
            return pickle.load(fh)

    print(f'Building concatenation (no cache at {fpath})', file=sys.stderr)
    nc = vh.ncLoader(**rd)
    nc.rd_original = rd
    with bz2.BZ2File(fpath, 'wb') as fh:
        pickle.dump(nc, fh)
    print(f'Cached to {fpath}', file=sys.stderr)
    return nc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--network', choices=list(NETWORK_DEFAULT_START),
                        required=True)
    parser.add_argument('--start', default=None,
                        help='Inclusive start YYYY-MM-DD (default: network-dependent)')
    parser.add_argument('--stop', default='2026-03-18',
                        help='Exclusive stop YYYY-MM-DD (default: 2026-03-18)')
    parser.add_argument('--region', default='World')
    parser.add_argument('--xb-min', type=float, default=60.0)
    parser.add_argument('--yb-km',  type=float, default=250.0)
    parser.add_argument('--rgc-min', type=float, default=0.0)
    parser.add_argument('--rgc-max', type=float, default=10000.0)
    parser.add_argument('--data-root', default='/scratch/w2naf/harc_plot_solarcycle',
                        help='Parent dir containing per-network config subfolders')
    parser.add_argument('--plot-root', default='/scratch/w2naf/harc_plot_solarcycle/plots',
                        help='Output dir for plots')
    parser.add_argument('--bands', nargs='+', type=int,
                        default=[28, 21, 14, 7, 3, 1],
                        help='Band keys (int MHz from BandData)')
    parser.add_argument('--xkey', default='slt_mid', choices=['slt_mid', 'ut_hrs'])
    parser.add_argument('--reset-cache', action='store_true')
    args = parser.parse_args(argv)

    if args.start is None:
        args.start = NETWORK_DEFAULT_START[args.network]

    sTime = dateutil.parser.isoparse(args.start)
    eTime = dateutil.parser.isoparse(args.stop)

    run_name = (f'{args.region}-{args.network}'
                f'-{int(args.xb_min)}min-{int(args.yb_km)}km'
                f'-{int(args.rgc_min)}-{int(args.rgc_max)}km')
    data_dir = os.path.join(args.data_root, run_name)
    plot_dir = os.path.join(args.plot_root, run_name)
    cache_dir = os.path.join(args.plot_root, 'cache')

    if not os.path.isdir(data_dir):
        parser.error(f'input dir not found: {data_dir} — did the batch run complete?')

    # ncLoader matches filenames against the date range, so giving it the full
    # glob is fine; out-of-range days are simply ignored.
    srcs = os.path.join(data_dir, '*.data.nc.bz2')

    rd = {
        'srcs':         srcs,
        'sTime':        sTime,
        'eTime':        eTime,
        'band_keys':    args.bands,
        'xkeys':        [args.xkey],
        'baseout_dir':  plot_dir,
        'plot_region':  args.region,
        'geospace_env': None,
        'plot_sza':     False,
        'plot_trend':   True,
        'plot_kpsymh':  False,  # Reference data is 2017-era only; turn back on once #4 is done.
        'plot_goes':    False,  # Ditto.
        'plot_f107':    True,   # F10.7 is the main solar-activity axis for this paper.
        'log_z':        False,
    }

    nc = _load_nc_cache(rd, cache_dir=cache_dir, reset_cache=args.reset_cache)
    if nc.datasets is None:
        print(f'No data loaded for {run_name} in [{sTime}, {eTime}). '
              f'Check the input dir; aborting.', file=sys.stderr)
        return 1

    paths = nc.plot(**rd)
    if paths:
        print('Wrote plots:', file=sys.stderr)
        for p in paths:
            print(f'  {p}', file=sys.stderr)


if __name__ == '__main__':
    sys.exit(main() or 0)
