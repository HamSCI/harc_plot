#!/usr/bin/env python3
"""
Parallel batch-histogram processor for the full Madrigal ham-radio archive.

Runs `harc_plot.calculate_histograms.main()` across a date range using a process
pool, producing one `YYYYMMDD.data.nc.bz2` per day under a config-named output
directory. Each worker processes one day at a time; days with existing output
are skipped unless --reprocess is given.

Networks are kept separate: choose one via --network (WSPRNet, RBN, PSKReporter,
or All for the combined view). For the three-networks-separate workflow, invoke
this script three times.

Typical usage for the full 5.1 TB pass:

    ./batch_histograms.py --start 2009-01-01 --stop 2027-01-01 \\
        --network WSPRNet --workers 8
"""
from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import os
import sys
import time
from functools import partial

import dateutil.parser
import tqdm

import harc_plot

# Map canonical network name -> harc_plot source integer (see gen_lib.sources).
NETWORK_TO_SOURCES = {
    'WSPRNet':     [1],
    'RBN':         [2],
    'PSKReporter': [3],
    'All':         [1, 2, 3],
}


def _process_day(date, *, data_sources, region, rgc_lim, xkeys, params,
                 xb_size_min, yb_size_km, data_dir):
    """Worker: process a single UTC day. Always reprocess=False here; the driver
    handles clearing the output dir once up front so workers do not race on it.

    Exceptions are caught and returned as (date, error_message) so one bad Madrigal
    file does not tear down the whole pool. Observed failure modes include truncated
    HDF5 files (HDF5ExtError) and files missing /Data/Table Layout (NoSuchNodeError)
    — all upstream ingestion issues at MIT Haystack, not bugs in the pipeline."""
    rd = {
        'sDate':              date,
        'eDate':              date + datetime.timedelta(days=1),
        'params':             params,
        'xkeys':              xkeys,
        'rgc_lim':            rgc_lim,
        'filter_region':      region,
        'filter_region_kind': 'mids',
        'xb_size_min':        xb_size_min,
        'yb_size_km':         yb_size_km,
        'loc_sources':        None,
        'data_sources':       data_sources,
        'reprocess':          False,
        'output_dir':         data_dir,
        'band_obj':           harc_plot.gl.BandData(),
    }
    try:
        harc_plot.calculate_histograms.main(rd)
        return (date, None)
    except Exception as e:
        return (date, f'{type(e).__name__}: {e}')


def _daterange(start, stop):
    """Inclusive [start, stop) daily range."""
    d = start
    while d < stop:
        yield d
        d = d + datetime.timedelta(days=1)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--start', required=True, help='Inclusive start date (YYYY-MM-DD)')
    parser.add_argument('--stop',  required=True, help='Exclusive stop date  (YYYY-MM-DD)')
    parser.add_argument('--network', choices=NETWORK_TO_SOURCES, default='All',
                        help='Network to include (default: All)')
    parser.add_argument('--workers', type=int, default=max(1, mp.cpu_count() // 2),
                        help='Parallel worker processes (default: half of CPU count)')
    parser.add_argument('--region', default='World',
                        help='Regional filter from harc_plot.gl.regions (default: World)')
    parser.add_argument('--rgc-min', type=float, default=0.0, help='Min path length, km')
    parser.add_argument('--rgc-max', type=float, default=10000.0, help='Max path length, km')
    parser.add_argument('--xb-min', type=float, default=60.0,
                        help='Time bin size, minutes (default: 60 for solarcycle overview)')
    parser.add_argument('--yb-km', type=float, default=250.0, help='Range bin size, km')
    parser.add_argument('--xkey', default='slt_mid', choices=['ut_hrs', 'slt_mid'],
                        help='Time-axis coordinate (default: slt_mid)')
    parser.add_argument('--output-root', default='data/solarcycle/histograms',
                        help='Parent dir for per-config output subfolders')
    parser.add_argument('--reprocess', action='store_true',
                        help='Overwrite existing day outputs instead of skipping')
    args = parser.parse_args(argv)

    sDate = dateutil.parser.isoparse(args.start)
    eDate = dateutil.parser.isoparse(args.stop)
    if sDate >= eDate:
        parser.error(f'--start ({args.start}) must be before --stop ({args.stop})')

    dates = list(_daterange(sDate, eDate))

    # Config-keyed output dir: this makes runs for different configs coexist cleanly.
    run_name = (f'{args.region}-{args.network}'
                f'-{int(args.xb_min)}min-{int(args.yb_km)}km'
                f'-{int(args.rgc_min)}-{int(args.rgc_max)}km')
    data_dir = os.path.join(args.output_root, run_name)
    # Clear the output dir exactly once up front (if --reprocess), then workers
    # always run with reprocess=False and simply skip existing per-day outputs.
    harc_plot.gl.prep_output({0: data_dir}, clear=args.reprocess)

    worker = partial(
        _process_day,
        data_sources = NETWORK_TO_SOURCES[args.network],
        region       = args.region,
        rgc_lim      = (args.rgc_min, args.rgc_max),
        xkeys        = [args.xkey],
        params       = ['spot_density'],
        xb_size_min  = args.xb_min,
        yb_size_km   = args.yb_km,
        data_dir     = data_dir,
    )

    print(f'Processing {len(dates)} days × network={args.network} × region={args.region}',
          file=sys.stderr)
    print(f'  workers:    {args.workers}', file=sys.stderr)
    print(f'  output dir: {data_dir}', file=sys.stderr)
    print(f'  reprocess:  {args.reprocess}', file=sys.stderr)

    t0 = time.time()
    failed = []  # list of (date, error_message)
    # spawn context avoids fork-safety issues with pytables / HDF5 file handles.
    ctx = mp.get_context('spawn')
    with ctx.Pool(args.workers) as pool:
        it = pool.imap_unordered(worker, dates, chunksize=1)
        for result_date, err in tqdm.tqdm(it, total=len(dates), desc='days', unit='day'):
            if err is not None:
                failed.append((result_date, err))
    dt = time.time() - t0
    print(f'Done in {dt:.1f}s ({dt/max(len(dates),1):.1f}s/day avg).', file=sys.stderr)
    if failed:
        print(f'{len(failed)} days failed:', file=sys.stderr)
        for d, err in sorted(failed):
            print(f'  {d.strftime("%Y-%m-%d")}  {err}', file=sys.stderr)
        # Also write a machine-readable list next to the output dir for later triage.
        fail_path = os.path.join(data_dir, '_failed_dates.txt')
        with open(fail_path, 'w') as fh:
            for d, err in sorted(failed):
                fh.write(f'{d.strftime("%Y-%m-%d")}\t{err}\n')
        print(f'Wrote failure list to {fail_path}', file=sys.stderr)


if __name__ == '__main__':
    main()
