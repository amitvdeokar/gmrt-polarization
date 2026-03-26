"""

Description
-----------
Dedisperses raw Stokes data and folds it using the pulsar’s rotation period,
producing per‑frequency folded profiles that are written to individual ASCII
files. The number of phase bins for each profile is set by the pulsar’s period
and the sampling time.

Parameters
----------
-sf : str
    Path to stokes file
-n : int
    Number of spectral channels
-sb : int
    Sideband (Option : 1 or -1)
-nd : int
    Number of data blocks
-nc : int
    Number of cores
-fl : int
     Lowest observational frequency (in MHz)
-fh : int
     Highest observational frequency (in MHz)
-st : float
    Sampling time (in sec)
-dm : float
    Dispersion Measure (pc/cm^3)
-pp : float
    Pulsar period (in sec)
-awin : list
    Analysis window

Returns
-------
A mprof file which includes number of channels, the number of bins, and the sampling time in the first line, followed by the folding
data.

Usage
-----
python psr_dsp_fold_v1.py -sf J0139+5814.raw.stki -n 8192 -nd 100 -fl 100 -fh 500 -sb -1 -st 655.35e-6 -dm 73.811 -pp 272.47466e-3 -nc 12 -awin 615-1815

Last Updated
------------
31st January 2026

Author
------
Amit Deokar (amit.deokar@students.mq.edu.au)

"""

#import required packages
import numpy as np
import os
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
import json

def parse_args():
    parser = argparse.ArgumentParser(description="Pulsar Dedispersion and Folding")
    parser.add_argument('-sf', type=str, help='raw data filename')
    parser.add_argument('-n', type=int, help='No. of spectral channels in data file')
    parser.add_argument('-nd', type=int, help='Number of Data Blocks')
    parser.add_argument('-fl', type=int, help='Lowest Frequency')
    parser.add_argument('-fh', type=int, help='Highest Frequency')
    parser.add_argument('-sb', type=int, help='Sideband(1/-1)')
    parser.add_argument('-st', type=float, help='Sampling Time')
    parser.add_argument('-dm', type=float, help='Dispersion Measure')
    parser.add_argument('-pp', type=float, help='Pulsar Period')
    parser.add_argument('-nc',type=int, help='Number of Cores')
    parser.add_argument('-var', type=str, help='variable json file')
    return vars(parser.parse_args())

def split_number(num, sn):
    quotient, remainder = divmod(num, sn)
    parts = [quotient] * sn
    for i in range(remainder):
        parts[i] += 1
    return parts

def dedisperse(raw_data, delays_samples, lp_idx,args):
    if args['sb'] == -1: raw_data = raw_data.T
    if args['sb'] == 1: raw_data = np.flipud(raw_data.T)
    temp_raw = np.zeros((raw_data.shape[0], raw_data.shape[1] + max(delays_samples)))
    temp_raw[:, -raw_data.shape[1]:] = raw_data

    dedspd_dt = np.zeros_like(temp_raw)
    for k, delay in enumerate(delays_samples):
        dedspd_dt[k] = np.roll(temp_raw[k], -delay)

    if lp_idx == 0:
        dedspd_dt = dedspd_dt[:, -raw_data.shape[1]:]

    return dedspd_dt

def dedispersion_chn_delay(args):
    channel_width = (args['fh'] - args['fl']) * 1e6 / args['n']
    frequencies_hz = (args['fl'] * 1e6) + (np.arange(args['n']) + 0.5) * channel_width
    frequencies_mhz = frequencies_hz / 1e6
    f_ref_mhz = args['fh']
    delays_ms = 4.148808e3 * args['dm'] * (1 / frequencies_mhz**2 - 1 / f_ref_mhz**2)
    delays_samples = np.round(delays_ms / args['st']).astype(int)
    delays_samples = np.flip(delays_samples)
    return delays_samples

def make_mprofs(final_arr, args, n_bins):
    tsamp_mil = args['st'] * 1e3
    
    header = f"{args['n']} {n_bins} {tsamp_mil}"
    chunk_len = 6

    # Create a list to hold all lines to be written to the file
    lines = [header]

    for chn in final_arr:
        chunks = [chn[i:i + chunk_len] for i in range(0, len(chn), chunk_len)]
        lines.extend(' '.join(map(str, chunk)) for chunk in chunks)

    # Write all lines to the file
    with open(f"{args['sf']}.mprofs", 'w') as file:
        file.write('\n'.join(lines))

def tcnt_arr(dshps, max_delay):
    dshps[1:] += max_delay
    tcnts = []
    for i in range(len(dshps)):
        if i == 0:
            tcnts.append(dshps[i] - 1)
        else:
            tcnts.append(tcnts[-1] - max_delay + dshps[i])
    return np.array(tcnts).astype(np.int64)

def process_data_block(p, args, file_path, r_size, delays_samples, tcnt_arrs, n_bins, max_delay):
    with open(file_path, mode="rb") as f:
        f.seek(np.sum(r_size[:p]).astype(np.uint))
        r_chunk = f.read(r_size[p])

        data = np.frombuffer(r_chunk, dtype=np.int16)
        raw_data = np.reshape(data, (-1, args['n']))
        raw_data1 = raw_data.copy().astype(float)
        raw_data1[raw_data1 == 0] = np.nan

        dedispered_data = dedisperse(raw_data=raw_data1, delays_samples=delays_samples, lp_idx=p, args=args)

        if p == 0:
            t = np.arange(dedispered_data.shape[1])
        else:
            tint = tcnt_arrs[p-1] + 1 - max_delay
            t = np.arange(tint, tint + dedispered_data.shape[1])
            drop_idx = np.atleast_1d(t < 0).nonzero()
            t = np.delete(t, drop_idx)
            dedispered_data = np.delete(dedispered_data, drop_idx, axis=1)

        phase = ((t * args['st']) % args['pp']) / args['pp']
        bin_index = np.round(phase * n_bins).astype(int) % n_bins

        blk_profile = np.zeros((args['n'], n_bins))
        blk_cnts_arr = np.zeros((args['n'], n_bins))
        for bn in range(n_bins):
            arr_t = np.where(bin_index == bn)[0]
            if arr_t.size > 0:
                ext_arr = dedispered_data[:, arr_t]
                nan_locs = np.where(np.isnan(ext_arr))
                ext_arr = np.nan_to_num(ext_arr, nan=0.0)
                arrs_add = np.sum(ext_arr, axis=1)
                blk_profile[:, bn] += arrs_add
                ext_arr[nan_locs[0], nan_locs[1]] = 1
                blk_cnts_arr[:, bn] += np.sum(np.where(ext_arr != 0, 1, 0), axis=1)

    return blk_profile, blk_cnts_arr

def fold_data(queue, folded_profile, bin_cnts_arr):
    while True:
        data = queue.get()
        if data is None:
            bin_cnts_arr[bin_cnts_arr == 0] = 1
            #np.save('folded_profile_mult.npy',folded_profile)
            folded_profile = folded_profile / bin_cnts_arr
            np.save('folded_Array.npy', folded_profile)
            #np.save('bin_arr_mult.npy', bin_cnts_arr)
            break
        else:
            fp, bca = data
            folded_profile += fp
            bin_cnts_arr += bca

if __name__ == "__main__":
    #print('\n**************Computing Dedispersion and Folding**************\n')
    args = parse_args()
    filepath, filename = os.path.split(args['sf'])

    fn = args['sf'].split('/')[-1]

    delays_samples = dedispersion_chn_delay(args)
    max_delay = max(delays_samples)
    n_bins = int(args['pp'] / args['st'])

    print(f"\nStoke File: {filename}")
    #print(f"Number of Channels: {args['n']}")
    #print(f"Frequency Bandwith: {args['fh']}-{args['fl']} MHz")
    #print(f"DM: {args['dm']} cm^-3 pc")
    #print(f"Sampling Time: {args['st']} sec")
    #print(f"Folding Period: {args['pp']} sec")
    #print(f"Folding Bins: {n_bins}\n")

    #Read the data and make data chunks for multiprocessing
    with open(args['sf'], mode="rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        tn = size // (args['n'] * 2)
        chunk = np.array(split_number(num=tn, sn=args['nd'])).astype(np.uint64)
        r_size = (args['n'] * chunk * 2).astype(np.uint64)


    tcnt_arrs = tcnt_arr(dshps=chunk, max_delay=max_delay)
    folded_profile = np.zeros((args['n'], n_bins))
    bin_cnts_arr = np.zeros((args['n'], n_bins))

    #Setup multiprocessing
    queue = multiprocessing.Queue()
    folding_process = multiprocessing.Process(target=fold_data, args=(queue, folded_profile, bin_cnts_arr))
    folding_process.start()

    #Execute dedispersion and folding
    with ProcessPoolExecutor(max_workers=args['nc']) as executor:
        futures = {executor.submit(process_data_block, p, args, args['sf'], r_size, delays_samples, tcnt_arrs, n_bins, max_delay): p for p in range(args['nd'])}
        for future in tqdm(as_completed(futures), total=args['nd'], desc="Processing Blocks", unit="block"):
            result = future.result()
            queue.put(result)

    print('\nFinishing Up Writing Queued Blocks . . .')

    queue.put(None)  # Signal the fold_data process to exit
    folding_process.join()

    final_arr = np.load('folded_Array.npy')

    os.remove('folded_Array.npy')
    
    make_mprofs(final_arr=final_arr, args=args, n_bins=n_bins)
    print('Saved mprof!')

    #Read temporary file for analysis window
    if os.path.exists(args['var']):
            with open(args['var'], "r") as jf:
                var_data = json.load(jf)

    if var_data['awin'] == 'full':
        final_profile = np.mean(final_arr,axis=0)
    else:
        anly_rng_arr = np.array(var_data['awin']) #np.array([list(map(int, pair.split(':'))) for pair in args['awin'].split(',')])
        anly_rng_exp = exp_array = np.concatenate([np.arange(start, end) for start, end in anly_rng_arr])
        final_profile = np.mean(np.take(final_arr,anly_rng_exp,axis=0),axis=0)

    max_bin = np.argmax(abs(final_profile))
    shift = (n_bins // 2) - max_bin
    final_cent = np.roll(final_profile, shift)
    final_cent = final_cent - np.median(final_cent)

    #Plot
    plt.plot(final_cent, '--')
    plt.xlabel('Bins')
    plt.ylabel('Counts')
    plt.title(f"folded_dsp_{filename.split('.')[-1]}")
    plt.savefig(f"{filepath}/folded_dsp_{filename.split('.')[-1]}.pdf", format='pdf', bbox_inches='tight')
    plt.close()
