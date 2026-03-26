"""

Descprition
-----------
Reads a gains file and raw R‑/L‑polarization voltage products, transforms and
scales them into Stokes parameters, and writes each Stokes component to its own
output file.

Parameters
----------
-f : str
    Path to raw data file
-gf : str
    Path to all_gains file
-n : int
    Number of spectral channels
-npol : int
    Number of polarization streams (Default : 4)
-b : int
    Raw data bit type (Option : 8 or 16)
-nd : int
    Number of data blocks
-nc : int
    Number of cores
-stk_m : str
    Stokes mode (Option : C or L)

Returns
-------
4 Stokes files (Raw format; Int16)

Usage
-----
python split_iquv_v1.py -f file.raw -gf all_gains_rfi_det.txt -n 8192 -b 4 -nd 500 -nc 16 -var {var_file} -stk_m C

Last Updated
------------
31st January 2026

Author
------
Amit Deokar (amit.deokar@students.mq.edu.au)

"""

#import required packages
import numpy as np
import argparse
from tqdm import tqdm
import warnings
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import sys
import json
# warnings.filterwarnings("ignore", category=RuntimeWarning)

def i_vq(c, scaled_data, args):
    rr = scaled_data[:, c[0]].copy()
    ll = scaled_data[:, c[1]].copy()
    i = (rr + ll).astype(int)
    scaled_data[:, c[0]] = i
    if args['stk_m'].upper() == 'C':
        v = (rr - ll).astype(int)
        scaled_data[:, c[1]] = v
    if args['stk_m'].upper() == 'L':
        q = (rr - ll).astype(int)
        scaled_data[:, c[1]] = q
    
def split_number(num, sn):
    quotient, remainder = divmod(num, sn)
    parts = [quotient] * sn
    for i in range(remainder):
        parts[i] += 1
    return parts

def scale_factor(n):
    if n[0] == 0:
        self_fact_r = 0
    else:
        self_fact_r = 1 / (n[0] / Max)
    
    if n[2] == 0:
        self_fact_l = 0
    else:
        self_fact_l = 1 / (n[2] / Max)
    
    if (self_fact_l == 0) or (self_fact_r == 0) or (np.sign(self_fact_l) != np.sign(self_fact_r)):
        cross_fact = 0
    else:
        cross_fact = 2 / (np.sqrt((1 / self_fact_r) * (1 / self_fact_l)))
    arr_s = np.array([self_fact_r, cross_fact, self_fact_l, cross_fact])
    return arr_s

def parse_args():
    parser = argparse.ArgumentParser(description="This program reads raw data from file which is the co- & cross-polar voltage products of R & L-polarization signal, and converts it into the Stokes parameters and writes each Stokes parameter in separate files.")
    parser.add_argument('-f', type=str, help='raw data filename')
    parser.add_argument('-gf', type=str, help='all gain filename')
    parser.add_argument('-n', type=int, help='No. of spectral channels in data file')
    parser.add_argument('-npol', type=int, default=4, help='No. of polarization stream')
    parser.add_argument('-b', type=int, help='Bit Format')
    parser.add_argument('-nd', type=int, help='Number of Data Blocks')
    parser.add_argument('-nc',type=int, help='Number of Cores')
    parser.add_argument('-var', type=str, help='variable json file')
    parser.add_argument('-stk_m', type=str, default='C', help='Stokes mode - Circular (C) or Linear (L)')
    args = vars(parser.parse_args())
    return args

def norm_fact_det(gain_f,mrng):
    mx_lst = []
    for rng in mrng:
        mx_rng = max([max(gain_f[rng[0]:rng[1],0]),max(gain_f[rng[0]:rng[1],2])])
        mx_lst.append(mx_rng)
    return max(mx_lst)

def process_data_block(p, args, r_size, chunk, chan_dtype, sf, i_idx, u_idx, v_idx, q_idx):
    try:
        with open(args['f'], mode="rb") as f:
            f.seek(np.sum(r_size[:p]).astype(np.uint))
            r_chunk = f.read(r_size[p])

        data_chunk = np.frombuffer(r_chunk, dtype=chan_dtype)
        data_chunk = data_chunk.reshape(chunk[p], args['n']) 
        
        data_chunk = np.column_stack([
                                        data_chunk["rr"].flatten(),
                                        data_chunk["imrl"].flatten(),
                                        data_chunk["ll"].flatten(),
                                        data_chunk["rerl"].flatten(),
                                    ]).flatten().reshape(chunk[p], args['n'] * args['npol'])
        
        scaled_data = np.multiply(data_chunk, sf).astype(int)
        tp_arr = np.arange(args['n'] * args['npol'])
        even_val = tp_arr[::2]
        self_pair = [[k, j] for k, j in zip(even_val[::2], even_val[1::2])]
        for j in self_pair:
            i_vq(c=j, scaled_data=scaled_data, args=args)
        
        if args['stk_m'].upper() == 'C':
            stks = [np.take(scaled_data, idx, axis=1) for idx in [i_idx, u_idx, v_idx, q_idx]]
        if args['stk_m'].upper() == 'L':
            v_idx_l = u_idx.copy()
            q_idx_l = v_idx.copy()
            u_idx_l = q_idx.copy()
            stks = [np.take(scaled_data, idx, axis=1) for idx in [i_idx, v_idx_l, q_idx_l, u_idx_l]]
        
        results = []
        for stk in stks:
            stk = stk.astype(np.int16).flatten()
            results.append(stk)
        
        return results
    except Exception as e:
        print(f"Error occurred: {e}")

def write_data(queue, file_names, filepath):
    while True:
        data = queue.get()
        if data is None:
            break
        file_name, stk = data
        with open(f"{filepath}/{file_name}", 'ab') as stk_f:
            stk.tofile(stk_f)

if __name__ == "__main__":
    print('\n**************Computing Stoke-IQUV**************\n')
    try:
        #Read the raw file
        args = parse_args()
        with open(args['f'], mode="rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)
        
        filepath, filename = os.path.split(args['f'])
        gfpath, gfname = os.path.split(args['gf'])
        
        print(f"Raw File: {filename}")
        gain_f = np.loadtxt(args['gf'], delimiter=' ')
        print(f"Calibration File: {gfname}")
        print(f"Number of Channels: {args['n']}")
        print(f"File Type: {args['b']} bits")
        print(f"Number of Data Blocks: {args['nd']}")
        
        #Read temporary file and extract normalization factor
        if os.path.exists(args['var']):
            with open(args['var'], "r") as jf:
                var_data = json.load(jf)

        if var_data['awin'] == 'full':
            Max = max([max(gain_f[:,0]),max(gain_f[:,2])])
        else:
            mrngs = np.array(var_data['awin'])
            Max = norm_fact_det(gain_f, mrngs)
        
        print(f'Normalization Factor: {Max}')
        
        #Scale raw data using the gain file
        print('\nExecuting:\n----------')
        sf = list(map(scale_factor,gain_f));sf = np.concatenate(sf)
        
        print("\nScale factors for all spectral channels computed\n")
        
        i_idx = list(range(0, args['n'] * args['npol'], 4))
        u_idx = list(range(1, args['n'] * args['npol'], 4))
        v_idx = list(range(2, args['n'] * args['npol'], 4))
        q_idx = list(range(3, args['n'] * args['npol'], 4))
        
        file_names = [f'{filename}.stki', f'{filename}.stku', f'{filename}.stkv', f'{filename}.stkq']
        
        for i in file_names:
            if os.path.exists(f'{filepath}/{i}'):
                os.remove(f'{filepath}/{i}')
        
        #Read the data as 8-bit or 16-bit and make data chunks for multiprocessing
        if args['b'] == 8:
            tn = size // (args['n'] * args['npol'])
            chunk = np.array(split_number(num=tn, sn = args['nd'])).astype(np.uint)
            r_size = (args['n'] * args['npol'] * chunk).astype(np.uint)
            chan_dtype = np.dtype([
                                    ("rr",   np.uint8),
                                    ("imrl", np.int8),
                                    ("ll",   np.uint8),
                                    ("rerl", np.int8),
                                ])
        if args['b'] == 16:
            tn = size // (args['n'] * args['npol'] * 2)
            chunk = np.array(split_number(num=tn, sn = args['nd'])).astype(np.uint)
            r_size = (args['n'] * args['npol'] * chunk * 2).astype(np.uint)
            chan_dtype = np.dtype([
                                    ("rr",   np.uint16),
                                    ("imrl", np.int16),
                                    ("ll",   np.uint16),
                                    ("rerl", np.int16),
                                ])

        #Run multiprocessing
        queue = multiprocessing.Queue()
        write_process = multiprocessing.Process(target=write_data, args=(queue, file_names, filepath))
        write_process.start()
        
        with ProcessPoolExecutor(max_workers=args['nc']) as executor:
            for result in tqdm(executor.map(partial(process_data_block, args=args, r_size=r_size, chunk=chunk, chan_dtype=chan_dtype, sf=sf, i_idx=i_idx, u_idx=u_idx, v_idx=v_idx, q_idx=q_idx), range(args['nd'])), total=args['nd'], desc="Processing Blocks", unit="block"):
                for file_name, stk in zip(file_names, result):
                    queue.put((file_name, stk))
        
        print('\nFinishing Up Writing Queued Blocks . . .')
        
        queue.put(None)  # Signal the write process to exit
        write_process.join()
        
        print('\nDone - Stokes Files Saved\n')
    except KeyboardInterrupt:
        
        print("\nProgram Terminated by User\n")

