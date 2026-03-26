"""
Description
-----------
Compute and apply gain corrections to dual‑polarization correlator data.

The module reads the full dataset, calculates average power for each co‑ and
cross‑polarization product, writes a gains file with scaling factors, and
applies these corrections to equalise power levels and improve the accuracy
of Stokes parameters (particularly RR* and LL*) for circular‑polarisation
receivers such as the GMRT.

Parameters
----------
-f : str
    Path to raw data file
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

Returns
-------
A text file with 4 columns (RR*, Im(RL), LL*, Re(RL)), each row representing time average of each polarizations product for a spectral channel

Usage
-----
python make_gains_v1.py -f file.raw -n 8192 -npol 4 -b 8 -nd 500 -nc 12

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
import matplotlib.pyplot as plt
from os import SEEK_END
import argparse
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import sys
from pathlib import Path

def split_number(num, sn):
    quotient, remainder = divmod(num, sn)
    parts = [quotient] * sn
    for i in range(remainder):
        parts[i] += 1
    return parts


def parse_args():
    parser = argparse.ArgumentParser(description="The program averages raw data, separately for each self/cross products. User input is taken for selecting the range of channels to be flagged.")
    parser.add_argument('-f',type=str, help='raw data file path')
    parser.add_argument('-n',type=int,help='No. of spectral channels in data file')
    parser.add_argument('-npol',type=int,default=4,help='No. of polarization stream')
    parser.add_argument('-b', type=int, help='Bit Format')
    parser.add_argument('-nd', type=int, help='Number of Data Blocks')
    parser.add_argument('-nc',type=int, help='Number of Cores')
    args = vars(parser.parse_args())
    return args

def process_data_block(p,args,r_size,chunk, chan_dtype):
    try:
        with open(args['f'], mode="rb") as f:
            f.seek(np.sum(r_size[:p]).astype(np.uint))
            r_chunk = f.read(r_size[p])
        
        data_chunk = np.frombuffer(r_chunk, dtype=chan_dtype)
        data_chunk = data_chunk.reshape(chunk[p], args['n'])
        
        avg_rr   = np.mean(data_chunk["rr"],   axis=0)
        avg_imrl = np.mean(data_chunk["imrl"], axis=0)
        avg_ll   = np.mean(data_chunk["ll"],   axis=0)
        avg_rerl = np.mean(data_chunk["rerl"], axis=0)
        
        avg_block = np.column_stack([avg_rr, avg_imrl, avg_ll, avg_rerl])
    
        return avg_block
        
    except Exception as e:
        print(f"An error occurred while processing data block {p}: {e}")
        return None

if __name__ == "__main__":
    print('\n**************Computing Make-Gains**************\n')
    try:
        args = parse_args() #Read all the arguments
        filepath, filename = os.path.split(args['f']) #split the full path into filepath and filename
        if not os.path.exists(args['f']): # check if the raw file exists
            print(f"Error: File '{args['f']}' does not exist.")
            sys.exit(1)
        
        #open raw file and get its size
        with open(args['f'], mode="rb") as f:
            f.seek(0, SEEK_END)
            size = f.tell()
            f.seek(0)
        
        print(f"Raw File: {filename}")
        print(f"File Size: {round((size / (1024 ** 3)),2)} GB")
        print(f"Number of Channels: {args['n']}")
        print(f"File Type: {args['b']} bits")
        print(f"Number of Data Blocks: {args['nd']}\n")
        
        #Read the data as 8-bit or 16-bit and make data chunks for multiprocessing
        if args['b'] == 8:
            tn = size // (args['n'] * args['npol'])
            chunk = np.array(split_number(num=tn, sn = args['nd'])).astype(np.uint)
            r_size = (args['n'] * args['npol'] * chunk).astype(np.uint)
            chan_dtype = np.dtype([
                    ("rr",   np.uint8),
                    ("imrl", np.int8),
                    ("ll",   np.uint8),
                    ("rerl", np.int8)
                ])
        if args['b'] == 16:
            tn = size // (args['n'] * args['npol'] * 2)
            chunk = np.array(split_number(num=tn, sn = args['nd'])).astype(np.uint)
            r_size = (args['n'] * args['npol'] * chunk * 2).astype(np.uint)
            chan_dtype = np.dtype([
                    ("rr",   np.uint16),
                    ("imrl", np.int16),
                    ("ll",   np.uint16),
                    ("rerl", np.int16)
                ])
        
        #Run multiprocessing
        with ProcessPoolExecutor(max_workers=args['nc']) as executor:
            results = []
            futures = []
            with tqdm(total=args['nd'], desc="Processing Blocks", unit="block") as pbar:
                for p in range(args['nd']):
                    future = executor.submit(process_data_block, p, args, r_size, chunk, chan_dtype)
                    future.add_done_callback(lambda p: pbar.update())
                    futures.append(future)
                
                for future in futures:
                    results.append(future.result())
        
        #Get results and take their average
        gains = np.mean(np.stack(results, axis=0), axis=0)
        
        gains[gains == 0] = 1e-6
        
        #Save the gains as a text file
        np.savetxt(f"{filepath}/all_gains_{Path(filename).stem}.txt", gains, fmt='%4.6f', delimiter=' ')
        print(f"\nSaved all_gains_{Path(filename).stem}.txt")
        
        print('\nDone')

    except Exception as e:
        print(f"An error occurred: {e}")
