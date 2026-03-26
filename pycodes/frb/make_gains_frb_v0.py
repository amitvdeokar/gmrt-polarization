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
    parser = argparse.ArgumentParser(description="The program averages whole data available in file, separately for each co- or cross products. User input is taken for selecting the range of channels to be flagged.")
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
        args = parse_args()
        filepath, filename = os.path.split(args['f'])
        if not os.path.exists(args['f']):
            print(f"Error: File '{args['f']}' does not exist.")
            sys.exit(1)
        
        with open(args['f'], mode="rb") as f:
            f.seek(0, SEEK_END)
            size = f.tell()
            f.seek(0)
        
        print(f"Raw File: {filename}")
        print(f"File Size: {round((size / (1024 ** 3)),2)} GB")
        print(f"Number of Channels: {args['n']}")
        print(f"File Type: {args['b']} bits")
        print(f"Number of Data Blocks: {args['nd']}\n")
        
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
        
        gains = np.mean(np.stack(results, axis=0), axis=0)
        
        gains[gains == 0] = 1e-6
        
        np.savetxt(f"{filepath}/all_gains_{Path(filename).stem}.txt", gains, fmt='%4.6f', delimiter=' ')
        print(f"\nSaved all_gains_{Path(filename).stem}.txt")
        
        print('\nDone')

    except Exception as e:
        print(f"An error occurred: {e}")
