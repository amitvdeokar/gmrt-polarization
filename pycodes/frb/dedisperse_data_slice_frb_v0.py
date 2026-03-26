import numpy as np
import matplotlib.pyplot as plt
import os
# from scipy.ndimage import median_filter 
import argparse
import shutil

def dedisperse(raw_data, delays_samples, args):
    raw_data = raw_data.T
    temp_raw = np.zeros((raw_data.shape[0],raw_data.shape[1]+max(delays_samples)))
    temp_raw[:,-raw_data.shape[1]:] = raw_data
    
    if args['b'] == 8:
        dedspd_dt = np.zeros_like(temp_raw).astype(np.uint8)
    else:
        dedspd_dt = np.zeros_like(temp_raw).astype(np.uint16)
    
    for k, delay in enumerate(delays_samples):
        dedspd_dt[k] = np.roll(temp_raw[k], -delay)
        
    dedspd_dt = dedspd_dt[:,max_delay:-max_delay]
    
    return dedspd_dt

def dedispersion_chn_delay(args):
    channel_width = (args['fh'] - args['fl']) / args['n']  # in MHz
    if args['sb']  == 1:
        frequencies_mhz = (args['fl']) + args['sb'] * (np.arange(args['n']) + 0.5) * channel_width
    if args['sb'] == -1:
        frequencies_mhz = (args['fh']) + args['sb'] * (np.arange(args['n']) + 0.5) * channel_width
        
    # Reference frequency in MHz (the highest frequency in the band)
    f_ref_mhz = args['fh']

    # Calculate delays for each channel in milliseconds
    delays_ms = 4.148808e3 * args['dm'] * (1 / frequencies_mhz**2 - 1 / f_ref_mhz**2)  # delays in sec

    # Convert delays to number of samples
    delays_samples = np.round(delays_ms / args['st']).astype(int)
    # delays_samples = np.flip(delays_samples)
    
    return delays_samples

def prepare_out_folder(args, out_folder):
    target_folder = os.path.join(os.path.dirname(args['f']), out_folder)

    if os.path.exists(target_folder):
        # Empty the folder
        for filename in os.listdir(target_folder):
            file_path = os.path.join(target_folder, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # remove file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # remove subfolder
    else:
        # Create the folder (and parents if needed)
        os.makedirs(target_folder, exist_ok=True)

    return target_folder


def parse_args():
    parser = argparse.ArgumentParser(description="This program reads raw polar data and extracts given time channels")
    parser.add_argument('-f', type=str, help='raw data filename')
    parser.add_argument('-n', type=int, help='No. of spectral channels in data file')
    parser.add_argument('-npol', type=int, default=4, help='No. of polarization stream')
    parser.add_argument('-b', type=int, help='Bit Format')
    parser.add_argument('-st', type=float, help='Sampling Time')
    parser.add_argument('-tchan',type=float, help='Channel Time')
    parser.add_argument('-cint', type=int, help='Slicing Interval')
    parser.add_argument('-fl', type=float, help='Lowest Frequency (in MHz)')
    parser.add_argument('-fh', type=float, help='Highest Frequency (in MHz)')
    parser.add_argument('-dm', type=float, help='Dispersion Measure')
    parser.add_argument('-sb', type=int, help='Sideband')
    args = vars(parser.parse_args())
    return args

if __name__ == "__main__":
    print('\n**************Extracting Time Channels**************\n')
    try:
        args = parse_args()
        fn = args['f'].split('/')[-1]
        delays_samples = dedispersion_chn_delay(args)
        max_delay = max(delays_samples)
        
        print(f"Raw File: {fn}")
        print(f"Number of Channels: {args['n']}")
        print(f"File Type: {args['b']} bits")
        print(f"Sampling Time: {args['st']} sec")
        print(f"Slicing Interval Around the Pulse: {args['cint']} channels")
        print(f"Tchan: {args['tchan']} sec")
        cent_chn = int(args['tchan']/args['st'])
        print(f"Pulse Center Channel: {cent_chn}")
        chn_i = np.int64(cent_chn - (args['cint']//2))
        chn_e = np.int64(cent_chn + (args['cint']//2) + max_delay)
        print(f"Slicing Ends: {chn_i} to {chn_e - max_delay}")
    
        print('\nExecuting\n---------')
        
        if args['b'] == 8:
            byt_i = (chn_i * args['n'] * args['npol'])
            byt_e = (chn_e * args['n'] * args['npol'])
            read_amt = (byt_e - byt_i)
        if args['b'] == 16:
            byt_i = (chn_i * args['n'] * args['npol'] * 2)
            byt_e = (chn_e * args['n'] * args['npol'] * 2)
            read_amt = (byt_e - byt_i)

        with open(args['f'], mode="rb") as f:
            f.seek(byt_i)
            r_chunk = f.read(read_amt)
            f.close()

        data_chunk = np.frombuffer(r_chunk, dtype=np.uint16 if args['b'] == 16 else np.uint8)
        data_chunk = np.reshape(data_chunk, (chn_e-chn_i, args['n'] * args['npol']))

        print('\nData Read')

        rr_idx = list(range(0, args['n'] * args['npol'], 4))
        im_idx = list(range(1, args['n'] * args['npol'], 4))
        ll_idx = list(range(2, args['n'] * args['npol'], 4))
        re_idx = list(range(3, args['n'] * args['npol'], 4))

        cross_prdt = [np.take(data_chunk, idx, axis=1) for idx in [rr_idx, im_idx, ll_idx, re_idx]]
        print('Data Split into individual polarization components\n')

        dd_prdt_lst = []
        for i in range(args['npol']):
            dedispersed_data = dedisperse(raw_data=cross_prdt[i],delays_samples=delays_samples, args=args)
            # dedispersed_data_med_filt = median_filter(dedispersed_data.T,size=3)
	    
            print(f'Polarization Component {i} dedispersed')
            if i == 0:
                rr = dedispersed_data
            if i == 1:
                im = dedispersed_data
            if i == 2:
                ll = dedispersed_data
            if i == 3:
                re = dedispersed_data
            dd_prdt_lst.append(dedispersed_data.T)

        dd_prdt_stacked = np.stack(dd_prdt_lst,axis=-1)

        dd_prdt_final = dd_prdt_stacked.reshape(args['cint'],-1)
        
        out_folder = f"tchan_{int(args['tchan'])}"
        tchan_fold = prepare_out_folder(args, out_folder)
        out_file = f"{tchan_fold}/{fn.split('.')[0]}_extracted_dedispersed_tchan_{args['tchan']}s.raw"

        with open(out_file, 'wb') as ext:
            dd_prdt_final.tofile(ext)
	    
        plt.imshow(rr,vmin=0.5 * np.median(rr), vmax=1.5 * np.median(rr), cmap="binary")
        plt.xlabel('Time')
        plt.ylabel('Channel')
        plt.title(f"Extracted and Dedispered Channels - TCHAN {args['tchan']} sec")
        plt.show() 
        print('\nDone')

        np.save(f'{tchan_fold}/rr.npy',rr);np.save(f'{tchan_fold}/im.npy',im);np.save(f'{tchan_fold}/ll.npy',ll);np.save(f'{tchan_fold}/re.npy',re)

    except Exception as e:
        print(f"An error occurred: {e}")
