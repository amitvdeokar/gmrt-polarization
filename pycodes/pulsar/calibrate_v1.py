"""

Description
-----------
Reads mprofs files, applies baseline correction and a series of PA corrections
(RM, ionospheric, and system), then produces a single folded, normalized and
centred profile for each Stokes parameter, saving the results to disk.

Parameters
----------
-f : str
    Path to raw data file
-fl : float
    Lowest observational frequency (in MHz)
-fh : float
    Highest observational frequency (in MHz)
-sb : int
    Sideband (Option : 1 or -1)
-jcr : int
    PA jump correction (Optiion : 0 or 1, Default : 0)
-iono : float
    Ionospheric Correction (in rad/m^2)
-pa_offset : float
    Offset with respect to EPN Data
-cal : int
    Calibration mode (Option : 0 or 1, Default : 0)
-cal_file : str
    Calibration File (to be passed when calibration mode is set to 1)
-plot : int
    Plot Figures (Option : 0 or 1, Default : 0)
-cnt_pa_mode : int
    Compute GMRT Center PA (Option : 0 or 1)
-flp : int
    Inverts sign for stokes-q or stokes-u (0 = No flip, 1 = flip stokes-q, 2 = flip stokes-u, Default : 0)

Returns
-------
This program produces a calibration file and ASCII file consisting of final folded profile

Usage
-----
python calibrate_v1.py -f *.mprof -fh 500 -fl 100 -sb -1 -iono 2.4 -jcr 0 -pa_offset 0 -cal 1 -cal_file J0139.json -plot 0 -cnt_pa_mode 0 -flp 0

Last Updated
------------
31st January 2026

Author
------
Amit Deokar (amit.deokar@students.mq.edu.au)

"""



from __future__ import annotations

import numpy as np
from scipy.stats import chisquare
from sklearn.linear_model import RANSACRegressor
import pandas as pd
from sklearn.cluster import DBSCAN
import argparse
import pywt
from itertools import groupby
from operator import itemgetter
import subprocess
import json
import sys
import glob
import os
# from pypdf import PdfWriter
from psrqpy import QueryATNF
from pathlib import Path
from matplotlib.gridspec import GridSpec
import warnings
import astropy.units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
#from spinifex import get_rm

# Ignore all runtime warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="Calibrates the Polarization Angle")
    parser.add_argument('-f', type=str, help='MPROFs filenames')
    parser.add_argument('-fh', type=int, help='Highest Frequency')
    parser.add_argument('-fl', type=int, help='Lowest Frequency')
    parser.add_argument('-sb', type=int, help='Sideband')
    parser.add_argument('-jcr', type=int, default=0, help='pa jump correction')
    parser.add_argument('-var', type=str, help='variable json file')
    parser.add_argument('-iono', type=float, default=0, help="Ionospheric Contribution to RM in rad/m^2")
    parser.add_argument('-pa_offset', type=float,help='Offset wrt EPN')
    parser.add_argument('-cal', type=int, help='Calibration Mode (1 or 0)')
    parser.add_argument('-cal_file', type=str, help = 'Calibration File' )
    parser.add_argument('-plot', type=int, default = 0, help="Disable/Enable matplotlib plots")
    parser.add_argument('-cnt_pa_mode', type=int, help="GMRT Center PA mode")
    parser.add_argument('-flp', type=int, default=0, help="Inverts sign for stokes-q or stokes-u. 0=No flip,1=flip stokes-q, 2=flip stokes-u")
    
    args = vars(parser.parse_args())
    return args

def iono_avg_rm(start_time: str, duration: float, unit: str = "hour") -> float:
    """
    Compute the average rotation measure (RM) for a given start time and observation duration.

    Parameters
    ----------
    start_time : str
        Start time in ISO format, e.g. "2025-05-13T08:11:27.883438080" (UTC).
    duration : float
        Observation duration in hours or minutes.
    unit : str
        Either "hour" or "min" to specify duration units.

    Returns
    -------
    float
        Average RM value in rad/m^2.
    """

    # Convert duration to astropy Quantity
    if unit == "hour":
        duration_quantity = duration * u.hour
    elif unit == "min":
        duration_quantity = duration * u.min
    else:
        raise ValueError("unit must be 'hour' or 'min'")

    # Create time array (every 5 minutes by default)
    start = Time(start_time, scale="utc", format="isot")
    n_steps = int(duration_quantity.to(u.min).value / 5) + 1
    times = start + np.arange(n_steps) * 5 * u.min

    # Define source (example from your notebook)
    source = SkyCoord(ra="01h39m19.7401s", dec="+58d14m31.819s", frame="icrs")

    # GMRT location (from your notebook)
    gmrt = EarthLocation(lon=74.056 * u.deg, lat=19.093 * u.deg, height=497 * u.m)

    # Compute RM
    rm = get_rm.get_rm_from_skycoord(loc=gmrt, times=times, source=source)

    # Average RM
    avg_rm = np.mean(rm.rm)

    return avg_rm

def pa_jump_corr(PA, hd):
    PA = np.nan_to_num(PA,nan=0.0)
    x = np.arange(len(PA))
    
    data = np.vstack((x, PA)).T

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=20, min_samples=5)  # Adjust eps and min_samples based on your data
    dbscan.fit(data)
    labels = dbscan.labels_

    # Separate points based on the clusters
    clusters = {}
    for label in np.unique(labels):
        if label == -1:
            continue  # Skip noise points
        clusters[label] = data[labels == label]

    fit = []
    for label, cluster_data in clusters.items():
        # plt.scatter(cluster_data[:, 0], cluster_data[:, 1])
        if hd == 0:
            f1 = np.polyfit(cluster_data[:, 0], cluster_data[:, 1], deg=1)
        if hd == 1:
            f1 = np.polyfit(cluster_data[:, 0], cluster_data[:, 1], deg=10)
        # plt.plot(cluster_data[:, 0],np.polyval(f1, cluster_data[:, 0]))
        fit.append(f1)


    for l in range(1,len(fit)-1):
        diff = np.polyval(fit[0],clusters[0][:,0][-1]) - np.polyval(fit[l],clusters[l][:,0][0])
        clusters[l][:,1] = clusters[l][:,1] + diff
        
    df_pa = pd.DataFrame(columns=['x','PA'])
    for label, cluster_data in clusters.items():
        df_t = pd.DataFrame(cluster_data, columns=['x','PA'])
        df_t = df_t.dropna(how="all")  # Remove rows where all values are NaN
        if not df_t.empty:
            df_pa = pd.concat([df_pa, df_t], ignore_index=True)

        
    f2 = np.polyfit(df_pa['x'],df_pa['PA'],deg=2)
        
    xvals = []
    for label, cluster_data in clusters.items():
        xvals.append(cluster_data[:,0])
        
    xstd = np.arange(len(PA))
    xvals = np.concatenate(xvals)

    xunq = sorted(set(xstd).symmetric_difference(set(xvals)))


    filler = np.polyval(f2,xunq) + np.random.random()
    df_fil = pd.DataFrame({'x':xunq,
                           'PA':filler})
          
    df_pa = pd.concat([df_pa, df_fil], ignore_index=True)
    df_pa = df_pa.sort_values(by='x', ignore_index=True)

    return df_pa['PA']

def read_mprof(filepath):
    with open(filepath) as f:
        lines = f.readlines()
    
    nch, n_bins, st = lines[0].split()
    nch, n_bins, st = int(nch), int(n_bins), float(st)
    
    # Remove the first line
    lines = lines[1:]

    # Flatten the remaining lines into a single list of numbers
    flattened_numbers = [float(num) for line in lines for num in line.split()]

    arr = np.reshape(flattened_numbers, (nch,n_bins))
        
    arr_ext = arr
    
    if '.stki.' in filepath:
        return arr_ext, nch, n_bins, st
    else:
        return arr_ext

def pulse_region(signal):
    # Perform wavelet transform
    coeffs = pywt.wavedec(signal, 'db4', level=4)

    # Thresholding wavelet coefficients
    threshold = np.median(np.abs(coeffs[-1])) / 0.6745 * np.sqrt(2 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, threshold, mode='soft') for c in coeffs]

    # Reconstruct signal from thresholded coefficients
    reconstructed_signal = pywt.waverec(coeffs, 'db4')

    # Identify pulse regions
    pulse_regions = np.where(reconstructed_signal > threshold+0.005)[0]

    pulse_ranges = []
    for k, g in groupby(enumerate(pulse_regions), lambda i_x: i_x[0] - i_x[1]):
        group = list(map(itemgetter(1), g))
        pulse_ranges.append((group[0], group[-1]))

    # Identify off-pulse regions
    off_pulse_ranges = []
    if pulse_ranges[0][0] > 0:
        off_pulse_ranges.append((0, pulse_ranges[0][0] - 1))
    for i in range(len(pulse_ranges) - 1):
        off_pulse_ranges.append((pulse_ranges[i][1] + 1, pulse_ranges[i + 1][0] - 1))
    if pulse_ranges[-1][1] < len(signal) - 1:
        off_pulse_ranges.append((pulse_ranges[-1][1] + 1, len(signal) - 1))
    
    return off_pulse_ranges

def rot_qu(phi,u,q):
    rot_mat = np.array([[np.cos(phi),-np.sin(phi)],[np.sin(phi),np.cos(phi)]])
    old_qu = np.array([u,q]).reshape(-1,1)
    nu,nq = np.matmul(rot_mat, old_qu)
    return nu[0],nq[0]

def rm_correction(args, rm, nch):
    channel_width = (args['fh'] - args['fl']) / nch
    freqs = args['fl'] + args['sb'] * (np.arange(nch) + 0.5) * channel_width if args['sb'] == 1 else \
            args['fh'] + args['sb'] * (np.arange(nch) + 0.5) * channel_width
    
    lambda_m = 300 / freqs
    # lambda_ref = 300/args['fh']

    phi_rm = 2 * rm * (lambda_m**2)
    
    return phi_rm

def baseline_corr(stk_i_arr, stk_q_arr, stk_u_arr, stk_v_arr, gs):
    temp_intg_sig = np.nanmean(stk_i_arr,axis=0)
    temp_intg_sig = temp_intg_sig - np.median(temp_intg_sig)
    temp_intg_sig = temp_intg_sig/max(temp_intg_sig)
    try:
        off_pul_reg = pulse_region(signal=temp_intg_sig)
        trimmed_off_pul_reg = []
        for reg in off_pul_reg:
            diff = reg[1] - reg[0]
            trim_amt = int(diff * 0.1)
            trimmed_off_pul_reg.append((reg[0]+trim_amt,reg[1]-trim_amt))
        
        # --- 2nd plot: Baseline correction detected pulse ---
        ax2 = fig.add_subplot(gs[0])
        ax2.plot(temp_intg_sig, label='Stokes I')
        for i in range(len(trimmed_off_pul_reg)):
            ax2.axvspan(trimmed_off_pul_reg[i][0], trimmed_off_pul_reg[i][1], color='red', alpha=0.1, label='Off Pulse Region')
            if i == 0:
                ax2.legend()
        ax2.set_title("Detected OFF-Pulse Region")
        ax2.set_xlabel('Bins')
        ax2.set_ylabel('Normalized Counts')
        
        print(f'Trimmed OFF-Pulse Region: {trimmed_off_pul_reg}')
        offpulse_idx = np.concatenate([np.arange(lo, hi) for lo, hi in trimmed_off_pul_reg])
        
        stk_arr = [stk_i_arr, stk_q_arr, stk_u_arr, stk_v_arr]
        
        # fig, [ax1, ax2, ax3, ax4] = plt.subplots(4, 1, sharex=True, figsize=(10,10))
        
        for i in range(len(stk_arr)):
            for j in range(len(stk_arr[i])):
                chn_off_reg = []
                for reg in trimmed_off_pul_reg:
                    chn_off_reg.append(stk_arr[i][j][reg[0]:reg[1]+1])
                base_val_corr = np.mean(np.concatenate(chn_off_reg))
                stk_arr[i][j] = stk_arr[i][j] - base_val_corr
                
        
        temp_q_intg = np.nanmean(stk_q_arr,axis=0); temp_u_intg = np.nanmean(stk_u_arr,axis=0)
        temp_l = np.sqrt(temp_q_intg**2 + temp_u_intg**2)
        
        return stk_i_arr, stk_q_arr, stk_u_arr, stk_v_arr, np.argmax(temp_l), offpulse_idx
    except Exception as e:
        print(f"Pulse region detection failed: {e}")
        return None, None, None, None, None
    
def linear_fit(x_bins, PA, awin):
    x_bins = x_bins.reshape(-1, 1)
    phi_l = np.full(len(x_bins), np.nan)

    for wind in awin:
        x_slice = x_bins[wind[0]:wind[1]]
        y_slice = PA[wind[0]:wind[1]]
        ransac = RANSACRegressor(random_state=25).fit(x_slice, y_slice)
        fit = np.array([ransac.estimator_.coef_[0], ransac.estimator_.intercept_])
        x_vals = np.squeeze(x_slice)
        y_fit = np.polyval(fit, x_vals)
        phi_l[wind[0]:wind[1]] = y_fit

    return phi_l

def final_normalized_cent_profs(stk_i_arr_blc, Q_n_arr, U_n_arr, stk_v_arr_blc):
    prof_i = np.nanmean(stk_i_arr_blc, axis=0)
    prof_q = np.nanmean(Q_n_arr, axis=0)
    prof_u = np.nanmean(U_n_arr, axis=0)
    prof_v = np.nanmean(stk_v_arr_blc, axis=0)
    
    max_bin = np.argmax(abs(prof_i))
    pk_stki = max(prof_i)
    shift = (nbins // 2) - max_bin
    
    prof_i = np.roll(prof_i/pk_stki, shift)
    prof_q = np.roll(prof_q/pk_stki, shift)
    prof_u = np.roll(prof_u/pk_stki, shift)
    prof_v = np.roll(prof_v/pk_stki, shift)

    prof_l = np.sqrt(prof_q**2 + prof_u**2)
    
    return prof_i, prof_q, prof_u, prof_v, prof_l


def QU_transform(Qarr, Uarr, nbins, nch, phi_v, pk_l, corr, ax1, awin, args):
    # Precompute the sine and cosine of the angles in phi_v
    cos_phi = np.cos(phi_v)
    sin_phi = np.sin(phi_v)
    
    # Initialize the output arrays
    Q_n_arr = np.zeros_like(Qarr)
    U_n_arr = np.zeros_like(Uarr)
    
    # Vectorized rotation for each column in the arrays
    for col in range(nbins):
        U_col = Uarr[:, col]
        Q_col = Qarr[:, col]
        
        # Apply rotation transformation in a vectorized way
        Qn = cos_phi * Q_col + sin_phi * U_col
        Un = -sin_phi * Q_col + cos_phi * U_col
        
        # Store results directly
        Q_n_arr[:, col] = Qn
        U_n_arr[:, col] = Un
    
    PA_n = np.arctan2(np.nanmean(U_n_arr[:,pk_l-3:pk_l+4],axis=1),np.nanmean(Q_n_arr[:,pk_l-3:pk_l+4],axis=1))
    PA_n_unwrapped = pa_unwrap_deg(awin, PA_n)
    
    if args['cal'] == 0:
        if corr == 0:
            # if args['jcr'] == 1:
            #     PA_n_unwrapped = pa_jump_corr(PA=PA_n_unwrapped, hd = 0)
            ax1.plot(PA_n_unwrapped, label='PA after - RM Correction')
        if corr == 1:
            # if args['jcr'] == 1:
            #     PA_n_unwrapped = pa_jump_corr(PA=PA_n_unwrapped, hd = 1)
            ax1.plot(PA_n_unwrapped, label='PA after - System Correction')
            ax1.legend()
    if args['cal'] == 1:
        if corr == 0:
            # if args['jcr'] == 1:
            #     PA_n_unwrapped = pa_jump_corr(PA=PA_n_unwrapped, hd = 0)
            ax1.plot(PA_n_unwrapped, label='PA after - System Correction')
        if corr == 1:
            # if args['jcr'] == 1:
            #     PA_n_unwrapped = pa_jump_corr(PA=PA_n_unwrapped, hd = 1)
            ax1.plot(PA_n_unwrapped, label='PA after - RM-Iono Correction')
        if corr == 2:
            # if args['jcr'] == 1:
            #     PA_n_unwrapped = pa_jump_corr(PA=PA_n_unwrapped, hd = 1)
            ax1.plot(PA_n_unwrapped, label='PA after - RM-Astro Correction')
            ax1.legend()
        
    return Q_n_arr, U_n_arr, PA_n_unwrapped

def red_chi_sq(obs_data,fitted_data):
    chi2_stat, p_val = chisquare(f_obs=obs_data, f_exp=fitted_data)
    degrees_of_freedom = len(obs_data) - 1
    reduced_chi2 = chi2_stat / degrees_of_freedom
    return reduced_chi2

def abspath(rel_path):
    return Path(rel_path).resolve()

def save_calibration_json(sys_corr, args, filename, good_rng):

    data = {
        "PA_offset": args['pa_offset'],
        "window": good_rng,
        "sys_fit": sys_corr
    }

    with open(f"{args['f']}/{filename}_cal.json", 'w') as jf:
        json.dump(data, jf, indent=2)
        
def save_data(args,df,rm,windows):
    df.to_excel(f"{args['f']}/{filename}_final_prof.xlsx")
    
    #-----------------Record Variables-----------------------------------------
    rec = {}
    rec['RM'] = rm #Rotation Measure rad/m^2
    rec['L/I'] = max(df['L']) #Imax = 1, therefore L/I simply Lmax. Could be wrong if L and I don't peak at same time
    rec['V/I'] = max(df['V']) #Imax = 1, therefore L/I simply Lmax. Could be wrong if L and I don't peak at same time
    rec['PA'] = df.loc[df['I'].idxmax(), 'PA'] #Center PA, where I is maximum
    rec['Freqs(MHz)'] = ", ".join(f"{freqs[start]:.2f}-{freqs[end]:.2f}" for start, end in windows) #frequency windows selected analyzed

    comments = {
    'RM': "Rotation Measure rad/m^2, If PSR, then the value is sum of astroRM+ionoRM. If FRB, then its simply astroRM",
    'L/I': "Imax = 1, therefore L/I simply Lmax. Could be wrong if L and I don't peak at same time",
    'V/I': "Imax = 1, therefore V/I simply Vmax. Could be wrong if V and I don't peak at same time",
    'PA': "Center PA, where I is maximum",
    'Freqs(MHz)': "Frequency windows analyzed"
    }

    # Save to text file
    with open(f"{args['f']}/results.txt", "w") as f:
        for key in rec:
            f.write(f"{key} : {rec[key]}    # {comments[key]}\n")
        
def compute_rm(args, filename, mask, stk_i, stk_q, stk_u, freqs, offpulse_idx):
    
    frequencies_hz = freqs * 1e6
    
    stk_i_pkl = np.nanmean(stk_i[:, pk_l-3:pk_l+4], axis=1)
    stk_q_pkl = np.nanmean(stk_q[:, pk_l-3:pk_l+4], axis=1)
    stk_u_pkl = np.nanmean(stk_u[:, pk_l-3:pk_l+4], axis=1)
    
    i_err = np.nanstd(stk_i[:, offpulse_idx], axis=1)
    q_err = np.nanstd(stk_q[:, offpulse_idx], axis=1)
    u_err = np.nanstd(stk_u[:, offpulse_idx], axis=1)
    
    df_rm_inp = pd.DataFrame({'Freq_Hz': frequencies_hz,'Stokes-I':stk_i_pkl, 'Stokes-Q': stk_q_pkl,
                        'Stokes-U':stk_u_pkl, 'Stokes-I-Err': i_err,
                        'Stokes-Q-Err': q_err, 'Stokes-U-Err':u_err})
    
    np.savetxt("rmtools_input.dat", df_rm_inp.values, fmt="%.6f", delimiter=" ")
    
    try:
        command = "rmsynth1d rmtools_input.dat -S"
        rval = subprocess.run(command.split(), check=True)
    except subprocess.CalledProcessError as err:
        print(f"An error occurred: {err}")
        sys.exit()
        
    # Load the JSON output file
    with open("rmtools_input_RMsynth.json", "r") as f:
        data = json.load(f)
    
    # Extract the RM value
    rm_value = data.get("phiPeakPIfit_rm2", None)
    
    # Print the result
    print(f"Calculated RM: {rm_value} rad/m²")
    
    rmout_folder = f"{args['f']}/rmtools_output"

    os.makedirs(rmout_folder, exist_ok=True)
    
    # Get list of matching files
    mv_files = glob.glob('rmtools_input*.*')
    
    # Delete each file
    for file in mv_files:
        try:
            rm_name = os.path.basename(file)
            rm_new_name = f"{filename}_{rm_name}"
            dst = os.path.join(rmout_folder, rm_new_name)
            os.rename(file,dst)
            
            # print(f"Deleted: {file}")
        except Exception as e:
            print(f"Error moving {file}: {e}")
            
    return rm_value

def load_calibration_json(filename):
    with open(abspath(filename), 'r') as jf:
        data = json.load(jf)

    return data["PA_offset"], data['sys_fit'], data['window']

def pa_unwrap_deg(awin, PA):
    PA_unwrapped = np.full_like(PA, np.nan)
    
    for start, end in awin:
        PA_unwrapped[start:end + 1] = np.unwrap(PA[start:end + 1])

    PA_unwrapped = np.degrees(PA_unwrapped)
    return PA_unwrapped

def split_l2_by_l1(l1, l2):
    result = []
    for b_start, b_end in l2:
        fragments = []
        for a_start, a_end in l1:
            # Check if there's an overlap
            start = max(a_start, b_start)
            end = min(a_end, b_end)
            if start <= end:
                fragments.append([start, end])
        result.extend(fragments)
    return result

if __name__ == "__main__":
    print('\n**************PA Calibration**************\n')
    args = parse_args()
    if args['plot'] == 1:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    else:
        import matplotlib
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt

    plt.close('all')
    
    order = ['stki', 'stkq', 'stku', 'stkv']
    files = [glob.glob(f"{args['f']}/*{o}*.mprofs")[0] for o in order]
    filename = os.path.split(files[0])[1]
    
    if os.path.exists(args['var']):
        with open(args['var'], "r") as f:
            var_data = json.load(f)
    obj_awin = np.array(var_data['awin'])
    
    if args['cal'] == 1:
        print('Running in Calibration mode ON')
        args['pa_offset'], phi_lnr, cal_awin = load_calibration_json(args['cal_file'])
        awin = split_l2_by_l1(l1=obj_awin, l2=cal_awin)
    else:
        print('Running in Calibration mode OFF')
        psr_name = filename.split('_')[0]
        awin = obj_awin
        
        
    #     # awin = [[897,1615],[2279,2407],[3278,3357]]
    # # awin = [[251, 494], [528, 734], [772, 1060], [1093, 1496]]
    # # awin = [[357,597],[694,767],[816,877],[926,1075],[1130,1227],[1273,1355]] #13may data
    # awin = [[360, 594], #02jun data
    #   [694, 1081],
    #   [1127, 1277],
    #   [1395, 1595],
    #   [1638, 1866],
    #   [1906, 2009],
    #   [2280, 2426]]
    
    stk_i_arr_full, nch, nbins, st = read_mprof(files[0])
    stk_q_arr_full = read_mprof(files[1])
    stk_u_arr_full = read_mprof(files[2])
    stk_v_arr_full = read_mprof(files[3])
    
    print(f"MPROF Files: {filename.replace('stki','*')}")
    print(f"Number of Channels: {nch}\nNumber of Bins: {nbins}\n\nExecuting:\n----------")
   
    mask = np.zeros(nch, dtype=int)
    for start, end in awin:
        mask[start:end+1] = 1
        
    mask2d = mask[:, None]
        
    stk_i_arr_full = np.where(mask2d == 1, stk_i_arr_full, np.nan)
    stk_q_arr_full = np.where(mask2d == 1, stk_q_arr_full, np.nan)
    stk_u_arr_full = np.where(mask2d == 1, stk_u_arr_full, np.nan)
    stk_v_arr_full = np.where(mask2d == 1, stk_v_arr_full, np.nan)
    
    fig = plt.figure(figsize=(8, 12))
    gs = GridSpec(4, 1, figure=fig, height_ratios=[1, 1, 0.6, 0.6])
    
    if args['flp'] == 0:
        stk_i_arr_blc, stk_q_arr_blc, stk_u_arr_blc, stk_v_arr_blc, pk_l, offpulse_idx = baseline_corr(
            stk_i_arr_full, stk_q_arr_full, stk_u_arr_full, stk_v_arr_full, gs)
    if args['flp'] == 1:
        stk_i_arr_blc, stk_q_arr_blc, stk_u_arr_blc, stk_v_arr_blc, pk_l, offpulse_idx = baseline_corr(
            stk_i_arr_full, -1*stk_q_arr_full, stk_u_arr_full, stk_v_arr_full, gs)
    if args['flp'] == 2:
        stk_i_arr_blc, stk_q_arr_blc, stk_u_arr_blc, stk_v_arr_blc, pk_l, offpulse_idx = baseline_corr(
            stk_i_arr_full, stk_q_arr_full, -1*stk_u_arr_full, stk_v_arr_full, gs)
        
    stk_i_pkl = np.nanmean(stk_i_arr_blc[:, pk_l-3:pk_l+4], axis=1)
    stk_q_pkl = np.nanmean(stk_q_arr_blc[:, pk_l-3:pk_l+4], axis=1)
    stk_u_pkl = np.nanmean(stk_u_arr_blc[:, pk_l-3:pk_l+4], axis=1)
    
    
    PA_i = np.arctan2(stk_u_pkl, stk_q_pkl)
    
    PA_i_unwrapped = pa_unwrap_deg(awin, PA_i)

        
    ax1 = fig.add_subplot(gs[1])
    ax1.plot(PA_i_unwrapped, label='PA Original')
    ax1.set_title("PA Corrections")
    ax1.set_xlabel("Channel")
    ax1.set_ylabel("PA (deg)")
    ax1.grid(True)
    ax1.legend()
    
    
    if args['cal'] == 0: #Calibration OFF
        for corr in range(2):
            if corr == 0: #Total RM Correction
                print(f'Querying RM value for PSR-{psr_name} from ATNF ...')
                query = QueryATNF(params=['RM'], psrs=[f'{psr_name}'])
                astro_rm = query.table['RM'][0]
                print(f'ATNF Catalogue RM value for PSR-{psr_name} = {astro_rm} rad/m^2')
                comp_rm = astro_rm + args['iono']#pass the know rm for the object here (astro + iono)
                phi_rm = rm_correction(args, comp_rm, nch)
                
                Q_r_corr, U_r_corr, PA_r_corr = QU_transform(
                    stk_q_arr_blc, stk_u_arr_blc, nbins, nch, phi_rm, pk_l, corr, ax1, awin, args)
    
            elif corr == 1:
                y_fit_l = linear_fit(np.arange(nch), PA_r_corr, awin)
                phi_lnr = np.deg2rad(y_fit_l) + np.deg2rad(args['pa_offset'])
                
                Q_rl_corr, U_rl_corr, PA_rl_corr = QU_transform(
                    Q_r_corr, U_r_corr, nbins, nch, phi_lnr, pk_l, corr, ax1, awin, args)
    
    if args['cal'] == 1: # Calibration ON
        for corr in range(3):
            if corr == 0: #System Correction
                if args['cal'] == 1:
                    phi_lnr = phi_lnr * mask
            
                    Q_l_corr, U_l_corr, PA_l_corr = QU_transform(
                        stk_q_arr_blc, stk_u_arr_blc, nbins, nch, phi_lnr, pk_l, corr, ax1, awin, args)
                    
            elif corr == 1: #Ionospheric RM Correction
                iono_rm = args['iono'] #pass the known iono contribution
                phi_rm = rm_correction(args, iono_rm, nch)
                Q_l_corr, U_l_corr, PA_l_corr = QU_transform(
                    Q_l_corr, U_l_corr, nbins, nch, phi_rm, pk_l, corr, ax1, awin, args)
                
            elif corr == 2: #Astrophysical RM Correction
                channel_width = (args['fh'] - args['fl']) / nch
                freqs = args['fl'] + args['sb'] * (np.arange(nch) + 0.5) * channel_width if args['sb'] == 1 else \
                        args['fh'] + args['sb'] * (np.arange(nch) + 0.5) * channel_width
                        
                comp_rm = compute_rm(args, filename, mask, stk_i_arr_blc, Q_l_corr , U_l_corr, freqs, offpulse_idx)
                phi_rm = rm_correction(args, comp_rm, nch)
                Q_rl_corr, U_rl_corr, PA_r_corr = QU_transform(
                    Q_l_corr, U_l_corr, nbins, nch, phi_rm, pk_l, corr, ax1, awin, args)
                
    final_cent_i, final_cent_q, final_cent_u, final_cent_v, final_cent_l = final_normalized_cent_profs(
            stk_i_arr_blc, Q_rl_corr, U_rl_corr, stk_v_arr_blc)
    
    pa_corrected = np.degrees(0.5 * np.arctan2(final_cent_u, final_cent_q))
    
    if args['cal'] == 0:
        var_data['gmrt_pa'] = pa_corrected[np.argmax(final_cent_i)]
        with open(args['var'], "w") as jf:
                        json.dump(var_data, jf, indent=2)


    df_final = pd.DataFrame({'I':final_cent_i,
                             'Q':final_cent_q,
                             'U':final_cent_u,
                             'V':final_cent_v,
                             'PA':pa_corrected})
    
    
    print(f'Center PA: {pa_corrected[np.argmax(final_cent_i)]}')

    # --- 3rd plot: Folded profile (split into 2 subplots) ---
    ax3a = fig.add_subplot(gs[2])
    ax3b = fig.add_subplot(gs[3], sharex=ax3a)
    
    # PA panel
    ax3a.plot(pa_corrected, 'o', color='green', label='PA')
    ax3a.set_title("Folded Profile - PA & Stokes")
    ax3a.set_ylabel("PA (deg)")
    ax3a.grid(True)
    ax3a.legend()
    
    # Stokes panel
    ax3b.plot(final_cent_i, 'k--', label="Stokes I")
    ax3b.plot(final_cent_q, 'b--', label="Stokes Q")
    ax3b.plot(final_cent_u, 'r--', label="Stokes U")
    ax3b.plot(final_cent_v, 'g--', label="Stokes V")
    ax3b.plot(final_cent_l, 'm--', label="Linear Pol")
    ax3b.set_xlabel("Bins")
    ax3b.set_ylabel("Normalized Stokes")
    ax3b.grid(True)
    ax3b.legend()
    
    plt.tight_layout()
    
    fig.savefig(f"{args['f']}/{filename}_corrections_results.pdf", format='pdf',bbox_inches='tight')
    plt.close(fig)
    
    fig_comb, [axc1, axc2] = plt.subplots(2, 1, sharex=True, figsize=(10, 7))
    axc1.set_title("Final Combined Folded Profile")
    axc1.plot(pa_corrected, 'o', color='green', label='PA')
    axc1.set_ylabel('PA (deg)')
    axc1.legend()
    axc1.grid(True)

    axc2.plot(final_cent_i, 'k--', label='I')
    axc2.plot(final_cent_q, 'b--', label='Q')
    axc2.plot(final_cent_u, 'r--', label='U')
    axc2.plot(final_cent_v, 'g--', label='V')
    axc2.plot(final_cent_l, 'm--', label='L')
    # axc2.set_xlim(261,338)
    axc2.set_ylabel('Normalized Stokes')
    axc2.set_xlabel('Bins')
    axc2.legend()
    axc2.grid(True)
    
    plt.tight_layout()
    fig_comb.savefig(f"{args['f']}/{filename}_final_prof.pdf", format='pdf', bbox_inches='tight')
    
    #save_data(args,df_final,comp_rm,awin)
    
    if args['cal'] == 0 and args['pa_offset'] != 0:
        save_calibration_json(list(phi_lnr), args, filename, awin.tolist())
        print('Calibration File Saved')
