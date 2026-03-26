import subprocess
import argparse
import sys
import os
import numpy as np
import hashlib
from datetime import datetime
from pathlib import Path
import json
import sqlite3
import pandas as pd
from pypdf import PdfWriter
import glob

def generate_config_hash(filepath, ignore_keys={"NUM_OF_BLOCKS", "NUM_CORES"}):
    config_entries = []

    with open(filepath, "r") as f:
        for line in f:
            # Normalize whitespace and skip header/separator
            line = line.strip()
            if not line or line.startswith("-") or "Configuration" in line:
                continue
            if ":" not in line:
                continue

            # Split key and value cleanly
            key, value = map(str.strip, line.split(":", 1))
            if key not in ignore_keys:
                config_entries.append((key, value))

    # Sort by key for consistency
    config_entries.sort()

    # Build a normalized string
    normalized = "\n".join(f"{k}:{v}" for k, v in config_entries)

    # Hash it
    return hashlib.sha256(normalized.encode()).hexdigest()

def load_pulsar_profile_closest_freq(db_path, jname, target_frequency):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Find all frequencies for the given pulsar
    cur.execute('''
        SELECT id, frequency FROM pulsars
        WHERE name = ?
    ''', (jname,))
    rows = cur.fetchall()

    if not rows:
        print(f"No data found for pulsar {jname} in database.")
        conn.close()
        return None, None, None

    # Find the row with the frequency closest to the target
    closest_row = min(rows, key=lambda x: abs(x[1] - target_frequency))
    pulsar_id, closest_freq = closest_row

    print(f"Found closest frequency: {closest_freq:.3f} MHz (requested: {target_frequency} MHz)")

    # Get Stokes parameters
    df_stokes = pd.read_sql_query('''
        SELECT phase_bin, I, Q, U, V, L
        FROM stokes
        WHERE pulsar_id = ?
        ORDER BY phase_bin
    ''', conn, params=(pulsar_id,))

    # Get Polarization Angle (if available)
    df_pa = pd.read_sql_query('''
        SELECT phase_bin AS PAx, PA AS PAy
        FROM pa
        WHERE pulsar_id = ?
        ORDER BY phase_bin
    ''', conn, params=(pulsar_id,))

    conn.close()
    return df_stokes, df_pa, closest_freq

def remove(txt):
    return txt.replace(' ', '').replace('\n', '').replace('\t', '').replace('\r', '')

def convert_to_number(s):
    try:
        float_value = float(s)
        if float_value.is_integer():
            return int(float_value)  # Convert to integer if it's a whole number
        else:
            return float_value  # Keep it as a float
    except ValueError:
        return s  # Cannot convert, keep it as a string
    
def log(msg, logfile):
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    with open(logfile, "a") as f:
        f.write(f"{timestamp} {msg}\n")
    print(f"{timestamp} {msg}")
    
def abspath(rel_path):
    return Path(rel_path).resolve()

print('GMRT Pulsar Polarization Pipeline\n------------------------------\n')

dir_path = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="GMRT Pulsar Polarization Pipeline")
parser.add_argument('-cfg', type=str, help='Configuration File')
parser.add_argument("--force", action="store_true", help="Force re-run all modules even if completed before")
args = vars(parser.parse_args())

config_path = abspath(args['cfg'])
if not config_path.exists():
    print(f"[ERROR] Config file '{config_path}' does not exist.")
    sys.exit()
    
# --- Prepare Hash Folder ---
#today_str = datetime.now().strftime("%Y-%m-%d")
#run_dir = Path("pipeline_status") / f"pip_run_{today_str}"
#run_dir.mkdir(parents=True, exist_ok=True)
#
# --- Prepare Hash Folder ---
today_str = datetime.now().strftime("%Y-%m-%d")
run_dir = dir_path/f"pipeline_status/pip_run_{today_str}"
run_dir.mkdir(parents=True, exist_ok=True)

config_hash = generate_config_hash(config_path)
hash_run_dir = run_dir / config_hash
hash_run_dir.mkdir(parents=True, exist_ok=True)

hash_file = hash_run_dir / "config_hash.txt"
status_file = hash_run_dir / "pipeline_status.json"
log_file = hash_run_dir / "run.log"
var_file = hash_run_dir / "variables.json"
db_file = dir_path / "epn_database/epn_pulsar_profiles.db"
py_path = dir_path / "pycodes/pulsar"

config_dict = {}
with open(args['cfg'],'r') as config:
    config_cnts = config.read().rstrip().splitlines()[3:]
    for i in config_cnts:
        cnt_spt = remove(i).split(':')
        if cnt_spt[0] == 'FREQ_RANGE(MHz)':
            f_h,f_l = cnt_spt[1].split('-')
            config_dict.update({'FREQ_LOW(MHz)':convert_to_number(f_l),'FREQ_HIGH(MHz)':convert_to_number(f_h)})
        else:
            config_dict.update({cnt_spt[0]:convert_to_number(cnt_spt[1])})

#py_path = abspath(config_dict['PY_CODES'])
file = abspath(config_dict['RAW_FILE_PATH'])
nch = config_dict['NUM_OF_CHANNELS']
npol = config_dict['NUM_POLARISATION']
bt = config_dict['BYTE_TYPE']
ndb = config_dict['NUM_OF_BLOCKS']
ncr = config_dict['NUM_CORES']
freq_l = config_dict['FREQ_LOW(MHz)']
freq_h = config_dict['FREQ_HIGH(MHz)']
sampt = config_dict['T_SAMP']
dm = config_dict['DM']
iono_rm = config_dict['IONO']
pul_prd = config_dict['PULSAR_PERIOD']
#out_path = abspath(config_dict['OUT_PATH'])
sdb = config_dict['SIDE_BAND']
flp = config_dict['FLIP']
stkm = config_dict['STOKE_MODE']

print('Completed reading configuration file\n')

if os.name == 'nt':
    file = file.replace("\\","/")

filepath = os.path.split(file)[0]
filename = os.path.split(file)[1]

PYTHON = "python3" if os.name != "nt" else "python"

steps = [
    ("make_gains", f"{PYTHON} {py_path}/make_gains_v1.py -f {file} -n {nch} -b {bt} -nd {ndb} -nc {ncr}"),
    ("rfi_det", f"{PYTHON} {py_path}/rfi_det_mitg_v1.py -gf {filepath}/all_gains_{Path(filename).stem}.txt -var {var_file}"),
    ("split_iquv", f"{PYTHON} {py_path}/split_iquv_v1.py -f {file} -gf {filepath}/all_gains_{Path(filename).stem}_rfi_mit.txt -n {nch} -b {bt} -nd {ndb} -nc {ncr} -var {var_file} -stk_m {stkm}")
]

stk = ['stki','stkq','stku','stkv']
for stk_name in stk:
    stk_file = f"{filepath}/{filename}.{stk_name}"
    steps.append((
        f"fold_{stk_name}",
        f"{PYTHON} {py_path}/psr_dsp_fold_v1.py -sf {stk_file} -n {nch} -nd 100 -fl {freq_l} -fh {freq_h} -st {sampt} -dm {dm} -pp {pul_prd} -sb {sdb} -nc {ncr} -var {var_file}"
    ))

freq_m = (freq_h + freq_l) / 2
stokes_df, pa_df, actual_freq = load_pulsar_profile_closest_freq(db_file, jname=filename.split('_')[0], target_frequency=freq_m)
# steps.append(("epn_scrap", f"{PYTHON} {py_path}/epn_scrapper_v1.py -psr {filename.split('_')[0]} -freq_val {freq_m}"))

max_i = stokes_df.loc[stokes_df['I'].idxmax(), 'phase_bin']
#epn_cent_pa = -4.391742880280664 
epn_cent_pa = pa_df['PAy'].iloc[(pa_df['PAx'] - max_i).abs().argmin()]
print(f'\nEPN Center PA : {epn_cent_pa} deg')

steps.append(("calibrate_pass1", f"{PYTHON} {py_path}/calibrate_v1.py -f {filepath} -fh {freq_h} -fl {freq_l} -sb {sdb} -pa_offset 0 -cal 0 -var {var_file} -plot 1 -cnt_pa_mode 1 -flp {flp} -iono {iono_rm}"))

steps.append(("calibrate_pass2", f"{PYTHON} {py_path}/calibrate_v1.py -f {filepath} -fh {freq_h} -fl {freq_l} -sb {sdb} -cal 0 -var {var_file} -cnt_pa_mode 0 -plot 1 -flp {flp} -iono {iono_rm}"))

# Load or initialize status
if args["force"]:
    status = {step[0]: "pending" for step in steps}
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    with open(var_file, "w") as vf:
        json.dump({}, vf)
    log("Force flag set: Re-running all steps from scratch", log_file)
elif not hash_file.exists():
    hash_file.write_text(config_hash)
    status = {step[0]: "pending" for step in steps}
    with open(status_file, "w") as f:
        json.dump(status, f, indent=2)
    log("Initialized new run with fresh status", log_file)
else:
    with open(status_file) as f:
        status = json.load(f)
    log("Resuming previous run", log_file)
    
# --- Execute Steps ---

for name, command in steps:
    if status.get(name) == "done":
        log(f"Skipping {name} (already done)", log_file)
        continue

    log(f"Running: {name}", log_file)
    try:
        # Adjust commands dynamically
        #if name == "calibrate_pass1":
        #    print('Computing GMRT Center PA')
        if name == "calibrate_pass2":
            # Compute offset from tmp_path
            with open(var_file, "r") as f:
                var_data = json.load(f)
            if var_data.get('offset') is None:
                gmrt_cent_pa = var_data.get("gmrt_pa")
                offset = -2 * (epn_cent_pa - gmrt_cent_pa)
                var_data['offset'] = offset
                with open(var_file, "w") as f:
                    json.dump(var_data, f, indent=2)
            else:
                offset = var_data['offset']
            print(f'Computed PA Offset : {offset}')
            command += f" -pa_offset {offset}"
            
        if name == 'calibrate_pass1':
           pdfs_path = glob.glob(f'{filepath}/folded_dsp_stk*.pdf')
           merger = PdfWriter()
           for pdf in pdfs_path:
               merger.append(pdf)
           merger.write(f'{filepath}/folded_dedispersed_stokes.pdf')
           merger.close()
           for pdf in pdfs_path:
               os.remove(pdf)
        #    rval = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        #    stdout, stderr = rval.communicate()
        #   if rval.returncode != 0:
        #        raise subprocess.CalledProcessError(rval.returncode, command, output=stdout, stderr=stderr)
        #else:
        subprocess.run(command.split(), check=True)

        status[name] = "done"
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        log(f"{name} completed successfully", log_file)

    except subprocess.CalledProcessError as err:
        log(f"[ERROR] Step {name} failed: {err}", log_file)
        sys.exit(1)

print("\nPipeline completed successfully.")
