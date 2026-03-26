import subprocess
import argparse
import sys
import os
import hashlib
from datetime import datetime
from pathlib import Path
import json
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

print('GMRT FRB Polarization Pipeline\n------------------------------\n')

dir_path = Path(__file__).resolve().parent

parser = argparse.ArgumentParser(description="GMRT FRB Polarization Pipeline")
parser.add_argument('-cfg', type=str, help='Configuration File')
parser.add_argument("--force", action="store_true", help="Force re-run all modules even if completed before")
args = vars(parser.parse_args())

config_path = abspath(args['cfg'])
if not config_path.exists():
    print(f"[ERROR] Config file '{config_path}' does not exist.")
    sys.exit()

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
py_path = dir_path / "pycodes/frb"

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

file = abspath(config_dict['RAW_FILE_PATH'])
cal_file = abspath(config_dict['CAL_FILE'])
nch = config_dict['NUM_OF_CHANNELS']
npol = config_dict['NUM_POLARISATION']
bt = config_dict['BYTE_TYPE']
# ndb = config_dict['NUM_OF_BLOCKS']
# ncr = config_dict['NUM_CORES']
freq_l = config_dict['FREQ_LOW(MHz)']
freq_h = config_dict['FREQ_HIGH(MHz)']
sampt = config_dict['T_SAMP']
tm_chn = config_dict['T_CHAN']
t_int = config_dict['T_INT']
dm = config_dict['DM']
iono_rm = config_dict['IONO']
#out_path = abspath(config_dict['OUT_PATH'])
sdb = config_dict['SIDE_BAND']
flp = config_dict['FLIP']
stkm = config_dict['STOKE_MODE']

print('Completed reading configuration file\n')

# if os.name == 'nt':
#     file = file.replace("\\","/")

PYTHON = "python3" if os.name != "nt" else "python"

steps =[("dd_slice", f"{PYTHON} {py_path}/dedisperse_data_slice_frb_v0.py -f {file} -n {nch} -b {bt} -st {sampt} -tchan {tm_chn} -cint {t_int} -fl {freq_l} -fh {freq_h} -dm {dm} -sb {sdb}")]

filepath = os.path.split(file)[0]
filename = os.path.split(file)[1]

file = abspath(f"{filepath}/tchan_{int(tm_chn)}/{filename.split('.')[0]}_extracted_dedispersed_tchan_{tm_chn}s.raw")
print(file)

filepath = os.path.split(file)[0]
filename = os.path.split(file)[1]
print(filepath)
print(filename)

steps.append(("make_gains", f"{PYTHON} {py_path}/make_gains_frb_v0.py -f {file} -n {nch} -b {bt} -nd 1 -nc 1"))
steps.append(("rfi_det", f"{PYTHON} {py_path}/rfi_det_mitg_frb_v1.py -gf {filepath}/all_gains_{Path(filename).stem}.txt -var {var_file}"))
steps.append(("split_iquv", f"{PYTHON} {py_path}/split_iquv_frb_v0.py -f {file} -gf {filepath}/all_gains_{Path(filename).stem}_rfi_mit.txt -n {nch} -b {bt} -nd 1 -nc 1 -var {var_file} -stk_m {stkm}"))

stk = ['stki','stkq','stku','stkv']
for stk_name in stk:
    stk_file = f"{filepath}/{filename}.{stk_name}"
    steps.append((
        f"convert_{stk_name}",
        f"{PYTHON} {py_path}/stk2mprof_frb_v0.py -sf {stk_file} -n {nch} -st {sampt} -nbins {t_int}"
    ))


steps.append(("calibrate_pass", f"{PYTHON} {py_path}/calibrate_frb_v1.py -f {filepath} -fh {freq_h} -fl {freq_l} -sb {sdb} -cal 1 -cal_file {cal_file} -var {var_file} -cnt_pa_mode 0 -plot 1 -flp {flp} -iono {iono_rm}"))

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
        subprocess.run(command.split(), check=True)

        status[name] = "done"
        with open(status_file, "w") as f:
            json.dump(status, f, indent=2)
        log(f"{name} completed successfully", log_file)

    except subprocess.CalledProcessError as err:
        log(f"[ERROR] Step {name} failed: {err}", log_file)
        sys.exit(1)

print("\nPipeline completed successfully.")
