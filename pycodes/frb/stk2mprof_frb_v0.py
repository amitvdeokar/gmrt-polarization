import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Converts Stokes File to MPROF")
    parser.add_argument('-sf', type=str, help='Stokes File Path')
    parser.add_argument('-n', type=int, help='Number of Spectral Channels')
    parser.add_argument('-st', type=float, help='Sampling Times (in sec)')
    parser.add_argument('-nbins', type=int, help='Number of Bins')
    return vars(parser.parse_args())

def make_mprofs(fp, args): 
    header = f"{args['n']} {args['nbins']} {args['st'] * 1e3}"
    chunk_len = 6

    # Create a list to hold all lines to be written to the file
    lines = [header]

    for chn in fp:
        chunks = [chn[i:i + chunk_len] for i in range(0, len(chn), chunk_len)]
        lines.extend(' '.join(map(str, chunk)) for chunk in chunks)

    # Write all lines to the file
    with open(f"{args['sf']}.mprofs", 'w') as file:
        file.write('\n'.join(lines))

if __name__ == "__main__":
    print('\n**************Stokes to MPROF**************\n')
    args = parse_args()
    data = np.fromfile(args['sf'], dtype=np.int16)
    print('File Read')
    raw_data = np.reshape(data, (-1, args['n']))

    fp = raw_data.T

    make_mprofs(fp, args)
    print('File Converted to MPROF and stored at')
