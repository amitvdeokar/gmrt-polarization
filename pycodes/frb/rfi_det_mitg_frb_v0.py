import numpy as np
import matplotlib.pyplot as plt
import argparse
import pandas as pd
from astropy.convolution import convolve, Box1DKernel
from scipy.signal import medfilt
#from scipy.signal import savgol_filter
from sklearn.linear_model import HuberRegressor
import os
from pathlib import Path
import json

def parse_args():
    parser = argparse.ArgumentParser(description="RFI Channel Detection and Mitigation")
    parser.add_argument('-gf', type=str, help='Gains file path')
    parser.add_argument('-var', type=str, help='variable json file')
    return vars(parser.parse_args())

def on_click(event):
    global click_count, x_coords, colors, range_count, plot_text

    if event.inaxes and event.button == 3:  # Right-click to select
        x_coords.append(event.xdata)
        click_count += 1

        # Update the plot text dynamically
        if click_count % 2 == 1:
            plot_text.set_text("Make another selection to complete the range.")
        else:
            # Sort the range to ensure the smaller value is saved first
            range_pair = sorted([x_coords[-2], x_coords[-1]])

            # Plot vertical lines for the range
            color = colors[range_count % len(colors)]  # Cycle through colors
            plt.axvline(x=range_pair[0], color=color, linestyle='--')
            plt.axvline(x=range_pair[1], color=color, linestyle='--')

            # ---- JSON Save Logic ----
            json_file = args['var']
            new_range = [int(range_pair[0]), int(range_pair[1])]

            # Check if JSON file exists and load existing data
            if os.path.exists(json_file):
                with open(json_file, "r") as jf:
                    try:
                        var_data = json.load(jf)
                    except json.JSONDecodeError:
                        var_data = {}  # Empty/corrupt file fallback
            else:
                var_data = {}  # Initialize if file doesn't exist
            var_data.setdefault("awin", [])
            # Append new range to "awin" list
            var_data["awin"].append(new_range)

            # Write updated JSON back to file
            with open(json_file, "w") as jf:
                json.dump(var_data, jf, indent=2)
            # --------------------------

            # Update the plot text
            plot_text.set_text("Right-click to start a new range.")

            # Increment the range counter
            range_count += 1

        plt.draw()  # Redraw the plot

        # Print to console
        if click_count % 2 == 0:
            print(f"Range Selected: {np.array(range_pair).astype(int)}")

def RFI_detect(df):
    df_o = df.copy()  # Original data with RFI
    df = abs(df)  # Processed data

    df2 = pd.DataFrame()
    # Calculate sum along rows
    x_points = np.array(df.sum(axis=1))

    # Median filtering
    y_points = medfilt(x_points, 11)

    # Prepare X data for regression
    X = np.array([[i] for i in x_points])

    # Calculate residuals and outlier rejection threshold
    huber = HuberRegressor().fit(X, y_points)
    residuals = y_points - huber.predict(X)
    mad = np.median(np.abs(residuals - np.mean(residuals)))
    k = 0.01  # Adjust this scaling factor according to your needs
    dx = k * mad

    # Plotting
    plotline_X = np.arange(y_points.min(), y_points.max()).reshape(-1, 1)
    fit_df = pd.DataFrame(index=plotline_X.flatten())
    fit_df["huber_regression"] = huber.predict(plotline_X)
    # l1 = fit_df['huber_regression'] + dx
    # l2 = fit_df['huber_regression'] - dx

    # Identify outliers
    x_outliers = []
    y_outliers = []
    for i in range(len(x_points)):
        if y_points[i] > x_points[i] and y_points[i] > dx + x_points[i]:
            x_outliers.append(x_points[i])
            y_outliers.append(y_points[i])
        elif y_points[i] < x_points[i] and y_points[i] < x_points[i] - dx:
            x_outliers.append(x_points[i])
            y_outliers.append(y_points[i])

    indices = np.where(np.in1d(x_points, x_outliers))[0]

    df2 = pd.DataFrame()
    for i in range(4):
        k = np.array(df_o[i])
        smoothed_signal_med = convolve(k, Box1DKernel(13))
        t = np.take(smoothed_signal_med,indices)
        np.put(k, indices, t)
        df2[i] = k

    return df2, indices


def plot(gains):
    ax.plot(gains[3:,0],'r',label='RR*')
    ax.plot(gains[3:,1],'b',label='Im(RL*)')
    ax.plot(gains[3:,2],'g',label='LL*')
    ax.plot(gains[3:,3],'y',label='Re(RL*)')
    ax.set_xlabel('Spectral Channel No.')
    ax.set_ylabel('Self/Cros voltage products')
    ax.set_title("RFI Mitigated Gains")
    plt.legend()
    
    # Connect mouse events
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    plt.show()

if __name__ == "__main__":
    print('\n**************RFI Channel Detection and Mitigation**************\n')
    args = parse_args()
    gfpath, gfname = os.path.split(args['gf'])
    
    # Read the data
    df = pd.read_csv(args['gf'], sep='\s+', header=None)
    
    for i in range(1):
        df2, indices = RFI_detect(df)
        df = df2
    gains = df.to_numpy()
    
    
    # Initialize variables
    click_count = 0
    x_coords = []
    colors = ['red', 'blue', 'green', 'orange', 'purple']  # Cycle through these colors
    range_count = 0
    

    fig, ax = plt.subplots()   
    
    plot_text = ax.text(0.5, 0.9, "Right-click to select RFI-free ranges.", transform=ax.transAxes, fontsize=10, color='black', ha='right', va='bottom')
    
    print('\nPlotting Gains . . .')
    plot(gains)
    plt.scatter(indices,np.zeros(len(indices))) 
    
    if click_count == 0:
        json_file = args['var']

        # Check if JSON file exists and load existing data
        if os.path.exists(json_file):
            with open(json_file, "r") as jf:
                try:
                    var_data = json.load(jf)
                except json.JSONDecodeError:
                    var_data = {}  # Empty/corrupt file fallback
        else:
            var_data = {}  # Initialize if file doesn't exist
        var_data.setdefault("awin", 'full')

        # Write updated JSON back to file
        with open(json_file, "w") as jf:
            json.dump(var_data, jf, indent=2)
    
    np.savetxt(f"{gfpath}/{Path(gfname).stem}_rfi_mit.txt", gains, fmt='%4.6f', delimiter=' ')
    print(f"\nSaved {Path(gfname).stem}_rfi_mit.txt")
