#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits
from astropy.table import Table, hstack
import glob
import os
from tabulate import tabulate


def make_lightcurve(list_of_files, filter, save_dir='.'):
    print(f"Making lightcurves for {galaxy} {source_id}")
    print(list_of_files)

    os.makedirs(save_dir, exist_ok=True)

    time_array = []
    flux_array = []
    flux_err_array = []
    figures = []

    for i in list_of_files:
        print(f"Reading file: {i}")
        hdul = fits.open(f"{i}")
        data = hdul[1].data
        hdul.close()
        df = pd.DataFrame(data)

        for _, row in df.iterrows():
            filter = row['FILTER'].strip()

            if filter == 'V':
                wl_eff = 5468 #angstroms
            elif filter == 'B':
                wl_eff = 4392
            elif filter == 'U':
                wl_eff = 3465
            elif filter == 'UVW1':
                wl_eff = 2600 
            elif filter == 'UVM2':
                wl_eff = 2246
            elif filter == 'UVW2':
                wl_eff = 1928 
            else:
                raise ValueError(f"Unknown filter: {filter}")

            time = (row['TSTART'] + row['TSTOP']) / 2  #picks the midpoint of exposure time
            time_array.append(time)

            flux = (row['AB_FLUX_AA']) * wl_eff
            flux_array.append(flux)

            flux_err = (row['AB_FLUX_AA_ERR'] * wl_eff)  #optional
            flux_err_array.append(flux_err)

            # lc_filename = f"light_curve_{galaxy}_{source_id}_{trimmed_name}_{filter}.png"
            # print(f"Making lightcurve for {galaxy} {source_id} {trimmed_name}")

    time_array = np.array(time_array)
    flux_array = np.array(flux_array)
    flux_err_array = np.array(flux_err_array)

    lightcurve_table = Table([time_array, flux_array, flux_err_array], names=('Time', 'Flux', 'Flux_Err'))
    print(len(time_array), len(flux_array), len(flux_err_array))
    lightcurve_table_sorted = lightcurve_table.copy()
    lightcurve_table_sorted.sort('Time')
    print(lightcurve_table_sorted)

    lightcurve_fits = f'again_log_lightcurve_{galaxy}_{source_id}_{filter}.fits'
    try:
        # print(lightcurve_table)
        lightcurve_table_sorted.write(lightcurve_fits)
        # lightcurve_table.write(lightcurve_fits)
        print(f"Wrote lightcurve file: {lightcurve_fits}")
    except Exception as e:
        print(f"Failed to write lightcurve file: {e}")

    hdul_lc = fits.open(f"{lightcurve_fits}")
    data_lc = hdul_lc[1].data
    hdul_lc.close()

    time = data_lc['Time']
    flux = data_lc['Flux']
    flux_err = data_lc['Flux_Err']


    #calculate fractional variability
    mean_flux = np.mean(flux)
    S2 = np.var(flux, ddof=1)
    mean_err2 = np.mean(flux_err**2)
    N = len(flux)

    Fvar = np.sqrt((S2 - mean_err2) / mean_flux**2) if S2 > mean_err2 else 0
    Fvar_err = (1 / (2 * N))**0.5 * S2 / (mean_flux**2 * Fvar) if Fvar > 0 else 0

    print(f"Fractional variability = {Fvar:.3f} ± {Fvar_err:.3f}")    



    fig, ax = plt.subplots()
    ax.errorbar(time, flux, yerr=flux_err, marker='s', mfc='red', mec='black', linestyle='None', capsize=5.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux (erg/s/cm²)')
    ax.set_yscale('log')
    ax.set_title(f"{galaxy} {source_id} Lightcurve ({filter})")
    ax.text(0.05, 0.05,
         f'F_var = {Fvar:.2f} ± {Fvar_err:.2f}',
         transform=plt.gca().transAxes,
         bbox=dict(facecolor='white', edgecolor='black'))

    fig_path = os.path.join(save_dir, f'lightcurve_{galaxy}_{source_id}_{filter}.png')
    fig.savefig(fig_path)
    print(f"Saved plot: {fig_path}")

    figures.append(fig) 

    return figures


top_directory = '/Users/sophianicolella/Desktop/WAVE/analysis2'
lc_folder = 'log_lightcurves_AGAIN'
os.mkdir(lc_folder)

with fits.open("reduced_reddened.fits") as hdul:
    data = hdul[1].data  # FITS_rec structure

    master_data = {}

    for name in data.names:
        col = data[name]

        # Convert to native-endian if needed
        if col.dtype.byteorder not in ('=', '|'):
            col = col.byteswap().view(col.dtype.newbyteorder('='))
        
        master_data[name] = col


reduced_df = pd.DataFrame(master_data)

unique_values = reduced_df['Name'].unique()
ngcs = [name for name in unique_values if str(name).startswith('NGC')]
print(ngcs)

for galaxy in ngcs:
    print(f'\n\n----------------- {galaxy} -----------------\n\n')
    pwd = os.getcwd()

    source_df = reduced_df[reduced_df["Name"] == galaxy]
    print(f"Number of sources in {galaxy}: {len(source_df)}")

    gal_ra = source_df['RAdeg'] #degrees
    gal_dec = source_df['DEdeg'] #degrees
    amaj = source_df['amaj'] #arcmin
    galaxy_radius = amaj / 60 #degrees

    sources = os.listdir(f'{top_directory}/{galaxy}')

    for source_id in sources:
        os.chdir(f'{top_directory}/{galaxy}/{source_id}')
        print(os.getcwd())

        save_path = f'{top_directory}/log_lightcurves_AGAIN'

        uvw1_files = glob.glob('phot_test*uw1*')
        uvw2_files = glob.glob('phot_test*uw2*')
        uvm2_files = glob.glob('phot_test*um2*')

        make_lightcurve(uvw1_files, 'UVW1', save_path)
        make_lightcurve(uvw2_files, 'UVW2', save_path)
        make_lightcurve(uvm2_files, 'UVM2', save_path)

        os.chdir(top_directory)