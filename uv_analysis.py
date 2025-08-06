#!/usr/bin/env python
# This is a script meant to automate the analysis of a catalog of ULX sources looking for corresponding bright UV emission
# Before running, make sure HEASoft is downloaded, henv is activated, and the CALDB variable is set and sourced

# Change before running: {top_directory} and {save_path}

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
import glob
import os
import argparse
import subprocess
from astroquery.mast import Observations
from tabulate import tabulate
import time


#---------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Specify catalog reduction parameters and UV data download process.')
parser.add_argument("--num-target-download", type=float, default=5, help='The number of targets for which to download observation files.')
parser.add_argument("--num-obs-download", type=float, default=11, help='The number of observation files to download per target (Input-1).')
args = parser.parse_args()


#---------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
#---------------------------------------------------------------------------------------------

def get_radius(amaj, dist): #dist in Mpc, amaj in arcmin
    r = (dist * 1e6) * ((np.pi)/180*60) * amaj 
    return r
    #used to calculate the radius of the galaxy in parsecs


def get_uv_info(file, dist):
    #the function used to get the absolute magnitude with reddening correction for a certain source
    #the file is the fits table for that specific source
    #now also gets mass estimate from absolute magnitude

    dist_pc = dist * 1e6 #converts from Mpc to pc!!!!!

    app_mags = []
    abs_mags = []
    dists = []
    dist_mods = []
    f_ratios = []
    sfr_masses_msun = []
    filters = []
    ebv_vals = []
    reddening = []

    
    hdul = astropy.io.fits.open(file)
    data = hdul[1].data 
    hdul.close()

    df = pd.DataFrame(data)

    for _, row in df.iterrows():
        app_mag = row["AB_MAG"]
        filter = row['FILTER']
        ebv_val = row['E_B_V_SFD_mean']

        print(f'Distance: {dist} Mpc or {dist_pc} pc')
        print(f'Apparent Magnitude: {app_mag}')
        print(f'Filter: {filter}')
        print(f'E(B-V) Value: {ebv_val}')

        #reddening coefficients for each filter in UVOT
        if filter == 'UVW1':
            ext_coef = 4.91
        elif filter == 'UVW2':
            ext_coef = 5.60
        elif filter == 'UVM2':
            ext_coef = 6.99
        elif filter == 'U':
            ext_coef = 4.13
        elif filter == 'V':
            ext_coef = 3.41
        elif filter == 'B':
            ext_coef = 2.57
        else: 
            raise ValueError(f"Unknown filter: {filter}")

        A_a = ext_coef * ebv_val


        print(f'A_a: {A_a}')
            
        #E(B-V) for ngc1365: 0.0205
        #E(B-V) for ngc5194: 0.0359


        abs_mag = (app_mag - 5 * np.log10(dist_pc/10)) - A_a #DISTANCE MUST BE IN PARSECS
        dist_mod = app_mag - abs_mag
        print(f'Absolute magnitude: {abs_mag}')
        print(f'Distance Modulus: {dist_mod}')

        o5v_mag = -8.00
        o5v_mass = 30 * 2*10**30 #kg

        f_ratio = 10**((o5v_mag - abs_mag) / 2.5)
        print(f'Number of O5V stars needed to reach an absolute magnitude of {abs_mag}: {f_ratio}')

        sfr_mass = o5v_mass * f_ratio
        sfr_mass_msun = sfr_mass / (2*10**30)
        print(f'Approx. mass of an SFR with abs. mag. {abs_mag}: {sfr_mass} kg or {sfr_mass_msun} solar masses')
        
        app_mags.append(app_mag)
        abs_mags.append(abs_mag)
        dists.append(dist_pc)
        dist_mods.append(dist_mod)
        f_ratios.append(f_ratio)
        sfr_masses_msun.append(sfr_mass_msun)
        filters.append(filter)
        ebv_vals.append(ebv_val)
        reddening.append(A_a)

    uv_data_table = Table([app_mags, abs_mags, dists, dist_mods, f_ratios, sfr_masses_msun, filters, ebv_vals, reddening], 
                          names=('App Mag', 'Abs Mag', 'Dist (pc)', 'Dist Mod', 'No. O5V', 'SFR Approx. Mass', 'Filter', 'E(B-V)', 'Red Factor (A_a)'))

    return uv_data_table
        

def get_luminosity(df, flux, dist):
    filter = df['FILTER'].iloc[0].strip()

    if filter == 'V':
        wl_eff = 5468
    elif filter == 'U':
        wl_eff = 3465
    elif filter == 'UVW1':
        wl_eff = 2600 #angstroms
    elif filter == 'UVM2':
        wl_eff = 2246
    elif filter == 'UVW2':
        wl_eff = 1928 #angstroms
    else:
        raise ValueError(f"Unknown filter: {filter}")

    d = dist * 3.086e18 #dist in pc, convert to centimeters
    L = (flux * wl_eff) * (d**2) * 4 * np.pi #make sure the flux density is in units of erg/s/cm^2/A NOT JANSKYS
    return L #in erg/s


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

    lightcurve_fits = f'lightcurve_{galaxy}_{source_id}_{filter}.fits'
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

    fig, ax = plt.subplots()
    ax.errorbar(time_array, flux_array, yerr=flux_err_array, marker='s', mfc='red', mec='black', linestyle='None', capsize=5.0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux (erg/s/cm²)')
    ax.set_title(f"{galaxy} {source_id} Lightcurve ({filter})")

    fig_path = os.path.join(save_dir, f'lightcurve_{galaxy}_{source_id}_{filter}.png')
    fig.savefig(fig_path)
    print(f"Saved plot: {fig_path}")

    figures.append(fig) 

    return figures


#---------------------------------------------------------------------------------------------
# Step 2: Analyzing UV Data
#---------------------------------------------------------------------------------------------

reduced_catalog = 'reduced_catalog.fits'

# change the file here for the catalog of sources
# need to append the reddening data first:
    # go to this website: https://irsa.ipac.caltech.edu/applications/DUST/
    # upload your table of sources (fits)
    # download E(B-V) + Extinction table 
    # the following will convert to fits and combine the files

extinction_tbl = 'extinction.tbl' #name of downloaded file
table = Table.read(extinction_tbl, format='ascii')
# for col in table.colnames:
#     print(f"{col}: {table[col].unit}")
for col in table.colnames:
    if table[col].unit == 'mags':
        table[col].unit = 'mag'  # FITS-compatible unit
table.write('extinction.fits', format='fits', overwrite=True)

extinction = 'extinction.fits'

table1 = Table.read(reduced_catalog)
table2 = Table.read(extinction)
combined_table = hstack([table1, table2])
combined_table.write('reduced_reddened.fits', format='fits')

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

print(f'Number of unique host galaxies: {len(ngcs)} galaxies')

top_directory = '/Users/sophianicolella/Desktop/WAVE/analysis2'

for galaxy in ngcs:
    print(f'\n\n----------------- {galaxy} -----------------\n\n')
    pwd = os.getcwd()

    source_df = reduced_df[reduced_df["Name"] == galaxy]
    print(f"Number of sources in {galaxy}: {len(source_df)}")

    gal_ra = source_df['RAdeg'] #degrees
    gal_dec = source_df['DEdeg'] #degrees
    amaj = source_df['amaj'] #arcmin
    galaxy_radius = amaj / 60 #degrees
    back_ra = gal_ra + galaxy_radius + (5 / 3600) #degrees 
    # when there is more than one source, may have to move this into the source_df loop?
    # but the galaxy ra and dec should be the same for each source within a host
    # might just need to specify with .iloc[0]
    try:
        os.chdir(top_directory)
        os.makedirs(galaxy)
    except Exception as e:
        print(f"failed to make directory for galaxy {galaxy}")

    os.chdir(f'{top_directory}/{galaxy}')
    print(os.getcwd())

    for _, row in source_df.iterrows():
        start = time.time()
        os.chdir(f"{top_directory}/{galaxy}")

        source_id = row['2SXPSID'] #use the swift ID since we are going to retreive UVOT data

        if source_id != 0:
            print(f'\n\n----------------- Swift source ID: {source_id} -----------------\n\n')

        else:
            print(f"Source ID was zero. Skipping.")

            dir_name = '0'
            while os.path.exists(dir_name):
                dir_name += '0'  # append another '0'

            os.mkdir(dir_name)
            print(f"Created placeholder directory: {dir_name}")

            if len(source_df) > 1:
                os.chdir(f"{top_directory}/{galaxy}")
                continue
            else: 
                os.chdir(f"{top_directory}")
                continue

        os.makedirs(str(source_id))
        print(os.getcwd())
        os.chdir(f'{top_directory}/{galaxy}/{source_id}')
        print(os.getcwd())

        dist = row['Dist'] #Mpc
        print(f"Distance: {dist} Mpc")
        amaj_indiv = row['amaj'] #arcmin
        radius = get_radius(amaj_indiv, dist) #pc
        gal_ra_indiv = row['RAdeg'] #degrees
        gal_dec_indiv = row['DEdeg'] #degrees
        src_ra = row['RASdeg'] #degrees
        src_dec = row['DESdeg'] #degrees

        print(f"Host galaxy RA: {gal_ra_indiv}")
        print(f"Host galaxy DE: {gal_dec_indiv}")

        # queries MAST archive for source data
        print("Retrieving UVOT Observation data...")
        # obs_table = Observations.query_region(f"{src_ra}, {src_dec}")
        try:
            obs_table = Observations.query_region(f"{src_ra} {src_dec}")
        except Exception as e:
            print(f"Astroquery failed for {galaxy}-{source_id}: {e}")

        obs_df = obs_table.to_pandas()

        orig_obs_id = obs_df['obs_id']

        obs_df['target_id'] = obs_df['obs_id'].astype(str).str[:-3]
        target_id = obs_df['target_id']

        obs_src_ra = obs_df['s_ra']
        obs_src_dec = obs_df['s_dec']

        tolerance = 1 / 3600  # 1 arcsecond in degrees
        target_df = obs_df[
            (abs(obs_df['s_ra'] - src_ra) < tolerance) & 
            (abs(obs_df['s_dec'] - src_dec) < tolerance)
            ]

        target_ids = obs_df['target_id'].to_numpy()
        obs_nums = obs_df['obs_id'].to_numpy()

        print(f"Total number of observations for source {source_id}: {len(obs_nums)}")


        # temporary fix until i figure out how to not blow up my computer with files
        # sorry macbook i love you
        unique_ids = np.unique(target_ids)
        if len(unique_ids) >= 10:
            unique_ids = unique_ids[:args.num_target_download]


        print(f"Downloading {len(unique_ids)} observations for {galaxy} {source_id}")

        # downloads image data for each obs_id and every observation (1-10) in a certain galaxy
        os.chdir(f'{top_directory}/{galaxy}/{source_id}')
        print(os.getcwd())

        obs_list = list(range(1,args.num_obs_download))
        for target_id in unique_ids:
            # target_df = obs_df[obs_df['target_id'] == target_id]
            print(f"UVOT Source ID: {target_id}")

            for i in obs_list:
                print(i)
                obs_id = f'{target_id}00{i}' 
                print(f"UVOT observation ID: {obs_id}")
                try:
                    wget_command = ["wget",
                                    "-q", "-nH", "--cut-dirs=6", "-r", "-l0", "-c", "-np", "-R", "index*", "-erobots=off", 
                                    f"http://archive.stsci.edu/missions/swift_uvot/{target_id}/{obs_id}/"]
                    # wget_command = ["wget",
                    #                 "-A", "uvw1", 
                    #                 "-q", "-nH", "--cut-dirs=6", "-r", "-l0", "-c", "-np", "-R", "index*", "-erobots=off", 
                    #                 f"http://archive.stsci.edu/missions/swift_uvot/{target_id}/{obs_id}/"]
                    subprocess.run(wget_command)
                except Exception as e:
                    print(f"Failed to download observation files for {obs_id}")

        obs_files = glob.glob('sw*')
        
        if not obs_files:
            print(f'\nNo observation files were downloaded for {galaxy} {source_id}. Skipping')
            os.chdir(f"{top_directory}")
            continue
        
        else:
            print(os.getcwd())

            sk_files = glob.glob(f"sw*.img")
            # ex_files = glob.glob(f"sw*_ex.img.gz")
            # trimmed_files = files[:13]

            for file in sk_files:
                # goes through the image files and identifies uv sources, writes to region files, then writes source info to a file
                print(file)
                # trimmed_file = file.split("_")[0]
                # print(trimmed_file)
                base = os.path.basename(file) # sw00032614001uvv_sk.img
                trimmed_file = base.replace('_sk.img', '') # sw00032614001uvv
                print(trimmed_file)
                try:
                    with fits.open(file) as hdul:
                        for i, hdu in enumerate(hdul):
                            print(f"Extension {i}: {hdu.name}, {type(hdu)}")

                            # identify uv sources in this image
                            detect_inputs = f'{file}+{i}\ndetectU_test_{trimmed_file}.fits\n{trimmed_file}_ex.img.gz\n3\n' 
                            detect_result = subprocess.run(['uvotdetect'], input=detect_inputs.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            print("\nRunning uvotdetect...")
                            if detect_result.returncode != 0:
                                print(f"uvotdetect failed for {file}")
                                print(detect_result.stderr.decode())

                            #pick a spot outside the galaxy region
                            back_reg_text = f'# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\ncircle({back_ra.iloc[0]},{gal_dec.iloc[0]},10.000")'
                            back_filename = f'Uback_{galaxy}_{source_id}_{trimmed_file}.reg'
                            with open(back_filename, 'w') as f:
                                f.write(back_reg_text)

                            # write the source info to a file
                            source_file_text = f'# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\ncircle({src_ra},{src_dec},5.000")'
                            src_filename = f'Usource_{trimmed_file}.reg'
                            with open(src_filename, 'w') as f:
                                f.write(source_file_text)

                            source_inputs = f'{file}+{i}\n{src_filename}\n{back_filename}\n5\nphot_test_{trimmed_file}.fits'
                            source_result = subprocess.run(['uvotsource'], input=source_inputs.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                            print("\nRunning uvotsource...")
                            if source_result.returncode != 0:
                                print(f"uvotsource failed for {file}")
                                print(source_result.stderr.decode())
                except Exception as e:
                    print(f"Failed to open file {file}: {e}")
                    continue

            print(os.getcwd())
            phot_files = glob.glob(f"phot_test*")
            for i in phot_files:
                print(i)
                base = os.path.basename(i) 
                trimmed_name = base.replace('.fits', '') 
                print(trimmed_name)

                hdul = fits.open(i)
                data = hdul[1].data 
                cols = hdul[1].columns
                hdul.close()

                # adds the reddening column to the phot file 
                n_rows = len(data)

                ebv = row['mean_E_B_V_SFD']
                new_column_array = np.full(n_rows, ebv, dtype=np.float32)
                new_col = fits.Column(name='E_B_V_SFD_mean', format='E', array=new_column_array)
                new_cols = fits.ColDefs(list(cols) + [new_col])
                new_hdu = fits.BinTableHDU.from_columns(new_cols)
                new_hdu.writeto(f'{i}', overwrite=True)

                uv_info = get_uv_info(f'{i}', dist)
                print(type(uv_info))
                print(f'{galaxy} {source_id} {i} UV information:\n{uv_info}')

                uv_info_file = f'{galaxy}_{source_id}_{trimmed_name}.fits'

                try:
                    if uv_info != None:
                        uv_info.write(uv_info_file, overwrite=True)
                        print(f"Wrote file: {uv_info_file}")
                except Exception as e:
                    print(f"Failed to write file: {e}")

            
            save_path = '/Users/sophianicolella/Desktop/WAVE/analysis2/lightcurves'

            try:
                uvw1_files = glob.glob('phot_test*uw1*')
                uvw2_files = glob.glob('phot_test*uw2*')
                uvm2_files = glob.glob('phot_test*um2*')

                make_lightcurve(uvw1_files, 'UVW1', save_path)
                make_lightcurve(uvw2_files, 'UVW2', save_path)
                make_lightcurve(uvm2_files, 'UVM2', save_path)

            except Exception as e:
                print(f"Failed to write lightcurve file: {e}")
            

            end = time.time()
            print(f"\n\n----------------- Elapsed time to analyze source {galaxy}-{source_id}: {end - start:.2f} seconds -----------------\n\n")

            # os.remove(f'{top_directory}/{galaxy}/{source_id}/sw*')
            # removes all image files to clear space

            current_path = os.getcwd()

            dir_count = sum(
                os.path.isdir(os.path.join(current_path, entry))
                for entry in os.listdir(current_path)
            )

            print(f"Number of directories in {galaxy}: {dir_count}")

            if dir_count == len(source_df):
                os.chdir(top_directory)    
            else:
                os.chdir(f"{top_directory}/{galaxy}")
