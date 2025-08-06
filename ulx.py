#!/usr/bin/env python
# This is a script meant to automate the analysis of a catalog of ULX sources looking for corresponding bright UV emission
# Before running, make sure HEASoft is downloaded, henv is activated, and the CALDB variable is set and sourced

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import astropy
from astropy.io import fits
from astropy.table import Table, hstack, vstack
from astropy.coordinates import SkyCoord
import glob
import os
import argparse
import subprocess
from astroquery.mast import Observations
from astroquery.ipac.irsa.irsa_dust import IrsaDust
from tabulate import tabulate
import time
from scipy.stats import chisquare


#---------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Specify catalog reduction parameters and UV data download process.')
parser.add_argument("--cutoff-distance", type=float, default=25, help='Define the upper limit for the distance in Mpc of a suitable source.')
parser.add_argument("--cutoff-axis-ratio", type=float, default=0.5, help='Define what ratio of bmin to amaj is "face-on".')
parser.add_argument("--luminosity-min", type=float, default=(1e39), help='Define the minimum luminosity a source may have (erg/s).')
parser.add_argument("--luminosity-max", type=float, default=(1e41), help='Define the maximum luminosity a source may have (erg/s).')
parser.add_argument("--num-target-download", type=float, default=5, help='The number of targets for which to download observation files.')
parser.add_argument("--num-obs-download", type=float, default=5, help='The number of observation files to download per target (Input-1).')
args = parser.parse_args()

print(f"Using cutoff distance: {args.cutoff_distance} Mpc")
print(f"Using cutoff axis ratio: {args.cutoff_axis_ratio}")
print(f"Using minimum luminosity: {args.luminosity_min} erg/s")
print(f"Using maximum luminosity: {args.luminosity_max} erg/s")

#---------------------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
#---------------------------------------------------------------------------------------------
# small functions
#--------------------------------------#

def is_ltg(col):
    return col.astype(float) >= 0
#returns only the sources with a t-type greater than 0, meaning they are late-type, or young
#these galaxies are more likely to have higher star-formation rates and therefore more ULXs/ULUVs


def is_close(col):
    return col.astype(float) < args.cutoff_distance
#returns only the sources closer than 25 Mpc
#sources at larger distances become difficult to resolve, more likely to have material causing extinction in the way
#requires that the catalog have distances listed with units of Mpc


def axis_ratio(amaj, bmin):
    ratio = bmin / amaj
    return ratio
#calculates the ratio between the semiminor and semimajor axes


def is_filled(col):
    return col.astype(str).str.strip() != ''
#used this to remove duplicate sources, need to modify to generalize


def is_bright(col):
    return (col.astype(float) > args.luminosity_min) & (col.astype(float) < args.luminosity_max)
#used to specify which luminosity range I want the resulting sample to remain within


def get_radius(amaj, dist): #dist in Mpc, amaj in arcmin
    r = (dist * 1e6) * ((np.pi)/180*60) * amaj 
    return r
#used to calculate the radius of the galaxy in parsecs

#--------------------------------------#
# large functions 
#--------------------------------------#
def l_to_use(df, sxps, xmm, csc2):
    l_to_use = []
    for _, row in df.iterrows():
        #get the Lpeak for each instrument
        val1 = row[sxps] #Swift (2SXPS)
        val2 = row[xmm] #XMM-Newton (4XMM)
        val3 = row[csc2] #Chandra (CSC2)
        
        #check which columns are filled
        values = [val for val in [val1, val2, val3] if pd.notnull(val) and str(val).strip() != '' and val != 0]
        
        #append the first value that appears in a certain row to remove duplicates for each detection
        if values:
            l_to_use.append(values[0])
        else:
            None
    df['L_to_use'] = l_to_use
    return df['L_to_use']



#the function used to get the absolute magnitude with reddening correction for a certain source
#need to figure out how to automate obtaining the reddening coefficient from a certain part of the sky
#the file is the fits table for that specific source
#now also gets mass estimate from absolute magnitude
def get_uv_info(file, dist):
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

    start_time = np.min(time_array)
    time_span = 1e7
    end_time = start_time + time_span
    mask = (time_array >= start_time) & (time_array <= end_time)

    time_filtered = time_array[mask]
    flux_filtered = flux_array[mask]
    error_filtered = flux_err_array[mask]

    lightcurve_table = Table([time_filtered, flux_filtered, error_filtered], names=('Time', 'Flux', 'Flux_Err'))
    print(len(time_filtered), len(flux_filtered), len(error_filtered))
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
    ax.errorbar(time_filtered, flux_filtered, yerr=error_filtered, marker='s', mfc='red', mec='black', linestyle='None')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Flux (erg/s/cm²)')
    ax.set_title(f"{galaxy} {source_id} Lightcurve ({filter})")

    fig_path = os.path.join(save_dir, f'lightcurve_{galaxy}_{source_id}_{filter}.png')
    fig.savefig(fig_path)
    print(f"Saved plot: {fig_path}")

    figures.append(fig) 

    return figures


#---------------------------------------------------------------------------------------------
# Step 1: Reducing the Catalog
#---------------------------------------------------------------------------------------------
#hdul = astropy.io.fits.open("J_MNRAS_509_1587_master.fits") #input your catalog file
# hdul = astropy.io.fits.open("selected_rows.fits")
# master_data = hdul[1].data 
# master_cols = hdul[1].columns
# hdul.close()

# orig_catalog = 'selected_rows.fits'
orig_catalog = 'J_MNRAS_509_1587_master.fits'

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

table1 = Table.read(orig_catalog)
table2 = Table.read(extinction)
combined_table = hstack([table1, table2])
combined_table.write('selected_new.fits', format='fits')


with fits.open("selected_new.fits") as hdul:
    data = hdul[1].data  # FITS_rec structure

    master_data = {}

    for name in data.names:
        col = data[name]

        # Convert to native-endian if needed
        if col.dtype.byteorder not in ('=', '|'):
            col = col.byteswap().view(col.dtype.newbyteorder('='))
        
        master_data[name] = col


master_df = pd.DataFrame(master_data)
master_df['L_to_use'] = l_to_use(master_df, 'Lpeak2SXPS', 'Lpeak4XMM', 'LpeakCSC2') #gets rid of duplicate luminosities per source (across instruments)

print('Length of original catalog:', len(master_df), 'sources')

print('Removing unsuitable sources...')

# determine the t-type of each source and only keep the late-type galaxies
master_df['is_ltg'] = is_ltg(master_df['T-Type'])
ltg_df = master_df[master_df['is_ltg'] > 0]
print('Length of catalog after removing ETGs:', len(ltg_df), 'sources')

# determine the distance of each source and keep only the ones closer than value specified in function defn. (Mpc)
ltg_df['is_close'] = is_close(ltg_df['Dist'])
close_df = ltg_df[ltg_df['is_close']]
print('Length of catalog after removing faraway galaxies:', len(close_df), 'sources')

# determine the approx. inclination of each source and keep only those that are face-on
close_df['axis_ratio'] = axis_ratio(close_df['amaj'], close_df['bmin'])
edge_on_df = close_df[close_df['axis_ratio'] < 0.5]
face_on_df = close_df[close_df['axis_ratio'] >= args.cutoff_axis_ratio] # cutoff value can be modified to user preferences
print('Length of catalog after removing edge-on galaxies:', len(face_on_df), 'sources')

# determine the x-ray luminosity of each source and keep only those within a certain specified range
face_on_df['is_bright'] = is_bright(face_on_df['L_to_use'])
bright_df = face_on_df[face_on_df['is_bright']]
print(f'Length of catalog after specifying luminosity range: {len(bright_df)} sources')

hdu = fits.BinTableHDU.from_columns([
    fits.Column(name=col, array=bright_df[col].to_numpy(), format='D')  # 'D' = 64-bit float
    for col in bright_df.columns
])

# Write to FITS file
hdu.writeto('reduced_catalog.fits', overwrite=True)



#---------------------------------------------------------------------------------------------
# Step 2: Analyzing UV Data
#---------------------------------------------------------------------------------------------
reduced_df = bright_df
unique_values = reduced_df['Name'].unique()
print(f'Number of unique host galaxies: {len(unique_values)} galaxies')
top_directory = '/Users/sophianicolella/Desktop/test2'

for galaxy in unique_values:
    print(f'\n\n----------------- {galaxy} -----------------\n\n')
    pwd = os.getcwd()

    source_df = reduced_df[reduced_df["Name"] == galaxy]
    # this is going to cause problems when there are multiple sources in one galaxy

    gal_ra = source_df['RAdeg'] #degrees
    gal_dec = source_df['DEdeg'] #degrees
    amaj = source_df['amaj'] #arcmin
    galaxy_radius = amaj / 60 #degrees
    back_ra = gal_ra + galaxy_radius + (5 / 3600) #degrees 
    # when there is more than one source, may have to move this into the source_df loop?
    # but the galaxy ra and dec should be the same for each source within a host
    # might just need to specify with .iloc[0]

    os.makedirs(galaxy)
    os.chdir(f'{pwd}/{galaxy}')
    print(os.getcwd())

    for _, row in source_df.iterrows():
        start = time.time()

        source_id = row['2SXPSID'] #use the swift ID since we are going to retreive UVOT data
        print(f'\n\n----------------- Swift source ID: {source_id} -----------------\n\n')

        os.makedirs(str(source_id))
        print(os.getcwd())
        os.chdir(f'/Users/sophianicolella/Desktop/test2/{galaxy}/{source_id}')
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
        obs_table = Observations.query_region(f"{src_ra} {src_dec}")

        obs_df = obs_table.to_pandas()

        orig_obs_id = obs_df['obs_id']
        obs_df['target_id'] = obs_df['obs_id'].astype(str).str[:-3]
        target_id = obs_df['target_id']
        obs_src_ra = obs_df['s_ra']
        print(len(obs_df))


        # Optional: update your other variables based on filtered dataframe
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
            raise FileNotFoundError('No observation files were downloaded.')
                

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

            with fits.open(file) as hdul:
                for i, hdu in enumerate(hdul):
                    print(f"Extension {i}: {hdu.name}, {type(hdu)}")

                    # identify uv sources in this image
                    detect_inputs = f'{file}+{i}\ndetectU_test_{trimmed_file}.fits\n{trimmed_file}_ex.img.gz\n3\n' 
                    detect_result = subprocess.run(['uvotdetect'], input=detect_inputs.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    print("Running uvotdetect...")
                    if detect_result.returncode != 0:
                        print(f"uvotdetect failed for {file}")
                        print(detect_result.stderr.decode())

                    #pick a spot outside the galaxy region
                    back_reg_text = f'# Region file format: DS9 version 4.1\nglobal color=green dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\nfk5\ncircle({back_ra.iloc[0]},{gal_dec.iloc[0]},5.000")'
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
                    print("Running uvotsource...")
                    if source_result.returncode != 0:
                        print(f"uvotsource failed for {file}")
                        print(source_result.stderr.decode())

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

        
        save_path = '/Users/sophianicolella/Desktop/test2/lightcurves_test'

        uvw1_files = glob.glob('phot_test*uw1*')
        uvw2_files = glob.glob('phot_test*uw2*')
        uvm2_files = glob.glob('phot_test*um2*')

        make_lightcurve(uvw1_files, 'UVW1', save_path)
        make_lightcurve(uvw2_files, 'UVW2', save_path)
        make_lightcurve(uvm2_files, 'UVM2', save_path)
        

        end = time.time()
        print(f"\n\n----------------- Elapsed time to analyze source {galaxy}-{source_id}: {end - start:.2f} seconds -----------------\n\n")

        # os.remove(f'/Users/sophianicolella/Desktop/test2/{galaxy}/{source_id}/sw*')
        # removes all image files to clear space

        os.chdir(top_directory)    
