#!/usr/bin/env python
# This is the first part of a script meant to automate the analysis of a catalog of ULX sources looking for corresponding bright UV emission
# This script will reduce the catalog to a new selection of suitable sources
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
parser.add_argument("--cutoff-axis-ratio", type=float, default=0.7, help='Define what ratio of bmin to amaj is "face-on".')
parser.add_argument("--luminosity-min", type=float, default=(1e40), help='Define the minimum luminosity a source may have (erg/s).')
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


#---------------------------------------------------------------------------------------------
# Step 1: Reducing the Catalog
#---------------------------------------------------------------------------------------------
orig_catalog = 'J_MNRAS_509_1587_master.fits'

with fits.open(orig_catalog) as hdul:
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

bright_df['is_ltg'] = bright_df['is_ltg'].astype(str)
bright_df['is_close'] = bright_df['is_close'].astype(str)
bright_df['is_bright'] = bright_df['is_bright'].astype(str)

columns = []

for col in bright_df.columns:
    data = bright_df[col].to_numpy()

    # Determine appropriate format
    if np.issubdtype(data.dtype, np.floating):
        fmt = 'D'  # 64-bit float
    elif np.issubdtype(data.dtype, np.integer):
        fmt = 'J'  # 32-bit int
    elif data.dtype.kind in {'U', 'S', 'O'}:  # string or object
        maxlen = max(len(str(x)) for x in data)
        fmt = f'A{maxlen}'
        data = np.array([str(x) for x in data])  # Ensure it's a string array
    else:
        raise ValueError(f"Unsupported dtype for column {col}: {data.dtype}")

    columns.append(fits.Column(name=col, array=data, format=fmt))

hdu = fits.BinTableHDU.from_columns(columns)
hdu.writeto('reduced_catalog.fits', overwrite=True)


# upload the resulting table to this website and download the resulting extinction table as 'extinction.tbl':
# https://irsa.ipac.caltech.edu/applications/DUST/
# then run the analysis script
