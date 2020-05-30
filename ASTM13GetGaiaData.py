#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:55:14 2018

@author: david
"""

import numpy as np
import coordTransform as coordT

from astroquery.gaia import Gaia
from astropy.io.votable import parse_single_table

mas2deg = 1.0/(3600*1000)
deg2rad = np.pi/180.0
mas2rad = mas2deg * deg2rad

source = 'ESA'
source = 'gaia8G_20180705'
source = 'gaia10G_20180705'
source = 'gaia11G_20180705'
source = 'gaia12G_20180705'

source = 'ESA'

if(source == 'ESA'):
    #coord = SkyCoord(ra=280, dec=-60, unit=(u.degree, u.degree), frame='icrs')
    #radius = u.Quantity(1.0, u.arcminute)
    #j = Gaia.cone_search_async(coord, radius)
    #r = j.get_results()
    #r.pprint()

    tables = Gaia.load_tables(only_names=True)
#    for table in (tables):
#        print (table.get_qualified_name())

    # An asynchronous query (asynchronous rather than synchronous queries should be performed when 
    #retrieving more than 2000 rows) centred on the Pleides (coordinates: 56.75, +24.1167) with a 
    #search radius of 1 degrees and save the results to a file.

    job = Gaia.launch_job_async("SELECT * FROM gaiadr2.gaia_source \
                                WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS', 56.75, 24.1167, 180))=1 \
                                AND phot_g_mean_mag<8.0 \
                                AND parallax IS NOT NULL \
                                AND pmra IS NOT NULL \
                                AND pmdec IS NOT NULL \
                                AND phot_g_mean_mag IS NOT NULL \
                                AND phot_bp_mean_mag IS NOT NULL \
                                AND phot_rp_mean_mag IS NOT NULL \
                                ;",dump_to_file=True)

    print (job)    

    data = job.get_results()
else:
    table = parse_single_table(source+'.vot')
    #print (table.get_qualified_name())
        
    data = table.array

print('Ready now')
source_id      = np.asarray(data['source_id']);

# Make them proper arrays
c = np.asarray(data['bp_rp']);
#G = −2.5 log(flux) + zeropoint
#  = -2.5*log10(550.00816) + 25.691439
G              = np.asarray(data['phot_g_mean_mag']);
BP             = np.asarray(data['phot_bp_mean_mag']);
RP             = np.asarray(data['phot_rp_mean_mag']);
ra             = np.asarray(data['ra']);
ra_error       = np.asarray(data['ra_error']);
dec            = np.asarray(data['dec']);
dec_error      = np.asarray(data['dec_error']);
parallax       = np.asarray(data['parallax']);
parallax_error = np.asarray(data['parallax_error']);
pmra           = np.asarray(data['pmra']);
pmra_error     = np.asarray(data['pmra_error']);                  
pmdec          = np.asarray(data['pmdec']);
pmdec_error    = np.asarray(data['pmdec_error']);

ra_dec_corr         = np.asarray(data['ra_dec_corr']);
ra_parallax_corr    = np.asarray(data['ra_parallax_corr']);
ra_pmra_corr        = np.asarray(data['ra_pmra_corr']);
ra_pmdec_corr       = np.asarray(data['ra_pmdec_corr']);
dec_parallax_corr   = np.asarray(data['dec_parallax_corr']);
dec_pmra_corr       = np.asarray(data['dec_pmra_corr']);
dec_pmdec_corr      = np.asarray(data['dec_pmdec_corr']);
parallax_pmra_corr  = np.asarray(data['parallax_pmra_corr']);                  
parallax_pmdec_corr = np.asarray(data['parallax_pmdec_corr']);
pmra_pmdec_corr     = np.asarray(data['pmra_pmdec_corr']);


# Statistical indicators to use for selecting sources
#-----------------------------------------------------
# astrometric_n_good_obs_al
# astrometric_gof_al
# astrometric_chi2_al
# astrometric_excess_noise
# astrometric_excess_noise_sig
# astrometric_primary_flag
# astrometric_matched_observations
# visibility_periods_used


icrsCov = coordT.CreateCovarianceMatrix_ra_dec(ra_error,dec_error,parallax_error,pmra_error,pmdec_error,
                                               # Correlations
                                               ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, 
                                               dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, 
                                               parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr)
                               
                                               


icrsCoords = np.vstack((ra,dec,parallax,pmra,pmdec)).T
icrsErrors = np.vstack((ra_error,dec_error,parallax_error,pmra_error,pmdec_error)).T
    
# Icrs to Gal: conversion w.  covarience matrix
GC = coordT.transformIcrsToGal(icrsCoords,icrsErrors,icrsCov)
                              
gcVal = GC[0]
gcErr = GC[1]
gcCov = GC[2]

#ICRS convention
wrap_angle = 0.0;
value = np.argwhere(gcVal[:,0] < wrap_angle)
gcVal[value,0] = gcVal[value,0] + 360.0 

l     = gcVal[:,0]
b     = gcVal[:,1]
varpi = gcVal[:,2]
mul   = gcVal[:,3]
mub   = gcVal[:,4]
sigma_l = np.sqrt(gcCov[:,0,0]);
sigma_b = np.sqrt(gcCov[:,1,1]);
sigma_varpi = np.sqrt(gcCov[:,2,2]);
sigma_mul = np.sqrt(gcCov[:,3,3]);
sigma_mub = np.sqrt(gcCov[:,4,4]);

# f = open('datafile.dat', 'w')
# f.write('%s %s %s %s %s %s %s %s %s %s %s %s %s\n' % ('source_id',
#         'l', 'b', 'varpi', 'mul', 'mub', 
#         'sigma_l', 'sigma_b', 'sigma_varpi', 'sigma_mul', 'sigma_mub', 
#         'c', 'G'))
# for i in range(len(G)):
#     f.write('%d %f %f %f %f %f %f %f %f %f %f %f %f\n' % (source_id[i],
#             l[i], b[i], varpi[i], mul[i], mub[i], 
#             sigma_l[i], sigma_b[i], sigma_varpi[i], sigma_mul[i], sigma_mub[i], 
#             c[i], G[i]))
# f.flush()
# f.close()


