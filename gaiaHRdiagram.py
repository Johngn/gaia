#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:55:14 2018

@author: david
"""

import numpy as np
import matplotlib.pyplot as plt
import coordTransform as coordT
from astroquery.gaia import Gaia
from astropy.io.votable import parse_single_table

source = 'mag10'
# source = 'ESA'
# %%
if(source == 'ESA'):
    tables = Gaia.load_tables(only_names=True)
    job = Gaia.launch_job_async("SELECT * FROM gaiadr2.gaia_source \
                                WHERE CONTAINS(POINT('ICRS',gaiadr2.gaia_source.ra,gaiadr2.gaia_source.dec),CIRCLE('ICRS', 56.75, 24.1167, 180))=1 \
                                AND phot_g_mean_mag<11.0 \
                                AND parallax IS NOT NULL \
                                AND pmra IS NOT NULL \
                                AND pmdec IS NOT NULL \
                                AND phot_g_mean_mag IS NOT NULL \
                                AND phot_bp_mean_mag IS NOT NULL \
                                AND phot_rp_mean_mag IS NOT NULL \
                                ;",dump_to_file=True)
    data = job.get_results()
else:
    table = parse_single_table(source+'.vot')        
    data = table.array
# %%
c              = np.asarray(data['bp_rp']);
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

icrsCov = coordT.CreateCovarianceMatrix_ra_dec(ra_error,dec_error,parallax_error,pmra_error,pmdec_error,
                                               ra_dec_corr, ra_parallax_corr, ra_pmra_corr, ra_pmdec_corr, 
                                               dec_parallax_corr, dec_pmra_corr, dec_pmdec_corr, 
                                               parallax_pmra_corr, parallax_pmdec_corr, pmra_pmdec_corr)      

icrsCoords = np.vstack((ra,dec,parallax,pmra,pmdec)).T
icrsErrors = np.vstack((ra_error,dec_error,parallax_error,pmra_error,pmdec_error)).T
    
GC = coordT.transformIcrsToGal(icrsCoords,icrsErrors,icrsCov)
                              
gcVal = GC[0]
gcErr = GC[1]
gcCov = GC[2]

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
#%%
err_filt = parallax > 20 * parallax_error
x = c[err_filt]
y = G[err_filt]+5*np.log10(parallax[err_filt]/100)

plt.figure(figsize=(10, 10))
plt.hist2d(x,y, cmap="jet", cmin=1, bins=2000 )
plt.xlim(-0.5, 3)
plt.ylim(-5, 10)
plt.gca().invert_yaxis()
plt.title("HR Diagram")
plt.xlabel('BP - RP [mag]')
plt.ylabel(r'$M_{g} [mag]$')
plt.savefig('hrdiagram10')
