#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 14:55:14 2018

@author: david
"""

import numpy as np
import coordTransform as coordT
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from astroquery.gaia import Gaia
from astropy.io.votable import parse_single_table

#%%

mas2deg = 1.0/(3600*1000)
deg2rad = np.pi/180.0
mas2rad = mas2deg * deg2rad

source = 'ESA'
source = 'gaia8G_20180705'
source = 'gaia10G_20180705'
source = 'gaia11G_20180705'
source = 'gaia12G_20180705'

source = 'async_20191119150304'

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
                                AND phot_g_mean_mag<7.0 \
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

    
#%%


# divide mul and l into equal sized colour ranges
kmul = [mul[np.logical_and(np.logical_and(c < i + 1, c > i), np.logical_and(parallax > 10 * parallax_error, np.logical_and(np.logical_and(parallax > 1, parallax < 1.1) , np.abs(b) < 20)))] * 4.7405 for i in np.arange(-0.2, 1.0, 0.2)]
newl = [l[np.logical_and(np.logical_and(c < i + 1, c > i), np.logical_and(parallax > 10 * parallax_error, np.logical_and(np.logical_and(parallax > 1, parallax < 1.1), np.abs(b) < 20)))] for i in np.arange(-0.2, 1.0, 0.2)]

#%%

for i in range(len(kmul)):
    plt.figure()
    plt.title(f'{round((i + 1)/10 - 0.4, 2)} to {round((i + 1)/10 - 0.2, 2)} BP-RP')
    plt.scatter(np.cos(np.radians(newl[i]) * 2), kmul[i], s=0.01)
    plt.ylim(-200, 200)
    plt.xlabel('cos2l')
    plt.ylabel(r'proper motion ($\mu_l$)')
    plt.savefig(f"coslresults{i + 1}")
    
    
#%%
    

def leastSquaresFit(x, A, B):
     return A * x + B

dataFit = [curve_fit(leastSquaresFit, np.cos(np.radians(newl[i]) * 2), kmul[i]) for i in range(len(kmul))]

oortConstants = []
oortErrors = []
for i in range(len(dataFit)):
    oortConstants.append(dataFit[i][0])
    oortErrors.append(np.sqrt(np.diag(dataFit[i][1])))


#%%


x = []
y = []

for i in range(len(b)):
    if parallax[i] > 20 * parallax_error[i]:
        X = c[i]
        Y = G[i] + 5 * np.log10(parallax[i]/ 100)
        x.append(X)
        y.append(Y)
        


plt.figure(figsize=(8, 10))
plt.hist2d(x,y, cmap = "jet", cmin = 1, bins=5000 )
plt.xlim(-0.5, 2)
plt.ylim(-5, 10)
plt.gca().invert_yaxis()
plt.title("HR Diagram")
plt.xlabel('BP - RP [mag]')
plt.ylabel(r'$M_{g} [mag]$')
plt.savefig('hrdiagram')