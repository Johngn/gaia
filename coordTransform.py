#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 09:46:35 2018

@author: David Hobbs, Lund Observatory
v 0.1 author Paul McMillan
"""
import numpy as np

mas2deg = 1.0/(3600*1000)
deg2rad = np.pi/180.0
mas2rad = mas2deg * deg2rad

Aprime  = np.asarray([[-0.0548755604162154, -0.8734370902348850, -0.4838350155487132],
                      [+0.4941094278755837, -0.4448296299600112, +0.7469822444972189],
                      [-0.8676661490190047, -0.1980763734312015, +0.4559837761750669]])


def transformGalToIcrs( GalCoords, GalUncerts, GalC = None) :
    '''Transforms coordinates and uncertainties (/covariance matrix) from Galactic coordinates to Equatorial (ICRS) coordinates

    This code impliments Gaia Data Release 2, Documentation v1.1,
    Section 3.1.7: Transformations of astrometric data and error propagation (Alexey Butkevich, Lennart Lindegren,
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html)
    Code written by David Hobbs & Paul McMillan

    Parameters:
    ------------

    GalCoords: Array of dimensions (N,5), with N number of coords
        (l, b, parallax, pm_lStar, pm_b). Units (respectively) deg, deg, mas, mas/yr, mas/yr.
                Must be in an array of dimensions (N,5) where N is number of coords

    GalUncerts: Array of dimensions (N,5), with N number of coords
    (l_error, b_error, parallax_error, pm_lStar_error, pm_b_error). Units mas or mas/yr
                Must be in an array of dimensions (N,5) where N is number of coords

    GalC [optional]: Array of dimensions (N,5,5), with N number of coords
        Covariance Matrix describing uncertainties in galactic coordinates.
        Writing the Errors_in array (above) as sigma_i, this matrix, C, has diagonal elements
        C_ii = sigma_i**2,
        and off-diagonal elements
        C_ij = rho_ij sigma_i sigma_j,
        with rho_ij the correlation coefficients

    Returns:
    -----------

    EqCoords: Array of dimensions (N,5), with N number of coords
        (ra, dec, parallax, pm_raStar, pm_dec). Units (respectively) deg, deg, mas, mas/yr, mas/yr.

    EqUncerts: Array of dimensions (N,5), with N number of coords
        (ra_error, dec_error, parallax_error, pm_raStar_error, pm_dec_error). Units mas or mas/yr.

    EqC : Array of dimensions (N,5,5), with N number of coords
        Covariance Matrix describing uncertainties in Galactic coordinates.
        Writing the Errors_in array (above) as sigma_i, this matrix, C, has diagonal elements
        C_ii = sigma_i**2,
        and off-diagonal elements
        C_ij = rho_ij sigma_i sigma_j,
        with rho_ij the correlation coefficients

    Notes:
    ------------
    If GalC is provided, EqErrors is ignored (as redundant)

    If GalC is not provided, correlation coefficients are assumed to be zero

    '''

    # Number of terms
    nterms = GalCoords.shape[0]

    CoordConverter = np.array([deg2rad,deg2rad,mas2rad,mas2rad,mas2rad])
    lbCoords = GalCoords * CoordConverter

    lbUncerts = GalUncerts*mas2rad
    if GalC is None :
        Cgal = np.zeros([nterms,5,5])
        row,col = np.diag_indices(5)
        Cgal[:,row,col] = lbUncerts**2
    else :
        Cgal = GalC*mas2rad*mas2rad
    l = lbCoords[:,0]
    b = lbCoords[:,1]
    # Construct rGal
    rGal = np.transpose(np.asarray([np.cos(l)*np.cos(b),
                                    np.sin(l)*np.cos(b),
                                    np.sin(b)]))
    # Construct rIcrs
    rIcrs = np.einsum('ij,kj->ki',np.transpose(Aprime),rGal)

    alpha = np.arctan2(rIcrs[:,1], rIcrs[:,0])
    delta = np.arctan2(rIcrs[:,2], np.sqrt(rIcrs[:,0]**2 + rIcrs[:,1]**2))

    # The transformation of the proper motion components
    pIcrs = np.asarray([-np.sin(alpha),
                         np.cos(alpha),
                         np.zeros_like(alpha)])
    qIcrs = np.asarray([-np.cos(alpha)*np.sin(delta),
                        -np.sin(alpha)*np.sin(delta),
                         np.cos(delta)])

    pGal  = np.asarray([-np.sin(l),
                        np.cos(l),
                        np.zeros_like(l)])
    qGal  = np.asarray([-np.cos(l)*np.sin(b),
                        -np.sin(l)*np.sin(b),
                        np.cos(b)])
    mulStar = lbCoords[:,3]
    mub = lbCoords[:,4]
    muGal =  pGal*mulStar + qGal*mub
    muIcrs  = np.einsum('ij,jk ->ik',np.transpose(Aprime),muGal)

    # Icrs proper motions
    muAlphaStar = np.sum(pIcrs*muIcrs,axis=0)
    muDelta     = np.sum(qIcrs*muIcrs,axis=0)

    gal  = (np.hstack((pGal.T, qGal.T))).reshape(nterms,2,3)
    icrs = (np.hstack((pIcrs.T, qIcrs.T))).reshape(nterms,2,3)

    tmp = np.einsum('ij,klj->kil',Aprime,icrs)
    galT = np.einsum('ijk->ikj', gal)
    G = np.einsum('ijk,ikl->ijl', gal,tmp)
    # Jacobian

    Arr0 = np.zeros(nterms)
    Arr1 = np.ones(nterms)
    J = np.asarray([[G[:,0,0], G[:,0,1], Arr0,     Arr0,     Arr0],
                    [G[:,1,0], G[:,1,1], Arr0,     Arr0,     Arr0],
                    [    Arr0,     Arr0, Arr1,     Arr0,     Arr0],
                    [    Arr0,     Arr0, Arr0, G[:,0,0], G[:,0,1]],
                    [    Arr0,     Arr0, Arr0, G[:,1,0], G[:,1,1]]])
    J = np.einsum('ijk->kij',J)
    JT = np.einsum('ijk->ikj',J)

    tmp = np.einsum('ijk,ikl->ijl',JT,Cgal)
    EqC = np.matmul(np.matmul(JT,Cgal),J)
    EqCoords = np.atleast_2d(np.asarray((alpha,delta,lbCoords[:,2],muAlphaStar,muDelta))).T/CoordConverter
    EqUncerts = np.atleast_2d(np.sqrt(np.diagonal(EqC,axis1=1,axis2=2)))/mas2rad
    EqC = EqC/mas2rad**2
    return EqCoords, EqUncerts, EqC


def transformIcrsToGal( EqCoords, EqUncerts, EqC = None) :
    '''Transforms coordinates and uncertainties (/covariance matrix) from Equatorial (ICRS) coordinates to Galactic coordinates

    This code impliments Gaia Data Release 2, Documentation v1.1,
    Section 3.1.7: Transformations of astrometric data and error propagation (Alexey Butkevich, Lennart Lindegren,
    https://gea.esac.esa.int/archive/documentation/GDR2/Data_processing/chap_cu3ast/sec_cu3ast_intro/ssec_cu3ast_intro_tansforms.html)
    Code written by David Hobbs & Paul McMillan

    Parameters:
    ------------

    EqCoords: Array of dimensions (N,5), with N number of coords
        (ra, dec, parallax, pm_raStar, pm_dec). Units (respectively) deg, deg, mas, mas/yr, mas/yr.
                Must be in an array of dimensions (N,5) where N is number of coords

    EqUncerts: Array of dimensions (N,5), with N number of coords
    (ra_error, dec_error, parallax_error, pm_raStar_error, pm_dec_error). Units mas or mas/yr
                Must be in an array of dimensions (N,5) where N is number of coords

    EqC [optional]: Array of dimensions (N,5,5), with N number of coords
        Covariance Matrix describing uncertainties in equatorial coordinates.
        Writing an individual objects EqUncerts array (above) as sigma_i, this matrix, C, has diagonal elements
        C_ii = sigma_i**2,
        and off-diagonal elements
        C_ij = rho_ij sigma_i sigma_j,
        with rho_ij the correlation coefficients

    Returns:
    -----------

    GalCoords: Array of dimensions (N,5), with N number of coords
        (l, b, parallax, pm_lStar, pm_b). Units (respectively) deg, deg, mas, mas/yr, mas/yr.

    GalUncerts: Array of dimensions (N,5), with N number of coords
        (l_error, b_error, parallax_error, pm_lStar_error, pm_b_error). Units mas or mas/yr.

    GalC : Array of dimensions (N,5,5), with N number of coords
        Covariance Matrix describing uncertainties in Galactic coordinates.
        Writing the GalUncerts array (above) as sigma_i, this matrix, C, has diagonal elements
        C_ii = sigma_i**2,
        and off-diagonal elements
        C_ij = rho_ij sigma_i sigma_j,
        with rho_ij the correlation coefficients

    Notes:
    ------------
    If EqC is provided, EqUncerts is ignored (as redundant)

    If EqC is not provided, correlation coefficients are assumed to be zero

    '''
    # Number of terms
    nterms = EqCoords.shape[0]

    # Unit conversion - rename things to IcrsXXX to avoid confusion
    CoordConverter = np.array([deg2rad,deg2rad,mas2rad,mas2rad,mas2rad])
    IcrsCoord = EqCoords * CoordConverter
    IcrsUncerts = EqUncerts*mas2rad

    # If no covariance matrix is provided, correlation coefficients are assumed to be 0,
    # and covariances taken for quoted uncertainties
    if EqC is None :
        C = np.zeros([nterms,5,5])
        row,col = np.diag_indices(5)
        C[:,row,col] = IcrsUncerts**2
    else :
        C = EqC*mas2rad*mas2rad

    alpha = IcrsCoord[:,0] #(IcrsCoord[:,0]).reshape(nterms)
    delta = IcrsCoord[:,1] #.reshape(nterms)

    # Construct rIcrs
    rIcrs = np.transpose(np.asarray([np.cos(alpha)*np.cos(delta),
                                    np.sin(alpha)*np.cos(delta),
                                    np.sin(delta)]))

    # Construct rGal
    rGal = np.einsum('ij,kj->ki',Aprime,rIcrs)

    l = np.arctan2(rGal[:,1], rGal[:,0])
    b = np.arctan2(rGal[:,2], np.sqrt(rGal[:,0]**2 + rGal[:,1]**2))

    # The transformation of the proper motion components, eq 3.64, 3.65
    pGal  = np.asarray([-np.sin(l),
                        np.cos(l),
                        np.zeros_like(l)])
    qGal  = np.asarray([-np.cos(l)*np.sin(b),
                        -np.sin(l)*np.sin(b),
                        np.cos(b)])
    pIcrs = np.asarray([-np.sin(alpha),
                         np.cos(alpha),
                         np.zeros_like(alpha)])
    qIcrs = np.asarray([-np.cos(alpha)*np.sin(delta),
                        -np.sin(alpha)*np.sin(delta),
                         np.cos(delta)])

    muAlphaStar = IcrsCoord[:,3]
    muDelta = IcrsCoord[:,4]
    # eq 3.66, 3.67
    muIcrs =  pIcrs*muAlphaStar + qIcrs*muDelta
    muGal  = np.einsum('ij,jk ->ik',Aprime,muIcrs)

    # Galactic proper motions, eq 3.70
    mulStar = np.sum(pGal*muGal,axis=0)
    mub     = np.sum(qGal*muGal,axis=0)

    # eq 3.80
    gal  = (np.hstack((pGal.T, qGal.T))).reshape(nterms,2,3)
    icrs = (np.hstack((pIcrs.T, qIcrs.T))).reshape(nterms,2,3)

    tmp = np.einsum('ij,klj->kil',Aprime,icrs)
    G = np.einsum('ijk,ikl->ijl', gal,tmp)

    # Jacobian, eq 3.77, 3.79
    Arr0 = np.zeros(nterms)
    Arr1 = np.ones(nterms)
    J = np.asarray([[G[:,0,0], G[:,0,1], Arr0,     Arr0,     Arr0],
                    [G[:,1,0], G[:,1,1], Arr0,     Arr0,     Arr0],
                    [    Arr0,     Arr0, Arr1,     Arr0,     Arr0],
                    [    Arr0,     Arr0, Arr0, G[:,0,0], G[:,0,1]],
                    [    Arr0,     Arr0, Arr0, G[:,1,0], G[:,1,1]]])
    # rearrange terms
    J = np.einsum('ijk->kij',J)
    JT = np.einsum('ijk->ikj',J)

    GalC = np.matmul(np.matmul(J,C),JT)
    #print(np.einsum('ijk,ikl->ijl',J,JT))
    GalCoords = np.atleast_2d(np.asarray((l,b,IcrsCoord[:,2],mulStar,mub))).T/CoordConverter
    GalUncerts = np.atleast_2d(np.sqrt(np.diagonal(GalC,axis1=1,axis2=2)))/mas2rad
    GalC = GalC/mas2rad**2
    return GalCoords, GalUncerts, GalC


def CreateCovarianceMatrix_ra_dec (ra_error, dec_error, parallax_error, pmra_error, pmdec_error,
                                   radecCorr=0, raparallaxCorr=0, rapmraCorr=0, rapmdecCorr=0,
                                   decparallaxCorr=0, decpmraCorr=0, decpmdecCorr=0,
                                   parallaxpmraCorr=0, parallaxpmdecCorr=0,
                                   pmrapmdecCorr=0) :
    '''Create a covarience matrix from input uncertainties and correlations in ICRS coordinates

    Parameters:
    ------------

    ra_error, dec_error, parallax_error, pmra_error, pmdec_error :
        Each must be an array of values (all of the same length, N)

    XXXCorr: corellations between parameters.
        Must be either an array of length N or a value.
        If not given, assumed to be zero.

    Returns:
    ---------

    C: Covariance matrix. Format numpy array of dimensions (nterms,5,5)

    '''
    nterms =  len(ra_error)

    for err in [dec_error, parallax_error, pmra_error, pmdec_error] :
        assert (len(err) ==  nterms), 'error terms must have the same length'

    FirstPart  = [0,0,0,0,1,1,1,2,2,3]
    SecondPart = [1,2,3,4,2,3,4,3,4,4]
    Correlations = [radecCorr, raparallaxCorr, rapmraCorr, rapmdecCorr,
                    decparallaxCorr, decpmraCorr, decpmdecCorr,
                    parallaxpmraCorr, parallaxpmdecCorr,
                    pmrapmdecCorr]


    C = np.zeros([nterms,5,5])
    C[:,0,0] = ra_error**2
    C[:,1,1] = dec_error**2
    C[:,2,2] = parallax_error**2
    C[:,3,3] = pmra_error**2
    C[:,4,4] = pmdec_error**2

    for a,b,Corr in zip(FirstPart, SecondPart, Correlations) :
        C[:,a,b] = C[:,b,a] = np.sqrt(C[:,a,a]*C[:,b,b])*Corr
    return C


def CreateCovarianceMatrix_l_b (l_error, b_error, varpi_error, pmlStar_error, pmb_error,
                                   lbCorr=0, lvarpiCorr=0, lpmlStarCorr=0, lpmbCorr=0,
                                   bvarpiCorr=0, bpmlStarCorr=0, bpmbCorr=0,
                                   varpipmlStarCorr=0, varpipmbCorr=0,
                                   pmlStarpmbCorr=0) :
    '''Create a covarience matrix from input uncertainties and correlations in Galactic coordinates

    Parameters:
    ------------

    l_error, b_error, varpi_error, pmlStar_error, pmb_error :
        Each must be an array of values (all of the same length, N)

    XXXCorr: corellations between parameters.
        Must be either an array of length N or a value.
        If not given, assumed to be zero.

    Returns:
    ---------

    C: Covariance matrix. Format numpy array of dimensions (nterms,5,5)
    '''
    nterms =  len(l_error)

    for err in [ b_error, varpi_error, pmlStar_error, pmb_error] :
        assert (len(err) ==  nterms), 'error terms must have the same length'

    FirstPart  = [0,0,0,0,1,1,1,2,2,3]
    SecondPart = [1,2,3,4,2,3,4,3,4,4]
    Correlations = [lbCorr, lvarpiCorr, lpmlStarCorr, lpmbCorr,
                    bvarpiCorr, bpmlStarCorr, bpmbCorr,
                    varpipmlStarCorr, varpipmbCorr,
                    pmlStarpmbCorr]

    C = np.zeros([nterms,5,5])
    C[:,0,0] = l_error**2
    C[:,1,1] = b_error**2
    C[:,2,2] = varpi_error**2
    C[:,3,3] = pmlStar_error**2
    C[:,4,4] = pmb_error**2

    for a,b,Corr in zip(FirstPart, SecondPart, Correlations) :
        C[:,a,b] = C[:,b,a] = np.sqrt(C[:,a,a]*C[:,b,b])*Corr
    return C


def testConversions() :
    # test values
    ra             = np.asarray([9.1185, 30.00379737])
    dec            = np.asarray([+01.08901332, -19.49883745])
    parallax       = np.asarray([3.54,          21.90])
    pmra           = np.asarray([-5.20,        181.21])
    pmdec          = np.asarray([-1.88,         -0.93])

    l,b,pmlStar,pmb = ra,dec,pmra,pmdec
    ##
    ra_error       = np.asarray([1.32,           1.28])
    dec_error      = np.asarray([0.74,           0.70])
    parallax_error = np.asarray([1.39,           3.10])
    pmra_error     = np.asarray([1.36,           1.74])
    pmdec_error    = np.asarray([0.81,           0.92])

    l_error,b_error,pmlStar_error,pmb_error = ra_error,dec_error,pmra_error,pmdec_error
    #
    icrsCoords = np.vstack((ra,dec,parallax,pmra,pmdec)).T
    icrsErrors = np.vstack((ra_error,dec_error,parallax_error,pmra_error,pmdec_error)).T

    # Obviously these are also valid coords, though not the same
    GalCoords = np.vstack((l,b,parallax,pmlStar,pmb)).T
    GalErrors = np.vstack((l_error,b_error,parallax_error,pmlStar_error,pmb_error)).T

    icrsCov = CreateCovarianceMatrix_ra_dec(ra_error,dec_error,parallax_error,pmra_error,pmdec_error)
    tol = 1e-10
    # Test 1 - Icrs to Gal: conversion w.o.  covarience matrix
    GC = transformIcrsToGal(icrsCoords,icrsErrors)
    Icrs = transformGalToIcrs(GC[0],GC[1],GC[2])
    if(np.max(Icrs[0]-icrsCoords)>tol) : print('Error in test 1 - Coords')
    elif(np.max(Icrs[1]-icrsErrors)>tol) : print('Error in test 1 - Errors')
    elif(np.max(Icrs[2]-icrsCov)>tol) : print('Error in test 1 - Cov')
    else : print('Passed test 1')

    # Test 2 - Icrs to Gal: conversion w.  covarience matrix
    GC = transformIcrsToGal(icrsCoords,icrsErrors,icrsCov)
    Icrs = transformGalToIcrs(GC[0],GC[1],GC[2])
    if(np.max(Icrs[0]-icrsCoords)>tol) : print('Error in test 2 - Coords')
    elif(np.max(Icrs[1]-icrsErrors)>tol) : print('Error in test 2 - Errors')
    elif(np.max(Icrs[2]-icrsCov)>tol) : print('Error in test 2 - Cov')
    else : print('Passed test 2')

    # Test 3 - Icrs to Gal: conversion w.o. uncertainties (cov only)
    GC = transformIcrsToGal(icrsCoords,0,icrsCov)
    Icrs = transformGalToIcrs(GC[0],GC[1],GC[2])
    if(np.max(Icrs[0]-icrsCoords)>tol) : print('Error in test 3 - Coords')
    elif(np.max(Icrs[1]-icrsErrors)>tol) : print('Error in test 3 - Errors')
    elif(np.max(Icrs[2]-icrsCov)>tol) : print('Error in test 3 - Cov')
    else : print('Passed test 3')

    GalCov = CreateCovarianceMatrix_l_b(ra_error,dec_error,parallax_error,pmra_error,pmdec_error)

    # Test 4 - Gal to Icrs: conversion w.o.  covarience matrix
    Icrs = transformGalToIcrs(GalCoords,GalErrors)
    GC = transformIcrsToGal(Icrs[0],Icrs[1],Icrs[2])
    if(np.max(GC[0]-GalCoords)>tol) : print('Error in test 4 - Coords')
    elif(np.max(GC[1]-GalErrors)>tol) : print('Error in test 4 - Errors')
    elif(np.max(GC[2]-GalCov)>tol) : print('Error in test 4 - Cov')
    else : print('Passed test 4')

    # Test 5 - Gal to Icrs: conversion w.  covarience matrix
    GalCov = CreateCovarianceMatrix_l_b(ra_error,dec_error,parallax_error,pmra_error,pmdec_error)
    Icrs = transformGalToIcrs(GalCoords,GalErrors,GalCov)
    GC = transformIcrsToGal(Icrs[0],Icrs[1],Icrs[2])
    if(np.max(GC[0]-GalCoords)>tol) : print('Error in test 5 - Coords')
    elif(np.max(GC[1]-GalErrors)>tol) : print('Error in test 5 - Errors')
    elif(np.max(GC[2]-GalCov)>tol) : print('Error in test 5 - Cov')
    else : print('Passed test 5')

    # Test 6 - Gal to Icrs: conversion w.  covarience matrix
    GalCov = CreateCovarianceMatrix_l_b(ra_error,dec_error,parallax_error,pmra_error,pmdec_error)
    Icrs = transformGalToIcrs(GalCoords,GalErrors)
    GC = transformIcrsToGal(Icrs[0],Icrs[1],Icrs[2])
    if(np.max(GC[0]-GalCoords)>tol) : print('Error in test 6 - Coords')
    elif(np.max(GC[1]-GalErrors)>tol) : print('Error in test 6 - Errors')
    elif(np.max(GC[2]-GalCov)>tol) : print('Error in test 6 - Cov')
    else : print('Passed test 6')

    # Test 7 & 8: test with non-zero covariances
    GalCov = CreateCovarianceMatrix_l_b(ra_error,dec_error,parallax_error,pmra_error,pmdec_error,
                                        lbCorr=0.7,pmlStarpmbCorr=np.array([-0.2,0.6]))

    Icrs = transformGalToIcrs(GalCoords,GalErrors,GalCov)
    GC = transformIcrsToGal(Icrs[0],Icrs[1],Icrs[2])
    if(np.max(GC[0]-GalCoords)>tol) : print('Error in test 7 - Coords')
    elif(np.max(GC[1]-GalErrors)>tol) : print('Error in test 7 - Errors')
    elif(np.max(GC[2]-GalCov)>tol) : print('Error in test 7 - Cov')
    else : print('Passed test 7')

    icrsCov = CreateCovarianceMatrix_ra_dec(ra_error,dec_error,parallax_error,pmra_error,pmdec_error,
                                           decparallaxCorr=np.array([0.3,0.7]), pmrapmdecCorr=-0.4)

    GC = transformIcrsToGal(icrsCoords,icrsErrors,icrsCov)
    Icrs = transformGalToIcrs(GC[0],GC[1],GC[2])
    if(np.max(Icrs[0]-icrsCoords)>tol) : print('Error in test 8 - Coords')
    elif(np.max(Icrs[1]-icrsErrors)>tol) : print('Error in test 8 - Errors')
    elif(np.max(Icrs[2]-icrsCov)>tol) : print('Error in test 8 - Cov')
    else : print('Passed test 8')

#testConversions()
