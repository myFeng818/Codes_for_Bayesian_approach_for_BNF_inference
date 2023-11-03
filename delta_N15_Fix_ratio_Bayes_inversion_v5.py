from ncload import ncload
import numpy as np
from netCDF4 import Dataset
import pandas as pd
import arviz as az
from scipy.interpolate import RectBivariateSpline
from sklearn.utils import resample
import pymc3 as pm
from osgeo import gdal
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
from scipy import interpolate
from sklearn.model_selection import train_test_split
from sklearn.ensemble._forest import _generate_unsampled_indices
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.model_selection import GridSearchCV
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

'''##################################################################################
######################################################################################
######################  Coupled Bayesian approach and MC method   ############################
######################     for BNF inference based on d15N maps  ############################  
####################################################################################
       
Finished by Maoyuan Feng (fengmy@pku.edu.cn)
Contact: "Shushi Peng (speng@pku.edu.cn), Maoyuan Feng (fengmy@pku.edu.cn)"
'''

# Subroutine to change the spatial resolution of DATA0
def Resolution_change(lat0,lon0,lat_n,lon_n,DATA0):
    length_lat0 = len(lat0)
    length_lon0 = len(lon0)
    length_latn = len(lat_n)
    length_lonn = len(lon_n)

    lat_range = np.round(length_lat0/length_latn)
    lon_range = np.round(length_lon0/length_lonn)

    DATA_n = np.zeros((len(lat_n),len(lon_n)))

    for ilat in range(0,len(lat_n)):
        for ilon in range(0,len(lon_n)):
            lat_start = int(ilat*lat_range)
            lat_end = int(np.minimum((ilat+1)*lat_range,length_lat0))
            lon_start = int(ilon*lon_range)
            lon_end = int(np.minimum((ilon+1)*lon_range,length_lon0))
            value_mat = DATA0[lat_start:lat_end,lon_start:lon_end]

            if np.nanmean(value_mat)==0:
                DATA_n[ilat,ilon] = np.nan
            else:
                DATA_n[ilat,ilon] = np.nanmean(value_mat)
    return DATA_n

# Subroutine for calculating the d15N_plant based on input variables
def Hobs_fun(xprior,input_vars):
    '''

    :param xprior:
           xprior[0]: f_fix
           xprior[1]: eps_uptake
    :param input_vars
           input_vars[0]: delta_soil
           input_vars[1]: delta_fix
    :return: delta_plant
    '''

    delta_soil = input_vars[0]
    delta_fix = input_vars[1]

    delta_dep = 1.5

    delta_plant = (1-xprior[0]-xprior[2])*(delta_soil-xprior[1]) + xprior[0]*(delta_fix) + xprior[2]*(delta_dep)

    return delta_plant

# Subroutine of Bayesian inversion approach
def Bayes_inversion(Nobs,Nen,xprior,input_vars,yobs,sigma_obs):
    '''
    This is a Bayes inversion function to estimate
    the best parameters that can maximize a posterior probability.
    Essentlally, when the error probability is assumed to be Gaussian distributed,
    the Bayes inversion is exactly the same as EnKF.

    The parameters are as follows:
    :param Npara: number of parameters to be optimized
    :param Nen: number of ensembles used
    :param xprior: prior distribution of parameters experessed in Ensemble, Nen*1
    :param input_vars: input_vars to calculate the predicted observation
    :param yobs: exact observations for different quantities, can be multiple
    :param sigma_obs_factor: factor to determine the sigma of observation error
    :return: xpost: post distribution of parameters expressed in Ensemble
    '''

    Rmeasurement = np.zeros((Nobs,Nobs))
    yobs_ensemble = np.zeros((Nobs,Nen))
    for ii in range(0,Nobs):
        Rmeasurement[ii,ii] = np.abs((input_vars[2]*sigma_obs[0])**2 + ((1-input_vars[2])*sigma_obs[1])**2)
        yobs_ensemble[ii,:] = yobs[ii] + input_vars[2]*np.random.normal(0,1,[1,Nen])*sigma_obs[0] + (1-input_vars[2])*np.random.normal(0,1,[1,Nen])*sigma_obs[1]
    # Dimension: Nobs*Nen
    ypredicted = Hobs_fun(xprior,input_vars)
    ypredicted = np.transpose(ypredicted[:,np.newaxis],(1,0))
    # Dimension: Nobs*Nen
    D = yobs_ensemble - ypredicted

    # Dimension: Nobs*Nen
    Y = ypredicted - (np.nanmean(ypredicted,axis=1))[:,np.newaxis]
    # Dimension: Nobs*Nobs
    COVyy = np.dot(Y,np.transpose(Y,(1,0)))/(Nen-1)
    # Dimension: Npara*Nen
    X = xprior - (np.nanmean(xprior,axis=1))[:,np.newaxis]
    # Dimension: Npara*Nobs
    COVxy = np.dot(X,np.transpose(Y,(1,0)))/(Nen-1)
    # Dimension: Npara*Nobs
    KK = np.dot(COVxy,np.linalg.inv(COVyy+Rmeasurement))
    # Dimension: Npara*Nen
    xpost = xprior + np.dot(KK,D)

    return xpost,D

# Set the longitude and latitude used for the d15N global maps
lon = np.arange(-180,180,0.1)
lat = np.arange(-56,84,0.1)

# Load the desert mask global map
desert_file = ncload('Dersert_percent.nc')
desert_value = desert_file.get('new_data')

new_data = np.array(desert_value[:])
mask_desert = (new_data>=85)

global_data_path = '***************'
GPP_file = '**********'
Leaf_AF_file = '***********'
Wood_AF_file = '***********'
Root_AF_file = '***********'

GPP_th_sources_nc = ncload(global_data_path+GPP_file)
GPP_Keenan_mean,GPP_Keenan_std = GPP_th_sources_nc.get('GPP_Keenan','GPP_Keenan_std')
GPP_Keenan_mean_val = np.array(GPP_Keenan_mean[:].filled(0))
GPP_Keenan_std_val = np.array(GPP_Keenan_std[:].filled(0))
GPP_Keenan_mean_val[GPP_Keenan_mean_val!=GPP_Keenan_mean_val] = 0.0
GPP_Keenan_std_val[GPP_Keenan_std_val!=GPP_Keenan_std_val] = 0.0

GPP_inter = np.array(GPP_Keenan_mean_val[:])
GPP_std_inter = np.array(GPP_Keenan_std_val[:])

# C allocation factors to leaf, wood, and root
Leaf_AF_data = ncload(global_data_path+Leaf_AF_file)
AF_leaf,AF_leaf_std = Leaf_AF_data.get('Mean','Standard_Deviation')
AF_leaf_inter = (np.array(AF_leaf[:]))[34:174]
AF_leaf_std_inter = np.array(AF_leaf_std[:])[34:174]

Wood_AF_data = ncload(global_data_path+Wood_AF_file)
AF_wood,AF_wood_std = Wood_AF_data.get('Mean','Standard_Deviation')
AF_wood_inter = np.array(AF_wood[:])[34:174]
AF_wood_std_inter = np.array(AF_wood_std[:])[34:174]

Root_AF_data = ncload(global_data_path+Root_AF_file)
AF_root,AF_root_std = Root_AF_data.get('Mean','Standard_Deviation')
AF_root_inter = np.array(AF_root[:])[34:174]
AF_root_std_inter = np.array(AF_root_std[:])[34:174]

# N resorption coefficient
NRE_data = ncload('NRE_mean_std.nc')
NRE_leaf_mean,NRE_leaf_std = NRE_data.get('NRE_mean','MRE_std')
NRE_leaf_inter = np.array(NRE_leaf_mean[:])[34:174]
NRE_leaf_std_inter = np.array(NRE_leaf_std[:])[34:174]

# AM, ECM, and Nfixing bacteria
file_name_list = ['am.nc','em.nc','nfix.nc']
var_name_list = ['am','em','nfix']
kk=0
for file_name in file_name_list:
    exec('file_p = ncload(global_data_path+"%s")'%file_name)
    exec('%s = file_p.get("%s")'%(var_name_list[kk],'tmp'))
    exec('%s_mat = np.array(%s[:].filled(np.nan))'%(var_name_list[kk],var_name_list[kk]))
#    exec('%s_vect = np.squeeze(%s_mat.reshape(-1,1))'%(var_name_list[kk],var_name_list[kk]))
    kk = kk+1

# PFT data
pft_file = '*********\\PFT_map_one_deg.nc'
pft_data = ncload(pft_file)
pft_frac = np.array(pft_data.get('pft_frac')[:].filled(np.nan))

lat_one_degree = np.arange(-90,90,1)
lon_one_degree = np.arange(-180,180,1)
pft_one_degree = np.zeros((15,180,360))
pft_one_degree = pft_frac[:]

# crop lands and pastural regions
pft_human_pft = pft_one_degree[10]+pft_one_degree[12]+pft_one_degree[13]+pft_one_degree[14]

# Fractions of symbiotic BNF to total BNF
S_bnf_percent1 = np.array([0.55,0.40,0.75,0.55,0.92,0.75,0.92,0.75,0.66,0.53,0.66,0.53,0.53,0.53])
pft_map_new = pft_one_degree[1:]
S_bnf_pert1_global = np.nansum(pft_map_new[:]*S_bnf_percent1[:,np.newaxis,np.newaxis],axis=0)/np.nansum(pft_map_new[:],axis=0)

# Leaf N concentration
lnc_file = ncload('LNC_mean.nc')
LNC = lnc_file.get('LNC')
LNC = np.array(LNC[:].filled(np.nan))

# Leaf C/N ratio calculated from the leaf N concentration
CNratio_lnc = np.where(LNC==0,np.nan,450/LNC)

# Convertion coefficient from ORCHIDEE model for different PFTs
fcn_root = np.array([1.0, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86, 0.86])
fcn_wood = np.array([1.0, 0.108, 0.110, 0.134, 0.128, 0.110, 0.134, 0.110, 0.077, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

fcn_wood_mat = np.nansum(fcn_wood[1:,np.newaxis,np.newaxis]*pft_map_new[:],axis=0)/np.nansum(pft_map_new[:],axis=0)
fcn_root_mat = np.nansum(fcn_root[1:,np.newaxis,np.newaxis]*pft_map_new[:],axis=0)/np.nansum(pft_map_new[:],axis=0)

# Calculate the C/N ratio of wood and root
LEAF_CNratio=CNratio_lnc[:]
WOOD_CNratio = LEAF_CNratio[:]/fcn_wood_mat[34:174]
ROOT_CNratio = LEAF_CNratio[:]/fcn_root_mat[34:174]

# Load the global maps of d15N of Plant, Soil, and their difference
RF_foliar_file = ncload('RF_foliar_N15.nc')
RF_N15_foliar,RF_N15_foliar_std = RF_foliar_file.get('N15_foliar','N15_foliar_std')
RF_N15_foliar_mat =np.transpose(RF_N15_foliar[:].filled(np.nan),(1,0))
RF_N15_foliar_std_mat = np.transpose(RF_N15_foliar_std[:].filled(np.nan),(1,0))

RF_soil_file = ncload('RF_soil_N15.nc')
RF_N15_soil = RF_soil_file.get('N15_soil')
RF_N15_soil_mat =np.transpose(RF_N15_soil[:].filled(np.nan),(1,0))

RF_foliar_soil_file = ncload('RF_foliar_soil_N15.nc')
RF_N15_foliar_soil,RF_N15_foliar_soil_std = RF_foliar_soil_file.get('N15_foliar_soil','N15_foliar_soil_std')
RF_N15_foliar_soil_mat =np.transpose(RF_N15_foliar_soil[:].filled(np.nan),(1,0))

RF_N15_foliar_soil_std_mat = np.transpose(RF_N15_foliar_soil_std[:].filled(np.nan),(1,0))

# the longitude and latitude used in the Bayes approach
lon_new = np.arange(-180,180,1)
lat_new = np.arange(-56,84,1)
LON,LAT=np.meshgrid(lon_new,lat_new)

# Change the spatial resolution of d15N (0.1 by 0.1 degree) into 1 by 1 degree,
# using the subroutine defined at the beginning of the code
N15_foliar_new = Resolution_change(lat0=lat,lon0=lon,lat_n=lat_new,lon_n=lon_new,DATA0=RF_N15_foliar_mat[:])
N15_soil_new = Resolution_change(lat0=lat,lon0=lon,lat_n=lat_new,lon_n=lon_new,DATA0=RF_N15_soil_mat[:])
N15_foliar_soil_new = Resolution_change(lat0=lat,lon0=lon,lat_n=lat_new,lon_n=lon_new,DATA0=RF_N15_foliar_soil_mat[:])

N15_foliar_std_new = Resolution_change(lat0=lat,lon0=lon,lat_n=lat_new,lon_n=lon_new,DATA0=RF_N15_foliar_std_mat[:])
N15_foliar_soil_std_new = Resolution_change(lat0=lat,lon0=lon,lat_n=lat_new,lon_n=lon_new,DATA0=RF_N15_foliar_soil_std_mat[:])

# Mask out the grids with zero GPPs
N15_foliar_new[GPP_inter==0] = np.nan
N15_soil_new[GPP_inter==0] = np.nan
N15_foliar_soil_new[GPP_inter==0] = np.nan

N15_foliar_std_new[GPP_inter==0] = np.nan
N15_foliar_soil_std_new[GPP_inter==0] = np.nan

S_bnf_pert1_mat = S_bnf_pert1_global[34:174]
S_bnf_pert1_mat[GPP_inter==0] = np.nan

# Load the prior estimate of isotope fractionation of plant uptake, eps_uptake
eps_uptake_fromFUN = ncload('eps_uptake_fromFUN_v1.nc')
eps_uptake_FUN_data = eps_uptake_fromFUN.get('eps_uptake')
eps_uptake_FUN_mat = np.transpose(np.array(eps_uptake_FUN_data[:].filled(np.nan)),(1,0))


for fix_mean in [2]:
    for iteration in range(0,14):
        if iteration >=1:
            # Set up the .nc file to store the results for the i-th iteration
            fix_ratio_file = ncload('Keenaneps_test1_Fix_and_uptake_test_withMC_%sth.nc'%(iteration-1))
            fix_ratio_data,fix_ratio_std_data,eps_uptake_data,eps_uptake_std_data,frac_dep_data,frac_dep_std_data\
                = fix_ratio_file.get('fix_ratio_org','fix_ratio_std_org','eps_uptake_org','eps_uptake_std_org','frac_dep','frac_dep_std')
            fix_ratio_new = np.array(fix_ratio_data[:].filled(np.nan))
            fix_std_new = np.array(fix_ratio_std_data[:].filled(np.nan))

            # when the SD of f_sBNF is higher than 0.08, it is set as 0.08,
            # which is used to control the variability to be within an acceptable range
            fix_std_new = np.where(fix_std_new>0.08,0.08,fix_std_new)

            eps_uptake_new = np.array(eps_uptake_data[:].filled(np.nan))
            eps_uptake_std_new = np.array(eps_uptake_std_data[:].filled(np.nan))

            frac_dep_new = np.array(frac_dep_data[:].filled(np.nan))
            frac_dep_std_new = np.array(frac_dep_std_data[:].filled(np.nan))
        else:

            # load the prior estimates from Monte Carlo method
            fix_ratio_file = ncload('Keenaneps_test1_Fix_and_uptake_test_MC.nc')
            fix_ratio_data,fix_ratio_std_data,eps_uptake_data,eps_uptake_std_data,frac_dep_data,frac_dep_std_data\
                = fix_ratio_file.get('fix_ratio','fix_ratio_std','eps_uptake','eps_uptake_std','frac_dep','frac_dep_std')
            fix_ratio_new = np.array(fix_ratio_data[:].filled(np.nan))
            fix_std_new = np.array(fix_ratio_std_data[:].filled(np.nan))
            eps_uptake_new = np.array(eps_uptake_data[:].filled(np.nan))
            #eps_uptake_std_new = 0.3*eps_uptake_new
            eps_uptake_std_new = np.array(eps_uptake_std_data[:].filled(np.nan))

            frac_dep_new = np.array(frac_dep_data[:].filled(np.nan))
            frac_dep_std_new = np.array(frac_dep_std_data[:].filled(np.nan))

#####################################################################################################
###################################  Initialization    ##############################################
################################### of all variables   ##############################################
#####################################################################################################

        # Change all the global maps into vectors
        N15_foliar_vect = np.squeeze(N15_foliar_new.reshape(-1,1))
        N15_soil_vect = np.squeeze(N15_soil_new.reshape(-1,1))
        N15_foliar_soil_vect = np.squeeze(N15_foliar_soil_new.reshape(-1,1))
        N15_foliar_std_vect = np.squeeze(N15_foliar_std_new.reshape(-1,1))
        N15_foliar_soil_std_vect = np.squeeze(N15_foliar_soil_std_new.reshape(-1,1))

        S_bnf_pert1_vect = np.squeeze(S_bnf_pert1_mat.reshape(-1,1))
        GPP_vect = np.squeeze(GPP_inter.reshape(-1,1))
        AF_leaf_vect = np.squeeze(AF_leaf_inter.reshape(-1,1))
        AF_wood_vect = np.squeeze(AF_wood_inter.reshape(-1,1))
        AF_root_vect = np.squeeze(AF_root_inter.reshape(-1,1))
        LEAF_CNratio_vect = np.squeeze(LEAF_CNratio.reshape(-1,1))
        WOOD_CNratio_vect = np.squeeze(WOOD_CNratio.reshape(-1,1))
        ROOT_CNratio_vect = np.squeeze(ROOT_CNratio.reshape(-1,1))
        NRE_leaf_vect = np.squeeze(NRE_leaf_inter.reshape(-1,1))
        NRE_root_vect = NRE_leaf_vect - NRE_leaf_vect + 0.275
        GPP_std_vect = np.squeeze(GPP_std_inter.reshape(-1,1))
        AF_leaf_std_vect = np.squeeze(AF_leaf_std_inter.reshape(-1,1))
        AF_wood_std_vect = np.squeeze(AF_wood_std_inter.reshape(-1,1))
        AF_root_std_vect = np.squeeze(AF_root_std_inter.reshape(-1,1))
        LEAF_CNratio_std_vect = 0.05*LEAF_CNratio_vect
        WOOD_CNratio_std_vect = 0.05*WOOD_CNratio_vect
        ROOT_CNratio_std_vect = 0.05*ROOT_CNratio_vect
        NRE_leaf_std_vect = np.squeeze(NRE_leaf_std_inter.reshape(-1,1))
        NRE_root_std_vect = 0.05*NRE_root_vect
        N15_soil_used_vect = N15_soil_vect
        N15_foliar_used_vect = N15_soil_vect + N15_foliar_soil_vect
        mask_true = N15_soil_used_vect==N15_soil_used_vect
        mask_not_true = np.logical_not(mask_true)
        N15_soil_to_bayes = np.array(N15_soil_used_vect[mask_true])
        N15_foliar_to_bayes = np.array(N15_foliar_used_vect[mask_true])
        N15_foliar_std_to_bayes = np.array(N15_foliar_std_vect[mask_true])
        N15_soil_std_to_bayes = np.array(N15_foliar_soil_std_vect[mask_true])

        fix_ratio_new[GPP_inter==0] = np.nan
        fix_std_new[GPP_inter==0] = np.nan

        FIX_ratio_vect = np.squeeze(fix_ratio_new.reshape(-1,1))
        FIX_std_vect = np.squeeze(fix_std_new.reshape(-1,1))

        FIX_ratio_to_beyes = np.array(FIX_ratio_vect[mask_true])
        FIX_std_to_bayes = np.array(FIX_std_vect[mask_true])

        eps_uptake_vect = np.squeeze(eps_uptake_new.reshape(-1,1))
        eps_uptake_std_vect = np.squeeze(eps_uptake_std_new.reshape(-1,1))

        frac_dep_new[GPP_inter==0] = np.nan
        frac_dep_std_new[GPP_inter==0] = np.nan

        frac_dep_vect = np.squeeze(frac_dep_new.reshape(-1,1))
        frac_dep_std_vect = np.squeeze(frac_dep_std_new.reshape(-1,1))

        frac_dep_to_bayes = np.array(frac_dep_vect[mask_true])
        frac_dep_std_to_bayes = np.array(frac_dep_std_vect[mask_true])

        S_bnf_pert1_to_bayes = np.array(S_bnf_pert1_vect[mask_true])
        GPP_to_bayes = np.array(GPP_vect[mask_true])
        AF_leaf_to_bayes = np.array(AF_leaf_vect[mask_true])
        AF_wood_to_bayes = np.array(AF_wood_vect[mask_true])
        AF_root_to_bayes = np.array(AF_root_vect[mask_true])
        LEAF_CNratio_to_bayes = np.array(LEAF_CNratio_vect[mask_true])
        WOOD_CNratio_to_bayes = np.array(WOOD_CNratio_vect[mask_true])
        ROOT_CNratio_to_bayes = np.array(ROOT_CNratio_vect[mask_true])
        NRE_leaf_to_bayes = np.array(NRE_leaf_vect[mask_true])
        NRE_root_to_bayes = np.array(NRE_root_vect[mask_true])
        GPP_std_to_bayes = np.array(GPP_std_vect[mask_true])
        AF_leaf_std_to_bayes = np.array(AF_leaf_std_vect[mask_true])
        AF_wood_std_to_bayes = np.array(AF_wood_std_vect[mask_true])
        AF_root_std_to_bayes = np.array(AF_root_std_vect[mask_true])
        LEAF_CNratio_std_to_bayes = np.array(LEAF_CNratio_std_vect[mask_true])
        WOOD_CNratio_std_to_bayes = np.array(WOOD_CNratio_std_vect[mask_true])
        ROOT_CNratio_std_to_bayes = np.array(ROOT_CNratio_std_vect[mask_true])
        NRE_leaf_std_to_bayes = np.array(NRE_leaf_std_vect[mask_true])
        NRE_root_std_to_bayes = np.array(NRE_root_std_vect[mask_true])

        # set all initial values as zero

        fix_ratio_global = N15_foliar_vect - N15_foliar_vect
        fix_ratio_std_global = N15_foliar_vect - N15_foliar_vect
        eps_uptake_global = N15_foliar_vect - N15_foliar_vect
        eps_uptake_std_global = N15_foliar_vect - N15_foliar_vect
        frac_dep_global = N15_foliar_vect - N15_foliar_vect
        frac_dep_std_global = N15_foliar_vect - N15_foliar_vect
        fix_ratio_global_org = N15_foliar_vect - N15_foliar_vect
        fix_ratio_std_global_org = N15_foliar_vect - N15_foliar_vect
        eps_uptake_global_org = N15_foliar_vect - N15_foliar_vect
        eps_uptake_std_global_org = N15_foliar_vect - N15_foliar_vect

        fix_ratio_lowq_global = N15_foliar_vect - N15_foliar_vect
        fix_ratio_midq_global = N15_foliar_vect - N15_foliar_vect
        fix_ratio_uppq_global = N15_foliar_vect - N15_foliar_vect
        eps_uptake_lowq_global = N15_foliar_vect - N15_foliar_vect
        eps_uptake_midq_global = N15_foliar_vect - N15_foliar_vect
        eps_uptake_uppq_global = N15_foliar_vect - N15_foliar_vect
        fffix_ratio_global = N15_foliar_vect - N15_foliar_vect
        fffix_ratio_std_global = N15_foliar_vect - N15_foliar_vect
        fffix_ratio_lowq_global = N15_foliar_vect - N15_foliar_vect
        fffix_ratio_midq_global = N15_foliar_vect - N15_foliar_vect
        fffix_ratio_uppq_global = N15_foliar_vect - N15_foliar_vect
        SBNF_lowq_global = N15_foliar_vect - N15_foliar_vect
        SBNF_midq_global = N15_foliar_vect - N15_foliar_vect
        SBNF_uppq_global = N15_foliar_vect - N15_foliar_vect
        SBNF_mean_global = N15_foliar_vect - N15_foliar_vect
        SBNF_std_global = N15_foliar_vect - N15_foliar_vect
        TBNF_lowq_global = N15_foliar_vect - N15_foliar_vect
        TBNF_midq_global = N15_foliar_vect - N15_foliar_vect
        TBNF_uppq_global = N15_foliar_vect - N15_foliar_vect
        TBNF_mean_global = N15_foliar_vect - N15_foliar_vect
        TBNF_std_global = N15_foliar_vect - N15_foliar_vect

        eps_uptake_to_bayes = eps_uptake_vect[mask_true]
        eps_uptake_std_to_bayes = eps_uptake_std_vect[mask_true]

###########################################################################################################
###########################################################################################################
###########################################################################################################
        # Main part of the codes


        Nobs = 1 # Numb of observations
        Nen = 1000 # Numb of ensembles
        Ntimes = 10 # Repeat times

        fix_ratio_obtained_times = np.zeros((Ntimes,len(N15_soil_to_bayes),2))
        eps_uptake_obtained_times = np.zeros((Ntimes,len(N15_soil_to_bayes),2))
        frac_dep_obtained_times = np.zeros((Ntimes,len(N15_soil_to_bayes),2))

        fix_ratio_obtained_org_times = np.zeros((Ntimes,len(N15_soil_to_bayes),2))
        eps_uptake_obtained_org_times = np.zeros((Ntimes,len(N15_soil_to_bayes),2))

        fix_ratio_quantile_times = np.zeros((Ntimes,len(N15_soil_to_bayes),3))
        eps_uptake_quantile_times = np.zeros((Ntimes,len(N15_soil_to_bayes),3))
        fffix_ratio_quantile_times = np.zeros((Ntimes,len(N15_soil_to_bayes),5))

        SBNF_25_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        SBNF_50_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        SBNF_75_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        SBNF_mean_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        SBNF_std_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        TBNF_25_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        TBNF_50_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        TBNF_75_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        TBNF_mean_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))
        TBNF_std_times = np.zeros((Ntimes,len(N15_soil_to_bayes)))

        # For each iteration, the model is run by Ntimes
        for itimes in range(0,Ntimes):

            # Empty Variables to store the inversed key variables
            fix_ratio_obtained = np.zeros((len(N15_soil_to_bayes),2))
            eps_uptake_obtained = np.zeros((len(N15_soil_to_bayes),2))

            frac_dep_obtained = np.zeros((len(N15_soil_to_bayes),2))

            fix_ratio_obtained_org = np.zeros((len(N15_soil_to_bayes),2))
            eps_uptake_obtained_org = np.zeros((len(N15_soil_to_bayes),2))

            fix_ratio_quantile = np.zeros((len(N15_soil_to_bayes),3))
            eps_uptake_quantile = np.zeros((len(N15_soil_to_bayes),3))
            fffix_ratio_quantile = np.zeros((len(N15_soil_to_bayes),5))
            S_BNF_all_pft = np.zeros((len(N15_soil_to_bayes),Nen))
            T_BNF_all_pft = np.zeros((len(N15_soil_to_bayes),Nen))

            # We implement the Bayesian approach in each grid cell
            for igrid in range(0,len(N15_soil_to_bayes)):
                print(fix_mean,iteration,itimes,igrid)

                # INput variables of soil d15N, d15N of BNF, and f_BNFs
                input_vars = np.zeros((3,1))
                input_vars[0] = N15_soil_to_bayes[igrid]
                input_vars[1] = -2.02

                # Observation of plant d15N
                yobs = np.array([N15_foliar_to_bayes[igrid]])
                sigma_obs = np.zeros((2,1))
                sigma_obs[0] = np.maximum(N15_foliar_std_to_bayes[igrid],0.01)
                sigma_obs[1] = np.maximum(N15_soil_std_to_bayes[igrid],0.01)

                # Prior estimates
                xprior = np.zeros((3,Nen))
                xprior[0] = FIX_ratio_to_beyes[igrid] + FIX_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                xprior[1] = eps_uptake_to_bayes[igrid] + eps_uptake_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                xprior[2] = frac_dep_to_bayes[igrid] + frac_dep_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])

                input_vars[2] = np.nanmean(xprior[0])

                # Implementation of Bayesian approach
                xpost0,obs_variability = Bayes_inversion(Nobs,Nen,xprior,input_vars,yobs,sigma_obs)
                # Original posterior estimates

                # In the following, the original posterior estimates are corrected
                delta_soil = input_vars[0]
                delta_fix = input_vars[1]
                #eps_uptake_variability = xpost0[1] - np.nanmean(xpost0[1])

                eps_uptake_org_emsemble = np.array(xpost0[1])
                xpost = np.array(xpost0[:])

                # Recalculate the plant d15N based on the posterior estimates
                delta_dep = 1.5
                delta_plant_new = (1-xpost[0]-xpost[2])*(delta_soil-eps_uptake_org_emsemble) + xpost[0]*(delta_fix) + xpost[2]*(delta_dep)

                # Correct the posterior estimates for the violations of physical constraints
                mask_smaller = delta_plant_new<delta_fix
                if np.sum(mask_smaller[0])>=1:
                    xpost[0,mask_smaller[0]] = 0.0
                    xpost[1,mask_smaller[0]] = eps_uptake_org_emsemble[0,mask_smaller[0]]

                mask_large_fix = xpost[0]>=1.0
                if np.sum(mask_large_fix[0])>=1:
                    xpost[0,mask_large_fix[0]] = np.nan
                    xpost[1,mask_large_fix[0]] = np.nan

                mask_zero_fix = xpost[0]<0
                if np.sum(mask_zero_fix[0])>=1:
                    xpost[0,mask_zero_fix[0]] = 0.0
                    xpost[1,mask_zero_fix[0]] = eps_uptake_org_emsemble[0,mask_zero_fix[0]]

                mask_eps_negative = xpost[1]<0
                if np.sum(mask_eps_negative[0])>=1:
                    xpost[1,mask_eps_negative[0]]=np.nan
                    xpost[0,mask_eps_negative[0]]=np.nan

                mask_large_fix = xpost[0]>=1.0
                if np.sum(mask_large_fix[0])>=1:
                    xpost[0,mask_large_fix[0]] = np.nan
                    xpost[1,mask_large_fix[0]] = np.nan

                # For the corrected emsembles, there are a lot of NaN values, we choose to resample
                # these values from the remaining ensembles using a bootstrap strategy
                mask_truevalue = (xpost[0] == xpost[0])
                xpost_fix_original_sample_true = xpost[0, mask_truevalue]
                xpost_eps_original_sample_true = xpost[1, mask_truevalue]
                if len(xpost_fix_original_sample_true) == 0:
                    xpost[0] = xprior[0]
                    xpost[1] = xprior[1]
                elif len(xpost_fix_original_sample_true) < Nen:
                    Bootstrap_samples_fix = resample(xpost_fix_original_sample_true,
                                                     n_samples=Nen - len(xpost_fix_original_sample_true), replace=True)
                    Bootstrap_samples_eps = resample(xpost_eps_original_sample_true,
                                                     n_samples=Nen - len(xpost_fix_original_sample_true), replace=True)

                    xpost[0] = np.where(xpost[0] != xpost[0], Bootstrap_samples_fix, xpost[0])
                    xpost[1] = np.where(xpost[1] != xpost[1], Bootstrap_samples_eps, xpost[1])

                # Remove ensembles that are not reasonable
                xpost[1] = np.where(xpost[1]>20,20.0,xpost[1])
                xpost[2] = np.where(xpost[2]>0.25,0.25,xpost[2])
                xpost[2] = np.where(xpost[2]<0.00,0.00,xpost[2])

                # In the following, we are going to calculate the N demands, and further BNFs
                S_pert1_ensemble = S_bnf_pert1_to_bayes[igrid] + 0.1*S_bnf_pert1_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                S_pert1_ensemble = np.where(S_pert1_ensemble>1,1.0,S_pert1_ensemble)
                S_pert1_ensemble = np.where(S_pert1_ensemble<0.001,0.001,S_pert1_ensemble)
                mask_positive = xpost[0]>=0
                xpost_original_sample = xpost[0,mask_positive]

                # Implementation of Monte Carlo method
                if len(xpost_original_sample)==0:
                    Bootstrap_samples_fix = np.zeros((Nen,))
                else:
                    # Bootstrap strategy to resample f_BNFs or just disturb the orders of ensembles
                    # This will not change the results of BNF estimates because all other variables
                    # (AFs, GPPs, NREs) related to N demand estimates are newly created randomly.
                    Bootstrap_samples_fix = resample(xpost_original_sample,n_samples=Nen,replace=True)

                # Calculate the Ndemand from GPP, AFs, NRE, and others
                GPP_ensemble = GPP_to_bayes[igrid] + GPP_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                AF_leaf_ensemble = AF_leaf_to_bayes[igrid] + AF_leaf_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                AF_wood_ensemble = AF_wood_to_bayes[igrid] + AF_wood_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                AF_root_ensemble = AF_root_to_bayes[igrid] + AF_root_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])

                LEAF_CNratio_ensemble = LEAF_CNratio_to_bayes[igrid] + LEAF_CNratio_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                WOOD_CNratio_ensemble = WOOD_CNratio_to_bayes[igrid] + WOOD_CNratio_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                ROOT_CNratio_ensemble = ROOT_CNratio_to_bayes[igrid] + ROOT_CNratio_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])

                NRE_leaf_ensemble = NRE_leaf_to_bayes[igrid] + NRE_leaf_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])
                NRE_root_ensemble = NRE_root_to_bayes[igrid] + NRE_root_std_to_bayes[igrid]*np.random.normal(0,1,[1,Nen])

                leaf_factor = np.where(LEAF_CNratio_ensemble==0,0.0,AF_leaf_ensemble/LEAF_CNratio_ensemble)
                wood_factor = np.where(WOOD_CNratio_ensemble==0,0.0,AF_wood_ensemble/WOOD_CNratio_ensemble)
                root_factor = np.where(ROOT_CNratio_ensemble==0,0.0,AF_root_ensemble/ROOT_CNratio_ensemble)

                Ndemand_ensemble = GPP_ensemble*365.0*(leaf_factor*(1-NRE_leaf_ensemble/100)+\
                                                 wood_factor+\
                                                 root_factor*(1-NRE_root_ensemble/100))
                # Estimate the symbiotic BNF for each grid
                S_BNF_all_pft[igrid] = Ndemand_ensemble*Bootstrap_samples_fix

                # Estimate the total BNF for each grid
                fffix_ratio_new = xpost[0]/S_pert1_ensemble
                fffix_ratio_new = np.where(fffix_ratio_new>1,1.0,fffix_ratio_new)
                fffix_original_sample = fffix_ratio_new[0,mask_positive]

                if len(fffix_original_sample)==0:
                    Bootstrap_samples_fffix = np.zeros((Nen,))
                else:
                    Bootstrap_samples_fffix = resample(fffix_original_sample,n_samples=Nen,replace=True)

                T_BNF_all_pft[igrid] = Ndemand_ensemble*Bootstrap_samples_fffix
                xpost_mean = np.nanmean(xpost[:,mask_positive],axis=1)  # estimate the mean of posterior variables
                xpost_std = np.power(np.nanvar(xpost[:,mask_positive],axis=1),0.5) # estimate the SD of posterior variables

                xpost0_mean = np.nanmean(xpost0,axis=1) # estimate the mean of posterior from original ensembles
                xpost0_std = np.power(np.nanvar(xpost0,axis=1),0.5) # estimate the SD of posterior from original ensembles

                # Calculate the quantiles
                fffix_ratio_quantile[igrid,0] = np.nanmean(fffix_ratio_new[:,mask_positive],axis=1)
                fffix_ratio_quantile[igrid,1] = np.nanvar(fffix_ratio_new[:,mask_positive],axis=1)**0.5
                fffix_ratio_quantile[igrid,2] = np.nanquantile(fffix_ratio_new[:,mask_positive],0.025)
                fffix_ratio_quantile[igrid,3] = np.nanquantile(fffix_ratio_new[:,mask_positive],0.50)
                fffix_ratio_quantile[igrid,4] = np.nanquantile(fffix_ratio_new[:,mask_positive],0.975)

                # First storage of corrected posterior
                fix_ratio_obtained[igrid,0] = xpost_mean[0]
                fix_ratio_obtained[igrid,1] = xpost_std[0]
                eps_uptake_obtained[igrid,0] = xpost_mean[1]
                eps_uptake_obtained[igrid,1] = xpost_std[1]

                frac_dep_obtained[igrid,0] = xpost_mean[2]
                frac_dep_obtained[igrid,1] = xpost_std[2]

                # First storage of original posterior
                fix_ratio_obtained_org[igrid,0] = xpost0_mean[0]
                fix_ratio_obtained_org[igrid,1] = xpost0_std[0]
                eps_uptake_obtained_org[igrid,0] = xpost0_mean[1]
                eps_uptake_obtained_org[igrid,1] = xpost0_std[1]

                fix_ratio_quantile[igrid,0] = np.nanquantile(xpost[0,mask_positive],0.025)
                fix_ratio_quantile[igrid,1] = np.nanquantile(xpost[0,mask_positive],0.50)
                fix_ratio_quantile[igrid,2] = np.nanquantile(xpost[0,mask_positive],0.975)
                eps_uptake_quantile[igrid,0] = np.nanquantile(xpost[1,mask_positive],0.025)
                eps_uptake_quantile[igrid,1] = np.nanquantile(xpost[1,mask_positive],0.50)
                eps_uptake_quantile[igrid,2] = np.nanquantile(xpost[1,mask_positive],0.975)

            # Second storage of all variables: Ntimes
            S_BNF_all_pft_sum = np.nansum(S_BNF_all_pft,axis=0)
            SBNF_quants25 = np.nanquantile(S_BNF_all_pft_sum,0.025,interpolation='lower')
            SBNF_quants50 = np.nanquantile(S_BNF_all_pft_sum,0.50,interpolation='lower')
            SBNF_quants75 = np.nanquantile(S_BNF_all_pft_sum,0.975,interpolation='lower')

            SBNF_25_times[itimes]= np.squeeze(S_BNF_all_pft[:,S_BNF_all_pft_sum==SBNF_quants25])
            SBNF_50_times[itimes]= np.squeeze(S_BNF_all_pft[:,S_BNF_all_pft_sum==SBNF_quants50])
            SBNF_75_times[itimes]= np.squeeze(S_BNF_all_pft[:,S_BNF_all_pft_sum==SBNF_quants75])

            SBNF_mean_times[itimes] = np.nanmean(S_BNF_all_pft,axis=1)
            SBNF_std_times[itimes] = np.nanvar(S_BNF_all_pft,axis=1)**0.5

            T_BNF_all_pft_sum = np.nansum(T_BNF_all_pft,axis=0)
            TBNF_quants25 = np.nanquantile(T_BNF_all_pft_sum,0.025,interpolation='lower')
            TBNF_quants50 = np.nanquantile(T_BNF_all_pft_sum,0.50,interpolation='lower')
            TBNF_quants75 = np.nanquantile(T_BNF_all_pft_sum,0.975,interpolation='lower')
            TBNF_25_times[itimes]= np.squeeze(T_BNF_all_pft[:,T_BNF_all_pft_sum==TBNF_quants25])
            TBNF_50_times[itimes]= np.squeeze(T_BNF_all_pft[:,T_BNF_all_pft_sum==TBNF_quants50])
            TBNF_75_times[itimes]= np.squeeze(T_BNF_all_pft[:,T_BNF_all_pft_sum==TBNF_quants75])
            TBNF_mean_times[itimes] = np.nanmean(T_BNF_all_pft,axis=1)
            TBNF_std_times[itimes] = np.nanvar(T_BNF_all_pft,axis=1)**0.5

            fix_ratio_obtained_times[itimes] = fix_ratio_obtained
            eps_uptake_obtained_times[itimes] = eps_uptake_obtained

            frac_dep_obtained_times[itimes] = frac_dep_obtained

            fix_ratio_obtained_org_times[itimes] = fix_ratio_obtained_org
            eps_uptake_obtained_org_times[itimes] = eps_uptake_obtained_org

            fix_ratio_quantile_times[itimes] = fix_ratio_quantile
            eps_uptake_quantile_times[itimes] = eps_uptake_quantile
            fffix_ratio_quantile_times[itimes] = fffix_ratio_quantile

        # Final storage of all variables, by taking the mean of N times
        fix_ratio_obtained = np.nanmean(fix_ratio_obtained_times,axis=0)
        eps_uptake_obtained = np.nanmean(eps_uptake_obtained_times,axis=0)
        frac_dep_obtained = np.nanmean(frac_dep_obtained_times,axis=0)

        fix_ratio_obtained_org = np.nanmean(fix_ratio_obtained_org_times,axis=0)
        eps_uptake_obtained_org = np.nanmean(eps_uptake_obtained_org_times,axis=0)

        fix_ratio_quantile = np.nanmean(fix_ratio_quantile_times,axis=0)
        eps_uptake_quantile = np.nanmean(eps_uptake_quantile_times,axis=0)
        fffix_ratio_quantile =  np.nanmean(fffix_ratio_quantile_times,axis=0)

        SBNF_final_quants25 = np.nanmean(SBNF_25_times,axis=0)
        SBNF_final_quants50 = np.nanmean(SBNF_50_times,axis=0)
        SBNF_final_quants75 = np.nanmean(SBNF_75_times,axis=0)
        SBNF_final_mean = np.nanmean(SBNF_mean_times,axis=0)
        SBNF_final_std = np.nanmean(SBNF_std_times,axis=0)

        TBNF_final_quants25 = np.nanmean(TBNF_25_times,axis=0)
        TBNF_final_quants50 = np.nanmean(TBNF_50_times,axis=0)
        TBNF_final_quants75 = np.nanmean(TBNF_75_times,axis=0)
        TBNF_final_mean = np.nanmean(TBNF_mean_times,axis=0)
        TBNF_final_std = np.nanmean(TBNF_std_times,axis=0)

        ### Store the calculated variables into the originally defined empty variables
        fix_ratio_global[mask_true] = fix_ratio_obtained[:,0]
        fix_ratio_std_global[mask_true] = fix_ratio_obtained[:,1]
        eps_uptake_global[mask_true] = eps_uptake_obtained[:,0]
        eps_uptake_std_global[mask_true] = eps_uptake_obtained[:,1]

        frac_dep_global[mask_true] = frac_dep_obtained[:,0]
        frac_dep_std_global[mask_true] = frac_dep_obtained[:,1]

        fix_ratio_global_org[mask_true] = fix_ratio_obtained_org[:,0]
        fix_ratio_std_global_org[mask_true] = fix_ratio_obtained_org[:,1]
        eps_uptake_global_org[mask_true] = eps_uptake_obtained_org[:,0]
        eps_uptake_std_global_org[mask_true] = eps_uptake_obtained_org[:,1]

        fix_ratio_lowq_global[mask_true] = fix_ratio_quantile[:,0]
        fix_ratio_midq_global[mask_true] = fix_ratio_quantile[:,1]
        fix_ratio_uppq_global[mask_true] = fix_ratio_quantile[:,2]
        eps_uptake_lowq_global[mask_true] = eps_uptake_quantile[:,0]
        eps_uptake_midq_global[mask_true] = eps_uptake_quantile[:,1]
        eps_uptake_uppq_global[mask_true] = eps_uptake_quantile[:,2]

        fffix_ratio_global[mask_true] = fffix_ratio_quantile[:,0]
        fffix_ratio_std_global[mask_true] = fffix_ratio_quantile[:,1]
        fffix_ratio_lowq_global[mask_true] = fffix_ratio_quantile[:,2]
        fffix_ratio_midq_global[mask_true] = fffix_ratio_quantile[:,3]
        fffix_ratio_uppq_global[mask_true] = fffix_ratio_quantile[:,4]

        SBNF_lowq_global[mask_true] = SBNF_final_quants25
        SBNF_midq_global[mask_true] = SBNF_final_quants50
        SBNF_uppq_global[mask_true] = SBNF_final_quants75
        SBNF_mean_global[mask_true] = SBNF_final_mean
        SBNF_std_global[mask_true] = SBNF_final_std

        TBNF_lowq_global[mask_true] = TBNF_final_quants25
        TBNF_midq_global[mask_true] = TBNF_final_quants50
        TBNF_uppq_global[mask_true] = TBNF_final_quants75
        TBNF_mean_global[mask_true] = TBNF_final_mean
        TBNF_std_global[mask_true] = TBNF_final_std

        # Reshape the vectors into global maps
        fix_ratio_global_mat = fix_ratio_global.reshape(140,-1)
        fix_ratio_std_global_mat = fix_ratio_std_global.reshape(140,-1)
        eps_uptake_global_mat = eps_uptake_global.reshape(140,-1)
        eps_uptake_std_global_mat = eps_uptake_std_global.reshape(140,-1)

        frac_dep_global_mat = frac_dep_global.reshape(140,-1)
        frac_dep_std_global_mat = frac_dep_std_global.reshape(140,-1)

        fix_ratio_global_org_mat = fix_ratio_global_org.reshape(140,-1)
        fix_ratio_std_global_org_mat = fix_ratio_std_global_org.reshape(140,-1)
        eps_uptake_global_org_mat = eps_uptake_global_org.reshape(140,-1)
        eps_uptake_std_global_org_mat = eps_uptake_std_global_org.reshape(140,-1)


        fix_ratio_lowq_global_mat = fix_ratio_lowq_global.reshape(140,-1)
        fix_ratio_midq_global_mat = fix_ratio_midq_global.reshape(140,-1)
        fix_ratio_uppq_global_mat = fix_ratio_uppq_global.reshape(140,-1)

        eps_uptake_lowq_global_mat = eps_uptake_lowq_global.reshape(140,-1)
        eps_uptake_midq_global_mat = eps_uptake_midq_global.reshape(140,-1)
        eps_uptake_uppq_global_mat = eps_uptake_uppq_global.reshape(140,-1)

        fffix_ratio_global_mat = fffix_ratio_global.reshape(140,-1)
        fffix_ratio_std_global_mat = fffix_ratio_std_global.reshape(140,-1)
        fffix_ratio_lowq_global_mat = fffix_ratio_lowq_global.reshape(140,-1)
        fffix_ratio_midq_global_mat = fffix_ratio_midq_global.reshape(140,-1)
        fffix_ratio_uppq_global_mat = fffix_ratio_uppq_global.reshape(140,-1)

        SBNF_lowq_global_mat = SBNF_lowq_global.reshape(140,-1)
        SBNF_midq_global_mat = SBNF_midq_global.reshape(140,-1)
        SBNF_uppq_global_mat = SBNF_uppq_global.reshape(140,-1)
        SBNF_mean_global_mat = SBNF_mean_global.reshape(140,-1)
        SBNF_std_global_mat = SBNF_std_global.reshape(140,-1)

        TBNF_lowq_global_mat = TBNF_lowq_global.reshape(140,-1)
        TBNF_midq_global_mat = TBNF_midq_global.reshape(140,-1)
        TBNF_uppq_global_mat = TBNF_uppq_global.reshape(140,-1)
        TBNF_mean_global_mat = TBNF_mean_global.reshape(140,-1)
        TBNF_std_global_mat = TBNF_std_global.reshape(140,-1)

##########################################################################################
##############################  Data Saving     ##########################################
        fix_uptake_file = Dataset('Keenaneps_test1_Fix_and_uptake_test_withMC_%sth.nc'%(iteration),'w')
        fix_uptake_file.createDimension('lat',len(lat_new))
        fix_uptake_file.createDimension('lon',len(lon_new))

        FIX_RATIO = fix_uptake_file.createVariable('fix_ratio','d',('lat','lon'))
        FIX_RATIO_STD = fix_uptake_file.createVariable('fix_ratio_std','d',('lat','lon'))

        FIX_RATIO_org = fix_uptake_file.createVariable('fix_ratio_org','d',('lat','lon'))
        FIX_RATIO_STD_org = fix_uptake_file.createVariable('fix_ratio_std_org','d',('lat','lon'))

        FRAC_DEP = fix_uptake_file.createVariable('frac_dep','d',('lat','lon'))
        FRAC_DEP_STD = fix_uptake_file.createVariable('frac_dep_std','d',('lat','lon'))

        FIX_RATIO_LOWQ = fix_uptake_file.createVariable('fix_ratio_lowq','d',('lat','lon'))
        FIX_RATIO_MIDQ = fix_uptake_file.createVariable('fix_ratio_midq','d',('lat','lon'))
        FIX_RATIO_UPPQ = fix_uptake_file.createVariable('fix_ratio_uppq','d',('lat','lon'))

        FFFIX_RATIO = fix_uptake_file.createVariable('fffix_ratio','d',('lat','lon'))
        FFFIX_RATIO_STD = fix_uptake_file.createVariable('fffix_ratio_std','d',('lat','lon'))
        FFFIX_RATIO_LOWQ = fix_uptake_file.createVariable('fffix_ratio_lowq','d',('lat','lon'))
        FFFIX_RATIO_MIDQ = fix_uptake_file.createVariable('fffix_ratio_midq','d',('lat','lon'))
        FFFIX_RATIO_UPPQ = fix_uptake_file.createVariable('fffix_ratio_uppq','d',('lat','lon'))

        EPS_UPTAKE = fix_uptake_file.createVariable('eps_uptake','d',('lat','lon'))
        EPS_UPTAKE_STD = fix_uptake_file.createVariable('eps_uptake_std','d',('lat','lon'))

        EPS_UPTAKE_org = fix_uptake_file.createVariable('eps_uptake_org','d',('lat','lon'))
        EPS_UPTAKE_STD_org = fix_uptake_file.createVariable('eps_uptake_std_org','d',('lat','lon'))

        EPS_UPTAKE_LOWQ = fix_uptake_file.createVariable('eps_uptake_lowq','d',('lat','lon'))
        EPS_UPTAKE_MIDQ = fix_uptake_file.createVariable('eps_uptake_midq','d',('lat','lon'))
        EPS_UPTAKE_UPPQ = fix_uptake_file.createVariable('eps_uptake_uppq','d',('lat','lon'))

        SBNF_LOWQ = fix_uptake_file.createVariable('S_BNF_lowq','d',('lat','lon'))
        SBNF_MIDQ = fix_uptake_file.createVariable('S_BNF_midq','d',('lat','lon'))
        SBNF_UPPQ = fix_uptake_file.createVariable('S_BNF_uppq','d',('lat','lon'))
        SBNF_MEAN = fix_uptake_file.createVariable('S_BNF_mean','d',('lat','lon'))
        SBNF_STD = fix_uptake_file.createVariable('S_BNF_std','d',('lat','lon'))

        TBNF_LOWQ = fix_uptake_file.createVariable('T_BNF_lowq','d',('lat','lon'))
        TBNF_MIDQ = fix_uptake_file.createVariable('T_BNF_midq','d',('lat','lon'))
        TBNF_UPPQ = fix_uptake_file.createVariable('T_BNF_uppq','d',('lat','lon'))
        TBNF_MEAN = fix_uptake_file.createVariable('T_BNF_mean','d',('lat','lon'))
        TBNF_STD = fix_uptake_file.createVariable('T_BNF_std','d',('lat','lon'))

        FIX_RATIO[:] = fix_ratio_global_mat[:]
        FIX_RATIO_STD[:] = fix_ratio_std_global_mat[:]

        FIX_RATIO_org[:] = fix_ratio_global_org_mat[:]
        FIX_RATIO_STD_org[:] = fix_ratio_std_global_org_mat[:]

        FRAC_DEP[:] = frac_dep_global_mat[:]
        FRAC_DEP_STD[:] = frac_dep_std_global_mat[:]

        FIX_RATIO_LOWQ[:] = fix_ratio_lowq_global_mat[:]
        FIX_RATIO_MIDQ[:] = fix_ratio_midq_global_mat[:]
        FIX_RATIO_UPPQ[:] = fix_ratio_uppq_global_mat[:]

        FFFIX_RATIO[:] = fffix_ratio_global_mat[:]
        FFFIX_RATIO_STD[:] = fffix_ratio_std_global_mat[:]
        FFFIX_RATIO_LOWQ[:] = fffix_ratio_lowq_global_mat[:]
        FFFIX_RATIO_MIDQ[:] = fffix_ratio_midq_global_mat[:]
        FFFIX_RATIO_UPPQ[:] = fffix_ratio_uppq_global_mat[:]

        EPS_UPTAKE[:] = eps_uptake_global_mat[:]
        EPS_UPTAKE_STD[:] = eps_uptake_std_global_mat[:]

        EPS_UPTAKE_org[:] = eps_uptake_global_org_mat[:]
        EPS_UPTAKE_STD_org[:] = eps_uptake_std_global_org_mat[:]

        EPS_UPTAKE_LOWQ[:] = eps_uptake_lowq_global_mat[:]
        EPS_UPTAKE_MIDQ[:] = eps_uptake_midq_global_mat[:]
        EPS_UPTAKE_UPPQ[:] = eps_uptake_uppq_global_mat[:]

        SBNF_LOWQ[:]=SBNF_lowq_global_mat[:]
        SBNF_MIDQ[:]=SBNF_midq_global_mat[:]
        SBNF_UPPQ[:]=SBNF_uppq_global_mat[:]
        SBNF_MEAN[:]=SBNF_mean_global_mat[:]
        SBNF_STD[:]=SBNF_std_global_mat[:]

        TBNF_LOWQ[:]=TBNF_lowq_global_mat[:]
        TBNF_MIDQ[:]=TBNF_midq_global_mat[:]
        TBNF_UPPQ[:]=TBNF_uppq_global_mat[:]
        TBNF_MEAN[:]=TBNF_mean_global_mat[:]
        TBNF_STD[:]=TBNF_std_global_mat[:]

        fix_uptake_file.close()

