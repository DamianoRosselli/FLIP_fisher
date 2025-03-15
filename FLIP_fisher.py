#code to run fisher forecast with FLIP for velocity only or density x velocity

#for density you need pypower and pmesh

#create a new conda enviroment with python 3.8.0

## Damiano's instructions
#conda install -c bccp pmesh
#python -m pip install git+https://github.com/cosmodesi/pypower
#install flip from github: https://github.com/corentinravoux/flip.git
# pip install classy (for the power spectrum)


import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import flip
from flip import fitter, plot_utils, utils
from flip.covariance import covariance, contraction
from flip import fisher
from pkg_resources import resource_filename
import warnings
warnings.filterwarnings("ignore")
import astropy.cosmology as acosmo
import yaml

yaml_file='input.yml'
with open(yaml_file, 'r') as file:
    config = yaml.safe_load(file)

#function to compute power spectrum using class
def init_PS(kmax,cosmo_dic, redshift=0,model='nonlinearbel', kmin=1.e-5 ):
    
    #pmm matter PS, pmt matter-divergwnce PS, ptt divergence PS, fiducial = fiducial cosmo par like s8 and fs8
   # _available_power_spectrum_model = ["linearbel", "nonlinearbel", "linear"]
  
    
    kh, pmm, pmt, ptt,fiducial = flip.power_spectra.compute_power_spectra(
    'class_engine',
    cosmo_dic, 
    redshift,
    kmin, 
    kmax, 
    1500, 
    normalization_power_spectrum='no_normalization',
    power_spectrum_model= model 
    )
    
    return kh,pmm,pmt,ptt,fiducial


#####################################################################

## define input #######

##for numerical computation (internal flip parallelization, if the code is slow try change this)#
size_batch = config['size_batch']
number_worker = config['number_worker']

#cosmology
cosmo_dic = config['cosmo_params']

cosmo= acosmo.FlatLambdaCDM(H0=cosmo_dic['h']*100, 
                            Om0=(cosmo_dic["omega_cdm"]/(cosmo_dic['h']**2))+(cosmo_dic["omega_b"]/(cosmo_dic['h']**2)))


#data file needs ra,dec (radians) and zobs##
gal_catalog= config['gal_catalog']
vel_catalog= config['vel_catalog']
resfile = config['resfile']


## input for velocity
sigma_u = config['sigma_u']
sigmau_fiducial = sigma_u
sigma_v = config['sigma_v']
sigma_M = config['sigma_M']


#input for density
gal_bias = config['gal_bias']
sigma_g = config['sigma_g']
window_size = config['window_size']


#PS input
kmax = config['kmax']
model= config['model']

## input redshifts
zmin= config['zmin']
zmax= config['zmax']
nbins= config['nbins']

### what compute
vel_only= config['vel_only']
cross = config['cross']


#need in flip do not change
fisher_prop = {
        "inversion_method": "inverse",
        "velocity_type": "scatter",
        }

#read the data
gal = pd.read_parquet(gal_catalog)
vel = pd.read_parquet(vel_catalog)

#add info
gal["rcom_zobs"] = cosmo.comoving_distance(gal.zobs.values).value * cosmo.h
gal["hubble_norm"] = cosmo.H(gal.zobs.values).value / cosmo.h

vel["rcom_zobs"] = cosmo.comoving_distance(vel.zobs.values).value * cosmo.h
vel["hubble_norm"] = cosmo.H(vel.zobs.values).value / cosmo.h

#redshift bins
zz = np.linspace(zmin,zmax,nbins)
zmean = (zz[:-1] + zz[1:]) * 0.5 #forecast computed at the mean redshift of each bin

res = {'type':[],
      'z':[],
       'fs8':[],
      'fs8_error':[],
      'fs8_percentage_error': []
    }

for i,z in enumerate(zmean):

    #compute PS 
    ktt,pmm,pmt, ptt,fid = init_PS(kmax,cosmo_dic,redshift=z,model=model)
    grid_window=flip.gridding.compute_grid_window(window_size, ktt, kind="cic", n=1500)
    power_spectrum_dict = {"gg": [[ktt, pmm * np.array(grid_window)**2],[ktt, pmt * np.array(grid_window)],[ktt, ptt]],
                       "gv": [[ktt, pmt * np.array(grid_window)* utils.Du(ktt, sigmau_fiducial)],[ktt, ptt* utils.Du(ktt, sigmau_fiducial)]],
                       "vv": [[ktt, ptt * utils.Du(ktt, sigmau_fiducial)**2]]}

    #selec data
    gal_data = gal[(gal.zobs<zz[i+1]) & (gal.zobs>zz[i])]
    vel_data = vel[(vel.zobs<zz[i+1]) & (vel.zobs>zz[i])]
    max_rcom = np.max((np.max(vel_data.rcom_zobs),np.max(gal_data.rcom_zobs)))
    
    kmin = 2*np.pi/(max_rcom)

    #prepare the data
    #the grid of density can take a bit long if you need again the griddata save them 
    if cross:              
        grid_data = flip.gridding.grid_data_density_pypower(
                    gal_data.ra.values,
                    gal_data.dec.values,
                    gal_data.rcom_zobs.values,
                    np.max(gal_data.rcom_zobs.values),
                    window_size,
                    "rect",
                    'cic',
                    Nrandom=10,
                    random_method="cartesian",
                    interlacing=2,
                    compensate=False,
                    coord_randoms=None,
                    min_count_random=0,
                    overhead=20,
                    )


        data_dens=flip.data_vector.FisherDens({'ra':grid_data['ra'],'dec':grid_data['dec'],
                                               'rcom_zobs':grid_data['rcom'],'density_error':grid_data['density_err']})


    if vel_only or cross:
        data_vel=flip.data_vector.FisherVelFromHDres({'ra':vel_data.ra.values,'dec':vel_data.dec.values,'zobs':vel_data.zobs.values,
                                                     'rcom_zobs':vel_data.rcom_zobs.values,'hubble_norm':vel_data.hubble_norm.values})


    #fisher_forecast vel only
    if vel_only:
        cov_vel= data_vel.compute_covariance('carreres23',power_spectrum_dict,
                           size_batch=size_batch,
                            number_worker=number_worker,)
        param_dict = { 'fs8':fid['fsigma_8'],"sigma_M":sigma_M,"sigv": sigma_v}

        fish_vel = fisher.FisherMatrix.init_from_covariance(
                    covariance=cov_vel,
                    data=data_vel,
                    parameter_values_dict=param_dict,
                    fisher_properties=fisher_prop)
        
        par_v,fmat_v = fish_vel.compute_fisher_matrix()
        err_fs8 = np.sqrt(np.linalg.inv([[fmat]]))

        res['type'].append('vel')
        res['z'].append(z)
        res['fs8'].append(fid['fsigma_8'])
        res['fs8_error'].append(err_fs8)
        res['fs8_percentage_error'].append(100*err_fs8/fid['fsigma_8'])
        

    #fisher forecast density x vel
    if cross:
        data=flip.data_vector.FisherDensVel(data_dens,data_vel)

    
        cov=data.compute_covariance(
                            'ravouxcarreres',power_spectrum_dict,
                               size_batch=size_batch,
                            number_worker=number_worker,
                            additional_parameters_values=(sigma_g,))   

        betaf= (fid['fsigma_8']/fid['sigma_8'])/gal_bias
        bs8= gal_bias * fid['sigma_8']
        param_dict = { 'beta_f':betaf, 'bs8':bs8, 'fs8':fid['fsigma_8']}

   
        fish = fisher.FisherMatrix.init_from_covariance(
            covariance=cov,
            data=data,
            parameter_values_dict=param_dict,
            fisher_properties=fisher_prop)


        par,fmat = fish.compute_fisher_matrix()
        err_fs8 = np.sqrt(np.linalg.inv([[fmat]]))[0][0]
        
        res['type'].append('cross')
        res['z'].append(z)
        res['fs8'].append(fid['fsigma_8'])
        res['fs8_error'].append(err_fs8)
        res['fs8_percentage_error'].append(100*err_fs8/fid['fsigma_8'])
        
    
res_df = pd.DataFrame(res)
res_df.to_csv(resfile)
    
   
