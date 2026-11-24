"""
Module to handle marginalization over equations of state both in the case 
of analysis of posterior samples and low-latency estimates
"""

import numpy as np
import os
import glob
from scipy.interpolate import interp1d
import bilby
from tqdm import tqdm
import json
from .logging import logger

def compute_threshold(eos_probs, n_wanted_eos):

    """
    Function to compute the likelihood threshold value to impose on the EoS probabilities in order to marginalize only over a wanted number of EoS, "n_wanted_eos". 
    Marginalization happens only over the "n_wanted_eos" with larger probability.

    Args:
        - eos_probs (list): list with posterior probabilities of all the EoSs
        - n_wanted eos (int): number of EoSs we want to marginalize over

    Return:
        - threshold value to impose on likelihood in order to consider the "n_wanted_eos" with largest probability
    """
    sorted_vals = np.sort(eos_probs)[::-1]
    return sorted_vals[n_wanted_eos - 1]

def get_available_eos_ids(eos_dir):
    """
    Return the sorted array of EoS ids that have a '<id>.dat' file
    present in eos_dir.
    """
    ids = []
    for path in glob.glob(os.path.join(eos_dir, "*.dat")):
        name = os.path.splitext(os.path.basename(path))[0]
        ids.append(int(name))
    return np.array(sorted(ids))

def lambdas_from_eos(eos_dir, eos_prob, num_eos, mass1_pos, mass2_pos):

    """
    Function to compute mass-weighted tidal deformability for posteriors marginalizing over EoSs.

    Args:
        - eos_dir (str): path to direcotry with the EoS files
        - eos_prob (str): path to file with posterior probabilities of the employed EoS state
        - num_eos (int or 'all'): number of EoSs we want to marginalize over; the needed threshold on likelihood is computed consequently.
            If 'all', all the EoS available in MRL_default are used
        - mass1_pos (list): mass1 posterior samples
        - mass2_pos (list): mass2 posterior samples

    Returns:
        - mass-weighted tidal deformbaility marginalized  over the wanted number of EoS for each masses sample
    """


    eos_pos = np.loadtxt(eos_prob, skiprows=1, unpack=True)
    
    if num_eos == 'all':
        eosids = get_available_eos_ids(eos_dir)
    else:
        lik_threshold = compute_threshold(eos_pos, num_eos)
        eosids = np.where(eos_pos >= lik_threshold)[0] + 1
        
        if len(eosids) > num_eos:
            logger.warning(
                "%d EoS selected for marginalization instead of the wanted %d due to ties at threshold",
                len(eosids), num_eos)

    mass1_pos = np.asarray(mass1_pos)
    mass2_pos = np.asarray(mass2_pos)
    n_masses = len(mass1_pos)

    lamT_weights = []

    for eosid in tqdm(eosids):

        posprob = eos_pos[eosid - 1]  
        eos_file = os.path.join(eos_dir, '%i.dat' % eosid)
        _, mass_eos, lam_eos, _ = np.loadtxt(eos_file, skiprows=1, unpack=True)
        mtov = mass_eos.max()

        f_m_lam_eosid = interp1d(mass_eos, lam_eos)

        below = mtov < mass2_pos
        mid   = (mtov < mass1_pos) & (mtov >= mass2_pos)
        above = (mtov >= mass1_pos) & (mtov >= mass2_pos)

        if not np.all(below | mid | above):
            raise ValueError("Masses values not recognized")

        lambda_tilde = np.zeros(n_masses)

        if np.any(mid):
            lam2 = f_m_lam_eosid(mass2_pos[mid])
            lambda_tilde[mid] = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
                np.zeros_like(lam2), lam2, mass1_pos[mid], mass2_pos[mid]
            )

        if np.any(above):
            lam1 = f_m_lam_eosid(mass1_pos[above])
            lam2 = f_m_lam_eosid(mass2_pos[above])
            lambda_tilde[above] = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(
                lam1, lam2, mass1_pos[above], mass2_pos[above]
            )

        posprob_arr = np.full(n_masses, posprob)
        lamT_weights.append(np.column_stack([lambda_tilde, posprob_arr]))

    lamT_arr = np.array(lamT_weights)

    lamT_pos = []
    for mm in range(n_masses):
        values  = lamT_arr[:, mm, 0]
        weights = lamT_arr[:, mm, 1]
        lamT_pos.append(np.average(values, weights=weights))

    return lamT_pos

def compute_low_lat_params(lowlat_file, eos_dir, eos_prob, num_eos):
    
    """
    Function to compute the quantities needed as input by the classifier from low-latency data.

    Args:
        - lowlat_file (str): path to file with low-latency point estimates of masses and spins, and distance estimate
        - eos_dir (str): path to direcotry with the EoS files
        - eos_prob (str): path to file with posterior probabilities of the employed EoS state
        - num_eos (int or 'all'): number of EoSs we want to marginalize over; the needed threshold on likelihood is computed consequently.
            If 'all', all the EoS available in MRL_default are used

    Returns:
        - total mass in source frame in units of solar masses (float) 
        - mass ratio, following convention q>1 (float)
        - effective inspiral spin, adimensional (float)
        - mass-weighted tidal deformbaility for each EoS and corresponding weight in the form as a 2D array (lambda_tilde, weight)
    """


    with open(lowlat_file, 'r') as file:
            event_data = json.load(file)
    mass1_det = event_data['m1_det']
    mass2_det = event_data['m2_det']
    chi1 = event_data['chi_1']
    chi2 = event_data['chi_2']
    dist = event_data['luminosity_distance']
  
    z = bilby.gw.conversion.luminosity_distance_to_redshift(dist)
    mass1 = mass1_det / (1. + z)
    mass2 = mass2_det / (1. + z)

    mtot = mass1 + mass2
    mratio = mass1 / mass2

    chi_eff = (chi1 * mass1_det + chi2 * mass2_det) / (mass1_det + mass2_det)

    lamT_weights = []

    eos_pos = np.loadtxt(eos_prob, skiprows=1, unpack=True)
   
    if num_eos == 'all':
        eosids = get_available_eos_ids(eos_dir)
    else:
        lik_threshold = compute_threshold(eos_pos, num_eos)
        eosids = np.where(eos_pos >= lik_threshold)[0] + 1
   
        if len(eosids) > num_eos:
            logger.warning(
                "%d EoS selected for marginalization instead of the wanted %d due to ties at threshold",
                len(eosids), num_eos)

    for eosid in tqdm(eosids):

        posprob = eos_pos[eosid-1]
   
        eos_file = os.path.join(eos_dir,'%i.dat'%(eosid))
        _, mass_eos, lam_eos, _ = np.loadtxt(eos_file, skiprows=1, unpack=True)
        mtov = np.max(mass_eos)

        if mtov>= mass1 and mtov >= mass2:
            
            try:
                f_m_lam_eosid = interp1d(mass_eos, lam_eos)
                    
                lam1 = f_m_lam_eosid(mass1)
                lam2 = f_m_lam_eosid(mass2)

                lambda_tilde = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(lam1, lam2, mass1, mass2)
                lamT_weights.append([lambda_tilde, posprob])
            
            except Exception as e:
                print("Error occurred:", e)
            
    lamT_weights = np.array(lamT_weights)      
    
    return mtot, mratio, chi_eff, lamT_weights 

