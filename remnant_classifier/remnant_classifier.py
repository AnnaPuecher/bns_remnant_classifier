'''
Copyright: (c) 2026 Anna Puecher, MIT License in LICENSE.md 
File Name: remnant_classifier.py
Description: Script to run ClassifierA and ClassifierB from Phys.Rev.D 110 (2024) 12, 123038
             The classifier take as input the parameters of a BNS system and predict the remnant formed after the merger.
             Application both to posterior sampels and low-latency point estimates. See https://git.ligo.org/anna.puecher/remnant-classifier/-/wikis/Review/Documentation

Developers: Anna Puecher, Tim Dietrich
Contact: anna.puecher@uni-potsdam.de
'''

import numpy as np
import joblib
import bilby
import pandas as pd
import logging
import json
from pathlib import Path
from .marginalization import lambdas_from_eos, compute_low_lat_params
from importlib.resources import files
from .logging import logger

def run_classifierA_analysis(xgw, probs=False):

    """
    Function to use ClassifierA. Reads the input features, scales them, and runs ClassifierA.

    Args:
        - xgw (dict):  input features (total mass in source frame, mass-weighted tidal deformability, mass ratio, effective inspiral spin)
        - probs (bool): if True, returns the porbability predicted for each class, if False returns only the most likely class (default False)
    """

    model_path = files("remnant_classifier").joinpath("models", "classifierA_model.pkl")
    with model_path.open("rb") as f:
        modelA = joblib.load(f)
    
    scaler_path = files("remnant_classifier").joinpath("models", "scaler_classifierA.pkl")
    with scaler_path.open("rb") as f:
        scalerA = joblib.load(f)

    xgw_sc = scalerA.transform(xgw)

    if probs:

        return modelA.predict_proba(xgw_sc)

    else:

        ygw= modelA.predict(xgw_sc)

        ybh = np.sum(ygw==1)
        yns = np.sum(ygw==0)

        pbh = ybh/(ybh+yns)
        pns = yns/(ybh+yns)

        return pbh, pns


def run_classifierB_analysis(xgw, probs=False):

    """
    Function to use ClassifierB. Reads the input features, scales them, and runs ClassifierB.

    Args:
        - xgw (dict):  input features (total mass in source frame, mass-weighted tidal deformability, mass ratio, effective inspiral spin)
        - probs (bool): if True, returns the porbability predicted for each class, if False returns only the most likely class (default False)
    """

    model_path = files("remnant_classifier").joinpath("models", "classifierB_model.pkl")
    with model_path.open("rb") as f:
        modelB = joblib.load(f)

    scaler_path = files("remnant_classifier").joinpath("models", "scaler_classifierB.pkl")
    with scaler_path.open("rb") as f:
        scalerB = joblib.load(f)

    xgw_sc = scalerB.transform(xgw)

    if probs:

        return modelB.predict_proba(xgw_sc)

    else:

        ygw= modelB.predict(xgw_sc)

        ybh = np.sum(ygw==1)
        yhmns = np.sum(ygw==2)
        ylong = np.sum(ygw==4)

        pbh = ybh/(ybh+yhmns+ylong)
        phmns = yhmns/(ybh+yhmns+ylong)
        plong = ylong/(ybh+yhmns+ylong)

        return pbh, phmns, plong


def prepare_pos_data(posfile, eos_dir=None, eos_prob=None, num_eos=None, compute_lambdas=False):


    with open(posfile, 'r') as file:
            pos_event = json.load(file)

    logging.info("Successfully read posteriors input file.")

    data_keys = pos_event.keys()

    if "total_mass_source" not in data_keys:
        if "mass_1_source" in data_keys and "mass_2_source" in data_keys:
            pos_event["total_mass_source"] = pos_event["mass_1_source"] + pos_event["mass_2_source"]
            pos_event["mratio"] = pos_event["mass_2_source"] / pos_event["mass_1_source"]
        elif "chirp_mass_source" in data_keys and "mass_ratio" in data_keys:
            pos_event["total_mass_source"] = bilby.gw.conversion.chirp_mass_and_mass_ratio_to_total_mass(pos_event["chirp_mass_source"], pos_event["mass_ratio"])
        else:
            raise ValueError("No masses values provided in the posteriors file! Either 'total_mass_source' and 'mass_ratio_source', 'chirp_mass_source' and 'mass_ratio', or 'mass_1_source' and 'mass_2_source' are needed.")

    if any(qv > 1 for qv in pos_event["mass_ratio"]):
        raise ValueError("ERROR: mass ratio is assumed as q = m2/m1, so we expect always q <=1")

    if "chi_eff" not in data_keys:

        if "chi_1" in data_keys and "chi_2" in data_keys:
            m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(pos_event["mass_ratio"], pos_event["total_mass"])
            chi_eff = (pos_event["chi_1"] * m1 + pos_event["chi_2"] * m2) / (pos_event["total_mass"])
            pos_event["chi_eff"] = chi_eff
        else:
            raise ValueError("No spin values provided in the posteriors file! Either 'chi_eff', or 'chi_1' and 'chi_2' are needed.")

    if compute_lambdas:

        if "mass_1_source" in data_keys and "mass_2_source" in data_keys:
            m1 = pos_event["mass_1_source"]
            m2 = pos_event["mass_2_source"]

        else:
            m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(np.array(pos_event["mass_ratio"]), np.array(pos_event["total_mass_source"]))

            pos_event['lambda_tilde'] = lambdas_from_eos(eos_dir, eos_prob, num_eos, m1, m2)

    else:
        if "lambda_tilde" not in data_keys:
            if "lambda_1" in data_keys and "lambda_2" in data_keys:
                if "mass_1_source" in data_keys and "mass_2_source" in data_keys:
                    m1 = pos_event["mass_1_source"]
                    m2 = pos_event["mass_2_source"]
                elif "mass_ratio" in data_keys and "total_mass_source" in data_keys:
                    m1, m2 = bilby.gw.conversion.total_mass_and_mass_ratio_to_component_masses(1./np.array(pos_event["mass_ratio"]), np.array(pos_event["total_mass_source"]))

                pos_event["lambda_tilde"] = bilby.gw.conversion.lambda_1_lambda_2_to_lambda_tilde(pos_event["lambda_1"], pos_event["lambda_2"], m1, m2)
            else:
                raise ValueError("EoS marginalization disabled but no tidal information available: provide tidal deformability posteriors or set pos_marginalization=Trueto marginalize over a set of EoS")

    data_pos = {'Mtot':pos_event['total_mass_source'], 'LambdaTilde': pos_event['lambda_tilde'], 'ChiEff':pos_event['chi_eff'], 'Mratio_fixed':(1./np.array(pos_event['mass_ratio'])).tolist()}

    return data_pos


def compute_bns_remnant_probs(classifiers="A", analysis="pos", par_file=None, pos_marginalization=False, eosweights=None, eosdir=None, lamweight=None):

    logging.info(f"Running classifier {classifiers}")

    if analysis == 'lowlat':

        logging.info("Running on LOW-LATENCY data")
        logging.info(f"EOS marginalization weighting over {lamweight}")

        if not eosweights or not eosdir:
            raise ValueError("For low-latency applications a directory with equation of state files and a file with equation of state probabilities must be provided.")

        if lamweight != 'lambda' and lamweight != 'prediction':
            raise ValueError(f'lamweight "{lamweight}" is not a valid entry. Options are "lambda" to weight lambda tilde or "prediction" to weight the classifier predictions')

        folder = Path(eosdir)
        count = len(list(folder.glob("*.dat")))

        total_mass, mass_ratio, effective_spin, lamT_and_weights = compute_low_lat_params(par_file, eosdir, eosweights, count)

        if lamweight == "lambda":

            lamT_wmean = np.average(lamT_and_weights[:,0], weights=lamT_and_weights[:,1])

            data_lowlat = {'Mtot':[total_mass], 'LambdaTilde': [lamT_wmean], 'ChiEff':[effective_spin], 'Mratio_fixed':[mass_ratio]}
            data_x = pd.DataFrame(data_lowlat)

            if "A" in classifiers:

                probs = run_classifierA_analysis(data_x, probs=True)[0]
                prob_pca = probs[1]
                prob_ns = probs[0]

            if "B" in classifiers:

                probs = run_classifierB_analysis(data_x, probs=True)[0]
                prob_pcb = probs[0]
                prob_hmns = probs[1]
                prob_long = probs[2]

        elif lamweight == 'prediction':

            N = len(lamT_and_weights[:,0])
            mtot = [total_mass] *(N)
            chieff = [effective_spin] * (N)
            mratio = [mass_ratio] * (N)
            prob_tot = sum(lamT_and_weights[:,1])

            data_lowlat = {'Mtot':mtot, 'LambdaTilde': lamT_and_weights[:,0], 'ChiEff':chieff, 'Mratio_fixed':mratio}
            data_x = pd.DataFrame(data_lowlat)

            if "A" in classifiers:

                probs = run_classifierA_analysis(data_x, probs=True)
                prob_pca = 0
                prob_ns = 0

                for ii in range(0,probs.shape[0]):

                    prob_ns += probs[ii,0]*lamT_and_weights[ii,1]
                    prob_pca += probs[ii,1]*lamT_and_weights[ii,1]

                prob_ns /= prob_tot
                prob_pca /= prob_tot

                probs_remn_a = [prob_pca, prob_ns]\

            else:
                probs_remn_a=None

            if "B" in classifiers:

                probs = run_classifierB_analysis(data_x, probs=True)

                prob_pcb = 0
                prob_hmns = 0
                prob_long = 0

                for ii in range(0,probs.shape[0]):

                    prob_pcb += probs[ii,0]*lamT_and_weights[ii,1]
                    prob_hmns += probs[ii,1]*lamT_and_weights[ii,1]
                    prob_long += probs[ii,2]*lamT_and_weights[ii,1]

                prob_pcb /= prob_tot
                prob_hmns /= prob_tot
                prob_long /= prob_tot

                probs_remn_b = [prob_pcb, prob_hmns, prob_long]

            else:
                probs_remn_b = None

    elif analysis == 'pos':

        logging.info("Running on POSTERIORS data")
        if pos_marginalization:
            if not eosweights or not eosdir:
                raise ValueError("A directory with equation of state files and a file with equation of state probabilities must be provided in order to marginalize over lambda.")
            else:
                folder = Path(eosdir)
                count = len(list(folder.glob("*.dat")))
        
            pos_event = prepare_pos_data(par_file, eosdir, eosweights, count, compute_lambdas=True)

        else:
            pos_event = prepare_pos_data(par_file)

        data_x = pd.DataFrame(pos_event)

        if "A" in classifiers:

            prob_pca, prob_ns = run_classifierA_analysis(data_x)
            probs_remn_a = [prob_pca, prob_ns]
        else:
            probs_remn_a=None

        if "B" in classifiers:

            prob_pcb, prob_hmns, prob_long = run_classifierB_analysis(data_x)
            probs_remn = [prob_pcb, prob_hmns, prob_long]
        else:
            probs_remn_b=None

    else:
        raise ValueError(f'Analysis "{analysis}" is not a valid entry. Options are "pos" for posteriors and "lowlat" for low-latency')


    return probs_remn_a, probs_remn_b


