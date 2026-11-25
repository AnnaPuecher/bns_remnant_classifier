"""
Python package to run ClassifierA and ClassifierB from Phys.Rev.D 110 (2024) 12, 123038
The classifiers take as input the parameters of a BNS system and predict the remnant formed after the merger.
Application possible both to parameters posterior samples and low-latency point estimates.
Marginalizes over equation of state when tidal information not available
"""

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

# Bundled default EoS data (the subset with non-zero probability in set A)
DEFAULT_EOS_DIR = files("bns_remnant_classifier").joinpath("eos_ns_properties_default", "MRL_default")
DEFAULT_EOS_PROB = files("bns_remnant_classifier").joinpath("eos_ns_properties_default", "eos_pos_setA.txt")

def run_classifierA_analysis(xgw, probs=False):

    """
    Function to use ClassifierA. Reads the input features, scales them, and runs ClassifierA.

    Args:
        - xgw (dict):  input features (total mass in source frame, mass-weighted tidal deformability, mass ratio, effective inspiral spin)
        - probs (bool): if True, returns the porbability predicted for each class, if False returns only the most likely class (default False)

    Returns:
        - if probs=True: array of per-class probabilities as predicted by the classifier
        - if probs=False: tuple (pbh, pns) - probability of prompt collapse to black hole and neutron star remnant respectively, summing over all the predictions
    """

    model_path = files("bns_remnant_classifier").joinpath("models", "classifierA_model.pkl")
    with model_path.open("rb") as f:
        modelA = joblib.load(f)
    
    scaler_path = files("bns_remnant_classifier").joinpath("models", "scaler_classifierA.pkl")
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

    Returns:
        - if probs=True: array of per-class probabilities as predicted by the classifier
        - if probs=False: tuple (pbh, phmns, plong) - probability prompt collapse to black hole, formation of  ahypermassive NS, or or a long-lived NS, respectively, summing over all the predictions
    """

    model_path = files("bns_remnant_classifier").joinpath("models", "classifierB_model.pkl")
    with model_path.open("rb") as f:
        modelB = joblib.load(f)

    scaler_path = files("bns_remnant_classifier").joinpath("models", "scaler_classifierB.pkl")
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

    """
    Reads a posterior-samples JSON file and assembles the feature set needed
    as classifier input, deriving any missing mass, spin, or tidal
    quantities from what is available. If compute_lambdas=True, it marginalizes lambdas over a set of EoS.

    Args:
        - posfile (str): path to a JSON file with posterior samples. Must contain
          either 'total_mass_source' and 'mass_ratio', or 'mass_1_source' and
          'mass_2_source', or 'chirp_mass_source' and 'mass_ratio'; and either
          'chi_eff', or 'chi_1' and 'chi_2'.
        - eos_dir (str, optional): path to directory with EoS files. Required if
          compute_lambdas is True.
        - eos_prob (str, optional): path to file with EoS posterior probabilities.
          Required if compute_lambdas is True.
        - num_eos (int or 'all', optional): number of EoS to marginalize over
          (see lambdas_from_eos). Required if compute_lambdas is True.
        - compute_lambdas (bool): if True, lambda_tilde is marginalized over the
          EoS set via lambdas_from_eos. If False, lambda_tilde is instead read
          directly from posfile (either as 'lambda_tilde', or computed from
          'lambda_1'/'lambda_2') (default False).

    Returns:
        - dict with keys 'Mtot', 'LambdaTilde', 'ChiEff', 'Mratio_fixed', ready
          to be used as classifier input.

    Raises:
        - ValueError if required mass, spin, or tidal information is missing
          from posfile.
    """

    with open(posfile, 'r') as file:
            pos_event = json.load(file)

    logger.info("Successfully read posteriors input file.")

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


def compute_bns_remnant_probs(classifiers="A", analysis="pos", par_file=None, pos_marginalization=False, eosweights=None, eosdir=None, count='all', lamweight=None):

    """
    Runs ClassifierA and/or ClassifierB to predict the remnant of a BNS merger,
    either from posterior samples or from a low-latency point estimate.

    Args:
        - classifiers (str): which classifier(s) to run - any combination of
          "A" and "B" (e.g. "A", "B", "AB") (default "A").
        - analysis (str): "pos" to analyse posterior samples, or "lowlat" to
          analyse a low-latency point estimate (default "pos").
        - par_file (str): path to the input file - a posterior-samples JSON
          for analysis="pos", or a low-latency point-estimate JSON for
          analysis="lowlat".
        - pos_marginalization (bool): for analysis="pos" only - if True,
          lambda_tilde is marginalized over a set of EoS instead of being
          read directly from par_file (default False).
        - eosweights (str, optional): path to the EoS posterior-probability
          file. If not provided, falls back to the bundled default
          (eos_pos_setA.txt).
        - eosdir (str, optional): path to the directory of EoS files. If not
          provided, falls back to the bundled default (MRL_default).
        - count (int or 'all'): number of EoS to marginalize over, passed
          through to the EoS marginalization functions (default 'all', i.e.
          every EoS available in eosdir).
        - lamweight (str): required for analysis="lowlat" - "lambda" to
          weight lambda_tilde by EoS probability before classification, or
          "prediction" to classify per-EoS and weight the resulting
          probabilities.

    Returns:
        - tuple (probs_remn_a, probs_remn_b):
            - probs_remn_a: [p_bh, p_ns] from ClassifierA, or None if "A" not in classifiers
            - probs_remn_b: [p_bh, p_hmns, p_long] from ClassifierB, or None if "B" not in classifiers

    Raises:
        - ValueError if analysis is not "pos" or "lowlat", if lamweight is
          invalid for analysis="lowlat", or if EoS marginalization is
          requested without the required files.
    """
    logger.info(f"Running classifier {classifiers}")

    if analysis == 'lowlat':

        logger.info("Running on LOW-LATENCY data")
        logger.info(f"EOS marginalization weighting over {lamweight}")

        if not eosdir:
            eosdir = str(DEFAULT_EOS_DIR)
            logger.warning("No EoS data provided, reverting to default 'MRL_default'")
        if not eosweights:
            eosweights = str(DEFAULT_EOS_PROB)
            logger.warning("No EoS posteriors file provided, reverting to default 'eos_pos_setA.txt'")

        if lamweight != 'lambda' and lamweight != 'prediction':
            raise ValueError(f'lamweight "{lamweight}" is not a valid entry. Options are "lambda" to weight lambda tilde or "prediction" to weight the classifier predictions')

        total_mass, mass_ratio, effective_spin, lamT_and_weights = compute_low_lat_params(par_file, eosdir, eosweights, count)

        if lamweight == "lambda":

            lamT_wmean = np.average(lamT_and_weights[:,0], weights=lamT_and_weights[:,1])

            data_lowlat = {'Mtot':[total_mass], 'LambdaTilde': [lamT_wmean], 'ChiEff':[effective_spin], 'Mratio_fixed':[mass_ratio]}
            data_x = pd.DataFrame(data_lowlat)

            if "A" in classifiers:

                probs = run_classifierA_analysis(data_x, probs=True)[0]
                prob_pca = probs[1]
                prob_ns = probs[0]

                probs_remn_a = [prob_pca, prob_ns]

            else:
                probs_remn_a = None

            if "B" in classifiers:

                probs = run_classifierB_analysis(data_x, probs=True)[0]
                prob_pcb = probs[0]
                prob_hmns = probs[1]
                prob_long = probs[2]

                probs_remn_b = [prob_pcb, prob_hmns, prob_long]

            else:
                probs_remn_b = None


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

                probs_remn_a = [prob_pca, prob_ns]

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

        logger.info("Running on POSTERIORS data")
        
        if pos_marginalization:

            if not eosdir:
                eosdir = str(DEFAULT_EOS_DIR)
                logger.warning("EoS marginalization requested but no EoS data provided, reverting to default 'MRL_default'")
            if not eosweights:
                eosweights = str(DEFAULT_EOS_PROB)
                logger.warning("EoS marginalization requested but no EoS posteriors file provided, reverting to default 'eos_pos_setA.txt'")
       
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
            probs_remn_b = [prob_pcb, prob_hmns, prob_long]
        else:
            probs_remn_b=None

    else:
        raise ValueError(f'Analysis "{analysis}" is not a valid entry. Options are "pos" for posteriors and "lowlat" for low-latency')


    return probs_remn_a, probs_remn_b


