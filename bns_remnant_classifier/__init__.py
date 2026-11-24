from .remnant_classifier import (
    compute_bns_remnant_probs,
    run_classifierA_analysis,
    run_classifierB_analysis,
)
from .marginalization import lambdas_from_eos, compute_low_lat_params


from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("bns_remnant_classifier")
except PackageNotFoundError:
    # package isn't installed (e.g. running directly from a source checkout)
    __version__ = "unknown"
