"""
Module to set up and handling logging 
"""
import logging

logger = logging.getLogger("bns_remnant_classifier")
logger.addHandler(logging.NullHandler())

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s-%(levelname)s: %(message)s"
    )


