"""
Module to set up and handling logging 
"""
import logging

logger = logging.getLogger()

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M",
        format="%(asctime)s-%(levelname)s: %(message)s"
    )

setup_logger()
