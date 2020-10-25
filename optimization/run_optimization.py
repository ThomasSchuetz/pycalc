# -*- coding: utf-8 -*-
import os
from .parameters import load_params
from .optim_model import run_optim

def run(obj_fn = "tac"):
    """
    objective function: tac or co2_gross
    """
    if not obj_fn in ("tac", "co2_gross"):
        raise Exception("undefined objective function " + obj_fn)
    
    # Define paths
    path_file = str(os.path.dirname(os.path.realpath(__file__)))
    
    # Load parameters
    nodes, param, devs = load_params(path_file)
    
    # Run device optimization
    param, capacities = run_optim(obj_fn, nodes, param, devs)
    
    return param, capacities