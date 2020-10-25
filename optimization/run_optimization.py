# -*- coding: utf-8 -*-
import os
import parameters
import optim_model

def run(obj_fn = "tac"):
    """
    objective function: tac or co2_gross
    """
    if not obj_fn in ("tac", "co2_gross"):
        raise Exception("undefined objective function " + obj_fn)
    
    # Define paths
    path_file = str(os.path.dirname(os.path.realpath(__file__)))
    
    # Load parameters
    nodes, param, devs = parameters.load_params(path_file)
    
    # Run device optimization
    param, capacities = optim_model.run_optim(obj_fn, nodes, param, devs)
    
    return param, capacities