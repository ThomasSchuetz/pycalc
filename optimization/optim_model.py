# -*- coding: utf-8 -*-
from ortools.linear_solver import pywraplp
import time
import numpy as np

def run_optim(obj_fn, dem, param, devs, dir_results):
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Load model parameter
    start_time = time.time()   
          
    # Create array of type days
    days = range(param["n_clusters"])
    # Create array of time steps per day
    time_steps = range(24)
    # Create array for every day during one year
    year = range(365)
    
    # Get sigma function which assigns every day of the year to a typeday
    sigma = param["sigma"]
    
    # Create set for devices
    all_devs = ["BOI", "CHP", "AC", "CC", "TES", "CTES", "BAT"]
    
         
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Setting up the model
    
    # Create a new model
    model = pywraplp.Solver('k_medoids', 
                             pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Create new variables
    
    # Piece-wise linear function variables
    if param["switch_cost_functions"]:
        lin = {}
        for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "BAT"]:   
            lin[device] = {}
            for i in range(len(devs[device]["cap_i"])):
                lin[device][i] = model.NumVar(0, model.infinity(), name="lin_" + device + "_i" + str(i))
            
    # Device's capacity (i.e. nominal power)
    cap = {}
    for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "BAT"]:
        cap[device] = model.NumVar(0, model.infinity(), name="nominal_capacity_" + str(device))
    
    # Gas flow to/from devices
    gas = {}
    for device in ["BOI", "CHP"]:
        gas[device] = {}
        for d in days:
            gas[device][d] = {}
            for t in time_steps:
                gas[device][d][t] = model.NumVar(0, model.infinity(), name="gas_" + device + "_d" + str(d) + "_t" + str(t))
        
    # Eletrical power to/from devices
    power = {}
    for device in ["CHP", "CC", "from_grid", "to_grid"]:
        power[device] = {}
        for d in days:
            power[device][d] = {}
            for t in time_steps:
                power[device][d][t] = model.NumVar(0, model.infinity(), name="power_" + device + "_d" + str(d) + "_t" + str(t))
       
    # Heat to/from devices
    heat = {}
    for device in ["BOI", "CHP", "AC"]:
        heat[device] = {}
        for d in days:
            heat[device][d] = {}
            for t in time_steps:
                heat[device][d][t] = model.NumVar(0, model.infinity(), name="heat_" + device + "_d" + str(d) + "_t" + str(t))
    
    # Cooling power to/from devices
    cool = {}
    for device in ["CC", "AC"]:
        cool[device] = {}
        for d in days:
            cool[device][d] = {}
            for t in time_steps:
                cool[device][d][t] = model.NumVar(0, model.infinity(), name="cool_" + device + "_d" + str(d) + "_t" + str(t))


    # Storage variables
      
    ch = {}  # Energy flow to charge storage device
    dch = {} # Energy flow to discharge storage device
    soc = {} # State of charge

    for device in ["TES", "CTES", "BAT"]:
        ch[device] = {}
        dch[device] = {}
        soc[device] = {}
        for d in days:
            ch[device][d] = {}
            for t in time_steps:
                ch[device][d][t] = model.NumVar(0, model.infinity(), name="ch_" + device + "_d" + str(d) + "_t" + str(t))
        for d in days:
            dch[device][d] = {}
            for t in time_steps:
                dch[device][d][t] = model.NumVar(0, model.infinity(), name="dch_" + device + "_d" + str(d) + "_t" + str(t))
        for day in year:
            soc[device][day] = {}
            for t in time_steps:
                soc[device][day][t] = model.NumVar(0, model.infinity(), name="soc_" + device + "_d" + str(day) + "_t" + str(t))
#        soc[device][len(year)-1][len(time_steps)] = model.NumVar(0, model.infinity(), name="soc_" + device + "_" + str(len(year)-1) + "_t" + str(len(time_steps)))
              
        
    # Variables for annual device costs     
    inv = {}
    c_inv = {}
    c_om = {}
    c_total = {}
    for device in all_devs:
        inv[device] = model.NumVar(0, model.infinity(), name="investment_costs_" + device)
    for device in all_devs:
        c_inv[device] = model.NumVar(0, model.infinity(), name="annual_investment_costs_" + device)
    for device in all_devs:
        c_om[device] = model.NumVar(0, model.infinity(), name="om_costs_" + device)
    for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "BAT"]:
        c_total[device] = model.NumVar(0, model.infinity(), name="total_annual_costs_" + device)            

  
    # grid maximum transmission power
    grid_limit_el = model.NumVar(0, model.infinity(), name="grid_limit_el")  
    grid_limit_gas = model.NumVar(0, model.infinity(), name="grid_limit_gas")
    
    # feed-in
    feed_in = {}
    for device in ["CHP"]:
        feed_in[device] = {}
        for d in days:
            feed_in[device][d] = {}
            for t in time_steps:
                feed_in[device][d][t] = model.NumVar(0, model.infinity(), name="feed_in_" + device + "_d" + str(d) + "_t" + str(t))
    
    # total energy amounts taken from grid and fed into grid
    from_grid_total = model.NumVar(0, model.infinity(), name="from_grid_total")
    to_grid_total = model.NumVar(0, model.infinity(), name="to_grid_total")
    gas_total = model.NumVar(0, model.infinity(), name="gas_total")
    
    # total revenue for feed-in
    revenue_feed_in = {}
    for device in ["CHP"]:
        revenue_feed_in[device] = model.NumVar(0, model.infinity(), name="revenue_feed_in_"+str(device))
    # Electricity costs
    electricity_costs = model.NumVar(0, model.infinity(), name="electricity_costs")    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Objective functions
    obj = {}
    obj["tac"] = model.NumVar(-model.infinity(), model.infinity(), name="total_annualized_costs") 
    obj["co2_gross"] = model.NumVar(-model.infinity(), model.infinity(), name="total_CO2") 

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Add constraints
 
    #%% DEVICE CAPACITIES
   
    # Storage maximum capacities   
    for device in ["TES", "CTES", "BAT"]:
        model.Add(cap[device] <= devs[device]["max_cap"])

    # Calculate the device capacities from piece-wise linear function variables      
    if param["switch_cost_functions"]:
        # TS: Not implemented yet!!! (SOS2 would have to be reformulated to binary variables...)
#        for device in ["BOI", "CHP", "AC", "CC", "TES", "CTES", "BAT"]:
#        
#            model.addConstr(cap[device] == sum(lin[device][i] * devs[device]["cap_i"][i] for i in range(len(devs[device]["cap_i"]))))
#            # lin: Special Ordered Sets of type 2 (SOS2 or S2): an ordered set of non-negative variables, of which at most two can be non-zero, and if 
#            # two are non-zero these must be consecutive in their ordering. 
#            model.addSOS(gp.GRB.SOS_TYPE2, [lin[device][i] for i in range(len(devs[device]["cap_i"]))])
#            
#            # Sum of linear function variables should be 1
#            model.addConstr(1 == sum(lin[device][i] for i in range(len(devs[device]["cap_i"]))))
        pass
            
        
    
    #%% LOAD CONTRAINTS: minimal load < load < capacity

    # Maximum and minimum storage soc
    for device in ["TES", "CTES", "BAT"]:
        for day in year:
            for t in time_steps:    
                model.Add(soc[device][day][t] <= devs[device]["max_soc"] * cap[device])
                model.Add(soc[device][day][t] >= devs[device]["min_soc"] * cap[device])

    # Maximum storage charge and discharge
    for device in ["TES", "CTES", "BAT"]:
        for d in days:
            for t in time_steps:   
                model.Add(ch[device][d][t] <= devs[device]["max_ch"] * cap[device])
                model.Add(dch[device][d][t] <= devs[device]["max_dch"] * cap[device])
    
    # Generation devices
    for d in days:
        for t in time_steps:
            for device in ["BOI"]:
                model.Add(heat[device][d][t] <= cap[device])
                
            for device in ["CHP"]:
                model.Add(power[device][d][t] <= cap[device])
            
            for device in ["CC", "AC"]:
                model.Add(cool[device][d][t] <= cap[device])
                      
            # limitation of power from and to grid   
            model.Add(model.Sum([gas[device][d][t] for device in ["BOI", "CHP"]]) <= grid_limit_gas)       
            for device in ["from_grid", "to_grid"]:
                model.Add(power[device][d][t] <= grid_limit_el)
            
        

    #%% INPUT / OUTPUT CONSTRAINTS
    for d in days:
        for t in time_steps:
            # Boiler
            model.Add(gas["BOI"][d][t] == heat["BOI"][d][t] / devs["BOI"]["eta_th"])
            
            # Combined heat and power
            model.Add(power["CHP"][d][t] == heat["CHP"][d][t] / devs["CHP"]["eta_th"] * devs["CHP"]["eta_el"])
            model.Add(gas["CHP"][d][t] == heat["CHP"][d][t] / devs["CHP"]["eta_th"])
            
            # Compression chiller
            model.Add(cool["CC"][d][t] == power["CC"][d][t] * devs["CC"]["COP"][d][t])  
    
            # Absorption chiller
            model.Add(cool["AC"][d][t] == heat["AC"][d][t] * devs["AC"]["eta_th"])
        


     #%% STORAGE DEVICES
        
    for device in ["TES", "CTES", "BAT"]:
        
        for day in year:        
            
            for t in np.arange(1, len(time_steps)):
                
                # Energy balance: soc(t) = soc(t-1) + charge - discharge
                model.Add(soc[device][day][t] == soc[device][day][t-1] * (1-devs[device]["sto_loss"])
                    + ch[device][sigma[day]][t] * devs[device]["eta_ch"] 
                    - dch[device][sigma[day]][t] / devs[device]["eta_dch"])
            
            # Transition between two consecutive days
            if day > 0:
                model.Add(soc[device][day][0] == soc[device][day-1][len(time_steps)-1] * (1-devs[device]["sto_loss"])                      
                    + ch[device][sigma[day]][0] * devs[device]["eta_ch"] 
                    - dch[device][sigma[day]][0] / devs[device]["eta_dch"])
                
        # Cyclic year condition
        model.Add(soc[device][0][0] ==  soc[device][len(year)-1][len(time_steps)-1] * (1-devs[device]["sto_loss"]) 
                    + ch[device][sigma[0]][0] * devs[device]["eta_ch"] 
                    - dch[device][sigma[0]][0] / devs[device]["eta_dch"])


    #%% GLOBAL ENERGY BALANCES
    
    for d in days:
        for t in time_steps:
            # Heat balance
            model.Add(heat["BOI"][d][t] + heat["CHP"][d][t] + dch["TES"][d][t] == dem["heat"][d][t] + heat["AC"][d][t] + ch["TES"][d][t])
    
            # Electricity balance
            model.Add(power["CHP"][d][t] + power["from_grid"][d][t] + dch["BAT"][d][t] == power["to_grid"][d][t] + power["CC"][d][t] + ch["BAT"][d][t])
    
            # Cooling balance
            model.Add(cool["AC"][d][t] + cool["CC"][d][t] + dch["CTES"][d][t] == dem["cool"][d][t] + ch["CTES"][d][t])  
        
    
    # Assure that real building peak loads are met despite demand clustering
    # heating
    model.Add(cap["CHP"]/devs["CHP"]["eta_el"]*devs["CHP"]["eta_th"] + cap["BOI"] >= param["peak_heat"])        
    # cooling
    model.Add(cap["CC"] + cap["AC"] >= param["peak_cool"])
    
    
    for d in days:
        for t in time_steps:
            # AC and TES can only be supplied by BOI and CHP
            model.Add(heat["BOI"][d][t] + heat["CHP"][d][t] >= heat["AC"][d][t] + ch["TES"][d][t])
            # CTES can only be supplied by CC and AC
            model.Add(cool["AC"][d][t] + cool["CC"][d][t] >= ch["CTES"][d][t])
    
    # Feed in
    for d in days:
        for t in time_steps:
            model.Add(feed_in["CHP"][d][t] == power["to_grid"][d][t])
            
        
    #%% SUM UP RESULTS
    
    # Total amound of gas taken from grid
    model.Add(gas_total == model.Sum([
            model.Sum([
                    model.Sum([gas[device][d][t] for t in time_steps]) * param["day_weights"][d] for d in days]) 
            for device in ["BOI", "CHP"]]))
  
    # Total electric energy from and to grid
    model.Add(from_grid_total == model.Sum([model.Sum([power["from_grid"][d][t] for t in time_steps]) * param["day_weights"][d] for d in days]))
    model.Add(to_grid_total == model.Sum([model.Sum([power["to_grid"][d][t] for t in time_steps]) * param["day_weights"][d] for d in days]))

    # Costs for electric energy taken from grid
    model.Add(electricity_costs == model.Sum([model.Sum([(power["from_grid"][d][t] * param["price_el"][d][t]) for t in time_steps]) * param["day_weights"][d] for d in days]))
    
    # Revenue for electric energy feed-in
    model.Add(revenue_feed_in["CHP"] == model.Sum([model.Sum([(power["to_grid"][d][t] * param["revenue_feed_in_CHP"][d][t]) for t in time_steps]) * param["day_weights"][d] for d in days]))

    
    # Total investment costs
    for device in all_devs:
        if param["switch_cost_functions"]:
            model.Add( inv[device] == model.Sum([lin[device][i] * devs[device]["inv_i"][i] for i in range(len(devs[device]["cap_i"]))]) )
        else:
            model.Add(inv[device] == devs[device]["inv_var"] * cap[device])        
        
    # Annual investment costs
    for device in all_devs:
        model.Add( c_inv[device] == inv[device] * devs[device]["ann_factor"] )
    
    # Operation and maintenance costs
    for device in all_devs:       
        model.Add( c_om[device] == devs[device]["cost_om"] * inv[device] )
    
    # Total annual costs
    for device in all_devs:
        model.Add( c_total[device] == c_inv[device] + c_om[device] )
        

    #%% OBJECTIVE FUNCTIONS
    # TOTAL ANNUALIZED COSTS
    model.Add(obj["tac"] == model.Sum([c_total[dev] for dev in all_devs])                                           # annualized device investment costs            
                                  + gas_total * param["price_gas"] + grid_limit_gas * param["price_cap_gas"]      # gas costs
                                  + electricity_costs + grid_limit_el * param["price_cap_el"]                     # electricity costs
                                  - revenue_feed_in["CHP"]   	                                                  # revenue for grid feed-in
                                  , "sum_up_TAC")                                    
    
    # ANNUAL CO2 EMISSIONS: Implicit emissions by power supply from national grid is penalized, feed-in is ignored
    model.Add(obj["co2_gross"] == gas_total * param["gas_CO2_emission"] + from_grid_total * param["grid_CO2_emission"], "sum_up_gross_CO2_emissions")

    
    
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Set model parameters and execute calculation
    
    print("Precalculation and model set up done in %f seconds." %(time.time() - start_time))
    
    # Set solver parameters
    solver_params = pywraplp.MPSolverParameters()
    solver_params.SetDoubleParam(solver_params.RELATIVE_MIP_GAP, param["MIPGap"])
#    model.Params.MIPGap     = param["MIPGap"]             # ---,  gap for branch-and-bound algorithm
#    model.Params.method     = 2                           # ---, -1: default, 0: primal simplex, 1: dual simplex, 2: barrier, etc. (only affects root node)
#    model.Params.Heuristics = 0                           # Percentage of time spent on heuristics (0 to 1)
#    model.Params.MIPFocus   = 2                           # Can improve calculation time (values: 0 to 3)
#    model.Params.Cuts       = 2                           # Cut aggressiveness (values: -1 to 3)
#    model.Params.PrePasses  = 8                           # Number of passes performed by presolving (changing can improve presolving time) values: -1 to inf
    
    # Execute calculation
    start_time = time.time()

    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Define objective function
    model.Minimize(obj[obj_fn])
    print("-----------\nSingle-objective optimization with objective function: " + obj_fn)

    status = model.Solve(solver_params)

    print("Optimization done. (%f seconds.)" %(time.time() - start_time))
    
   
    #%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    # Check if optimal solution was found
    if status in (model.FEASIBLE, model.OPTIMAL):
        # Print objectives
        for obj_function in ("co2_gross", "tac"):
            print(f"{obj_function}: {str(obj[obj_function].SolutionValue())}")
        
        # Store parameters as json-file       
        # param
        for item in ["day_matrix", "day_weights", "price_el", "sigma", "T_air", "revenue_feed_in_CHP"]:
            param[item] = param[item].tolist()               
        
        # Store demands as json-file
        for item in ["heat", "cool"]:
            dem[item] = dem[item].tolist()
        
        result_capacities = {}
        for device, variable in cap.items():
            result_capacities[device] = variable.SolutionValue()
            print(f"Capacity of device {device}: {variable.SolutionValue()}")
    
        return param, result_capacities
    
    else:
        model.computeIIS()
        print('Optimization result: No feasible solution found.')
