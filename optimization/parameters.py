# -*- coding: utf-8 -*-
import math
import numpy as np
from .clustering_medoid import cluster

def load_params(path_file):
    
    # Path for input data
    path_input = path_file + "\\input_data\\"     
        
    # load building data 
    path_nodes = path_input + "nodes.txt"
    path_demands = path_input + "demands\\"
    names = np.genfromtxt(open(path_nodes, "rb"),dtype = 'str', delimiter = ",", usecols=(3))           # --,       node names            
    # Create dictionary with node data
    nodes = {}
    for index in range(len(names)):
        nodes[index] = {
                        "number": index,
                        "name": names[index],
                        "heat": np.loadtxt(open(path_demands + names[index] + "_heating.txt", "rb"),delimiter = ",", usecols=(0)),       # kW, node heating demand
                        "cool": np.loadtxt(open(path_demands + names[index] + "_cooling.txt", "rb"),delimiter = ",", usecols=(0)),       # kW, node cooling demand                                                                                
                        }
    
        
#%% GENERAL PARAMETERS
    param = {
            "interest_rate":  0.05,         # ---,          interest rate
             "observation_time": 20.0,      # a,            project lifetime
               
             # Gap for branch-and-bound
             "MIPGap":      1,           # ---,          MIP gap              
             
             # Parameter switches
             "switch_variable_price": 1,            # ---,      1: variable electricity price
             "switch_var_revenue": 1,               # ---,      1: variable feed-in revenue                                                                              
             "switch_cost_functions": 0,            # ---,      1: Use piece-wise linear cost functions for BU devices, 0: Use constant specific investment costs (kEUR/MW)
             "switch_post_processing": 1,           # ---,      post processing on / off
                          
             # Type-day clustering
             "n_clusters": 12,                    
                          
             # BU Devices
             "feasible_TES": 1,             # ---,          are thermal energy storages feasible for BU?
             "feasible_BAT": 1,             # ---,          are batteries feasible for BU?
             "feasible_CTES": 1,            # ---,          are cold thermal energy storages feasible for BU?
             "feasible_BOI": 1,             # ---,          are gas-fired boilers feasible for BU?
             "feasible_CHP": 1,             # ---,          are CHP units feasible for BU?
             "feasible_CC": 1,              # ---,          are compression chiller feasible for BU?
             "feasible_AC": 1,              # ---,          are absorption chiller feasible for BU?
             }
    
    
    
    #%% ELECTRICITY PRICES
    
    # Price for electricity taken from grid
    if param["switch_variable_price"]:
        # Load EPEX SPOT electricity prices from 2015
        spot_prices = np.loadtxt(open(path_input + "Spotpreise15.txt", "rb"), delimiter = ",",skiprows = 0, usecols=(0)) / 1000        # kEUR/MWh 
        param["price_el"] = 0.10808 + spot_prices       # kEUR/MWh
    else:
        param["price_el"] = 0.14506 * np.ones(8760)     # kEUR/MWh
    
    
    # Feed-in revenue
    param["revenue_feed_in"] = {}
    if param["switch_var_revenue"]:
        # Load time series of feed-in revenue
        param["revenue_feed_in_CHP"] = np.loadtxt(open(path_input + "revenue_feed_in.txt", "rb"), delimiter = ",",skiprows = 0, usecols=(0)) / 1000        # kEUR/MWh 
    else:
        param["revenue_feed_in_CHP"] = 0.06 * np.ones(8760)         # kEUR/MWh                 
        
    # Grid capacity price
    param["price_cap_el"] =  59.660                         # kEUR/MW        
    

 
    
    #%% GAS PRICES
    
    # Gas price per MWh
    param["price_gas"] = 0.02824     # kEUR/MWh
    
    # Grid capacity price
    param["price_cap_gas"] = 12.149  # kEUR/MW



    
    #%% CO2 EMISSIONS
    
    param["gas_CO2_emission"] = 0.201       # t_CO2/MWh,    specific CO2 emissions (natural gas)
    param["grid_CO2_emission"] = 0.516      # t_CO2/MWh,    specific CO2 emissions (grid)
    
    
    #%% PRIMARY ENERGY FACTORS
    
    param["PEF_gas"] = 1.1
    param["PEF_power"] = 1.8    
    
    
    #%% TEMPERATURE PARAMETERS
    
    # cooling network
    param["T_supply_cooling"] = 14       # °C, cooling supply temperature (it is needed to calculate the compression chiller COP)
    param["T_return_cooling"] = 18       # °C, cooling return temperature (it is needed to calculate the compression chiller COP)

    # consumers supply temperatures (they are needed to calculate exergetic efficiency)
    param["T_heating_consumers"] = 60        # °C,   heating supply temperature
    param["T_cooling_consumers"] = 16        # °C,   cooling supply temperature
    param["T_ref"] = 25                      # °C,   reference temperature for exergy calculation
    
    #%% WEATHER DATA
    param["T_air"] = np.loadtxt(open(path_input + "weather.csv", "rb"), delimiter = ",",skiprows = 1, usecols=(0))       # °C,    air temperature
   
    #%% WATER PROPERTIES
    param["rho_f"] = 1000       # kg/m^3,   density
    param["c_f"] = 4180         # J/(kg*K)  specific heat capacity
    
    
    #%% SUM UP DEMANDS
    
    dem = {}
    dem["heat"] = np.sum(nodes[n]["heat"] for n in nodes) / 1000        # MW,      time series of total heating demand
    dem["cool"] = np.sum(nodes[n]["cool"] for n in nodes) / 1000        # MW,      time series of total cooling demand

    # Consider thermal network losses    
    heat_loss = np.loadtxt(open(path_input + "heat_losses_low_temperatures.txt", "rb"),delimiter = ",", usecols=(0)) # MW, time series of network heat losses (low temperature scenario, ECOS paper)
#    heat_loss = np.loadtxt(open(path_input + "heat_losses_high_temperatures.txt", "rb"),delimiter = ",", usecols=(0)) # MW, time series of network heat losses (high temperature scenario, ECOS paper)
    cool_loss = np.loadtxt(open(path_input + "cool_losses.txt", "rb"),delimiter = ",", usecols=(0)) # MW, time series of network losses in cooling network (ECOS paper)

    dem["heat"] += heat_loss
    dem["cool"] += cool_loss

    # Check small values
    for demand in ["heat", "cool"]:
        for t in range(8760):
            if dem[demand][t] < 0.001:
                dem[demand][t] = 0
            
    
    #%% CLUSTER TIME SERIES INTO TYPE-DAYS
        
        
    # Save peak demands
    param["peak_heat"] = np.max(dem["heat"])
    param["peak_cool"] = np.max(dem["cool"])
    

    # Collect demand time series    
    # Only the demand time series are clustered using k-medoids algorithm.
    time_series = []
    for demand in ["heat", "cool"]:
        time_series.append(dem[demand])        
    inputs_clustering = np.array(time_series)
    
    # Execute clustering algorithm
    (clustered_series, nc, z) = cluster(inputs_clustering, 
                                     param["n_clusters"],
                                     norm = 2,
                                     mip_gap = 0.01, 
                                     )
    
     # Retrieve clustered time series
    dem["heat"] = clustered_series[0]
    dem["cool"] = clustered_series[1]
    
    
    # save frequency of typical days
    param["day_weights"] = nc
    # Save correlation between type days and days of the year
    param["day_matrix"] = z
    
    # For each day of the year, find the corresponding type-day
    # Collect days used as typedays
    typedays = np.zeros(param["n_clusters"], dtype = np.int32)
    d = 0
    for y in range(365):
        if any(z[y]):
            typedays[d] = y
            d += 1
    # Assign each day of the year to its typeday
    sigma = np.zeros(365, dtype = np.int32)
    for y in range(len(sigma)):
        d = np.where(z[:,y] == 1)[0][0]
        sigma[y] = np.where(typedays == d)[0][0]
    param["sigma"] = sigma
          
            
    # Cluster secondary time series manually according to the results of k-medoid algorithm
    for series in ["T_air", "price_el", "revenue_feed_in_CHP"]:
        series_clustered = np.zeros((param["n_clusters"], 24))
        for d in range(param["n_clusters"]):
            for t in range(24):
                series_clustered[d][t] = param[series][24*typedays[d]+t]
        # Replace original time series with the clustered one
        param[series] = series_clustered
       
    
    
    #%% LOAD DEVICE PARAMETERS
    
    devs = {}
    
    #%% BOILER
    
    devs["BOI"] = {
                   "eta_th": 0.9,       # ---,    thermal efficiency
                   "life_time": 20,     # a,      operation time (VDI 2067)
                   "cost_om": 0.03,     # ---,    annual operation and maintenance costs as share of investment (VDI 2067)
                   }
    
    # Boiler investment costs
    
    # Piece-wise linear specific investment costs
    if param["switch_cost_functions"]:
        devs["BOI"]["cap_i"] =  {  0: 0,        # MW_th 
                                   1: 0.5,      # MW_th
                                   2: 5         # MW_th
                                   }
        
        devs["BOI"]["inv_i"] = {    0: 0,       # kEUR
                                    1: 33.75,   # kEUR
                                    2: 96.2     # kEUR
                                    }
    # Constant specific investment costs
    else:
        devs["BOI"]["inv_var"] = 67.5
        

    #%% COMBINED HEAT AND POWER - INTERNAL COMBUSTION ENGINE POWERED BY NATURAL GAS
    
    devs["CHP"] = {
                   "eta_el": 0.419,     # ---,           electrical efficiency
                   "eta_th": 0.448,     # ---,           thermal efficiency
                   "life_time": 15,     # a,             operation time (VDI 2067)
                   "cost_om": 0.08,     # ---,           annual operation and maintenance costs as share of investment (VDI 2067)
                   }   
    
    if param["switch_cost_functions"]:
        devs["CHP"]["cap_i"] =  {  0: 0,        # MW_el
                                   1: 0.25,     # MW_el
                                   2: 1,        # MW_el
                                   3: 3         # MW_el
                                   }
        
        devs["CHP"]["inv_i"] = {    0: 0,           # kEUR
                                    1: 211.15,      # kEUR
                                    2: 410.7,       # kEUR
                                    3: 707.6        # kEUR
                                    } 
    else:
        devs["CHP"]["inv_var"] = 750
        

    
    #%% ABSORPTION CHILLER
    
    devs["AC"] = {
                  "eta_th": 0.68,       # ---,        nominal thermal efficiency (cooling power / heating power)
                  "life_time": 18,      # a,          operation time (VDI 2067)
                  "cost_om": 0.03,      # ---,        annual operation and maintenance costs as share of investment (VDI 2067)
                  }
    
    if param["switch_cost_functions"]:   
        devs["AC"]["cap_i"] =   {  0: 0,        # MW_th
                                   1: 0.25,     # MW_th
                                   2: 3,        # MW_th
    
                                   }
        
        devs["AC"]["inv_i"] = {     0: 0,           # kEUR
                                    1: 131.3,       # kEUR
                                    2: 763.2        # kEUR
                                    } 
    else:
         devs["AC"]["inv_var"] = 525


    


    #%% GROUND SOURCE HEAT PUMP
#    
#    devs["HP"] = {                  
#                  "dT_pinch": 2,                                         # K,    temperature difference between heat exchanger sides at pinch point
#                  "dT_min_soil": 2,                                      # K,    minimal temperature difference between soil and brine
#                  "life_time": 20,                                       # a,    operation time (VDI 2067)
#                  "cost_om": 0.025,                                      #---,   annual operation and maintenance as share of investment (VDI 2067)
#                  "dT_evap": 5,                                          # K,    temperature difference of water in evaporator
#                  "dT_cond": param["T_hot"] - param["T_cold"],           # K,    temperature difference of water in condenser
#                  "eta_compr": 0.8,                                      # ---,  isentropic efficiency of compression
#                  "heatloss_compr": 0.1,                                 # ---,  heat loss rate of compression
#                  "COP_max": 7,                                          # ---,  maximum heat pump COP
#                  "q_soil": 50,                                          # W/m,   heat flow from soil into bride per meter (VDI 4640, for lambda_soil = 2 W/mK and low full load hours, assumption: no thermal interaction between boreholes)
#                  "c_borehole": 90,                                      # EUR/m, borehole costs (BMVBS)
#                  "t_max": 400                                           # m,     maximum borehole depth covered by VDI4640
#                  }
#    
#    # Temperatures
#    t_c_in = param["T_soil_deep"] - devs["HP"]["dT_min_soil"] + 273.15          # heat source inlet (deep soil temperature - minimal temperature difference)
#    dt_c= devs["HP"]["dT_evap"]                                                 # heat source temperature difference
#    t_h_in = param["T_cold"] + 273.15                                           # heat sink inlet temperature 
#    dt_h = devs["HP"]["dT_cond"]                                                # cooling water temperature difference
# 
#    # Calculate heat pump COPs
#    devs["HP"]["COP"] = calc_COP(devs, param, "HP", [t_c_in, dt_c, t_h_in, dt_h])
#    
#    # Piece-wise linear cost function including borehole-costs
#    if param["switch_cost_functions"]:
#        devs["HP"]["cap_i"] =   {  0: 0,        # MW_th
#                                   1: 0.2,      # MW_th
#                                   2: 0.5,      # MW_th
#                                   3: 2         # MW_th
#                                   }
#        
#        devs["HP"]["inv_i"] = {     0: 0,           # kEUR
#                                    1: 388.57,      # kEUR
#                                    2: 921.43,      # kEUR
#                                    3: 3385.72      # kEUR
#                                    } 
#    else:
#        devs["HP"]["inv_var"] = 1000


    #%% COMPRESSION CHILLER
    
    devs["CC"] = {
                  "life_time": 15,        # a,               operation time (VDI 2067)
                  "cost_om": 0.035,       # ---,             annual operation and maintenance costs as share of investment (VDI 2067)
                  "dT_cond": 5,           # K,               temperature change of cooling water in condenser
                  "dT_min_cooler": 10,    # K,               minimum temperature difference between air and cooling water
                  "dT_pinch": 2,          # K,               minimum temperature difference in evaporator/condenser 
                  "eta_compr": 0.75,      # ---,             isentropic efficiency of compression
                  "heatloss_compr": 0.1,  # ---,             heat loss rate of compression 
                  "COP_max": 6            # ---,             maximum COP
                  }
    
    # Temperatures
    t_c_in = param["T_return_cooling"] *np.ones((param["n_clusters"],24)) + 273.15      # evaporater inlet temperatur     
    dt_c = param["T_return_cooling"] - param["T_supply_cooling"]                        # evaporator temperature difference   
    t_h_in = param["T_air"] + devs["CC"]["dT_min_cooler"] + 273.15                      # condenser inlet temperature
    dt_h = devs["CC"]["dT_cond"]                                                        # condenser temeprature difference

    # Calculate CC COP
    devs["CC"]["COP"] = calc_COP(devs, param, "CC", [t_c_in, dt_c, t_h_in, dt_h])
    
    
    if param["switch_cost_functions"]:
        devs["CC"]["cap_i"] = { 0: 0,       # MW_th
                                1: 0.5,     # MW_th
                                2: 4        # MW_th
                                }
        
        
        devs["CC"]["inv_i"] = { 0: 0,      # kEUR
                                1: 111,     # kEUR
                                2: 632.2     # kEUR
                                } 
    else:
        devs["CC"]["inv_var"] = 170


    #%% THERMAL ENERGY STORAGES
    
    for device in ["TES", "CTES"]:
    
        devs[device] = { 
                       "max_ch": 0.25,      # 1/h,              maximum soc change per hour by charging
                       "max_dch": 0.25,     # 1/h,              maximum soc change per hour by discharging
                       "min_soc": 0,        # ---,              minimum state of charge
                       "max_soc": 1,        # ---,              maximum state of charge
                       "sto_loss": 0.005,   # 1/h,              standby losses over one time step
                       "eta_ch": 0.95,      # ---,              charging efficiency
                       "eta_dch": 0.95,     # ---,              discharging efficiency
                       "life_time": 20,     # a,                operation time (VDI 2067 Trinkwasserspeicher)
                       "cost_om": 0.02,     # ---,              annual operation and maintenance costs as share of investment (VDI 2067 Trinkwasserspeicher)   
                       }
        
    # difference between minimum and maximum temperature
    devs["TES"]["dT"] = 30     # K                                                    
    devs["CTES"]["dT"] = 10    # K                                                          
        
    # Investment costs
    # piece-wise linear
    if param["switch_cost_functions"]:
        for device in ["TES", "CTES"]: 
    
            devs[device]["V_i"] = { 0: 0,           # m^3
                                    1: 100,         # m^3
                                    }
            
    
            devs[device]["inv_i"] = {   0: 0,            # kEUR
                                        1: 640,          # kEUR,        
                                        }
            
            devs[device]["cap_i"] = {}
            for i in range(len(devs[device]["V_i"])):
                # Calculate storage capacity in MWh out of storage volume and temperature difference
                devs[device]["cap_i"][i] = param["rho_f"] * devs[device]["V_i"][i] * param["c_f"] * devs[device]["dT"] / (1e6 * 3600)
            devs[device]["max_cap"] = devs[device]["cap_i"][len(devs[device]["cap_i"])-1]
    
    # constant
    else:
        for device in ["TES", "CTES"]:
            # investment costs per m^3
            devs[device]["inv_vol"] = 640      # EUR/m^3
            devs[device]["V_max"] = 100        # m^3
            # calculate investment costs per MWh
            devs[device]["inv_var"] = devs[device]["inv_vol"] / (param["rho_f"] * param["c_f"] * devs[device]["dT"]) * 1000 * 3600      # kEUR/MWh
            devs[device]["max_cap"] = devs[device]["V_max"] * param["rho_f"] * param["c_f"] * devs[device]["dT"] /(1e6 * 3600)          # MWh


      
    #%% BATTERY STORAGE
    
    devs["BAT"] = {
                   "max_ch": 0.333,         # 1/h,              maximum soc change per hour by charging
                   "max_dch": 0.333,        # 1/h,              maximum soc change per hour by discharging
                   "min_soc": 0.2,          # ---,              minimum state of charge
                   "max_soc": 0.8,          # ---,              maximum state of charge
                   "sto_loss": 0.001,       # 1/h,              standby losses over one time step
                   "eta_ch": 0.96,          # ---,              charging efficiency      
                   "eta_dch": 0.96,         # ---,              discharging efficiency
                   "life_time": 10,         # a,                operation time
                   "cost_om": 0.01,         # ---,              annual operation and maintenance costs as share of investment
                   } 


    if param["switch_cost_functions"]:
        devs["BAT"]["cap_i"] = { 0: 0,      # MWh_el
                                 1: 100,    # MWh_el
                                }
        
        
        devs["BAT"]["inv_i"] = { 0: 0,       # kEUR
                                 1: 1000,    # kEUR
                                } 

        devs["BAT"]["max_cap"] = devs[device]["cap_i"][len(devs[device]["cap_i"])-1]

    else:
        
        devs["BAT"]["inv_var"] = 1000       # kEUR/MWh
        devs["BAT"]["max_cap"] = 100        # MWh
        
    #%% PV
    
#    # PV module parameters based on model LG Solar LG360Q1C-A5 NeON R
#    # https://www.lg.com/global/business/download/resources/solar/DS_NeONR_60cells.pdf
#    devs["PV"] =        {
#                            "eta_el_stc": 0.208,        # ---,     electrical efficiency under standard test conditions (STC)
#                            "t_cell_stc": 25,           # °C
#                            "G_stc": 1000,              # W/m^2
#                            "t_cell_noct": 44,          # °C       nominal operation cell temperature (NOCT)
#                            "t_air_noct": 20,           # °C,
#                            "G_noct": 800,              # W/m^2,       
#                            "gamma": -0.003,            # 1/K,
#                            "eta_opt": 0.9,             # ---,     optical efficiency according to https://www.homerenergy.com/products/pro/docs/3.11/solar_transmittance.html
#                            "life_time": 20,            # a,       operation time (VDI 2067)
#                            "inv_var": 900,             # EUR/kW,  PV investment costs   (https://www.photovoltaik4all.de/lg-solar-lg360q1c-a5-neon-r)
#                            "inv_fix": 0,               # EUR  
#                            "cost_om": 0.01,            #---,      annual operation and maintenance as share of investment (VDI)
#                            "max_area": 8000            # m^2,     maximum free roof area for PV installation
#                            } 
#
#
#    # Calculate PV efficiency time series
#    # Cell temperature according to https://www.homerenergy.com/products/pro/docs/3.11/how_homer_calculates_the_pv_cell_temperature.html   
#    t_cell = (param["t_air"] + (devs["PV"]["t_cell_noct"] - devs["PV"]["t_air_noct"])*(param["G_sol"]/devs["PV"]["G_noct"])*(1 - (devs["PV"]["eta_el_stc"]*(1-devs["PV"]["gamma"]*devs["PV"]["t_cell_stc"]))/devs["PV"]["eta_opt"])) / (
#             (1 + (devs["PV"]["t_cell_noct"] - devs["PV"]["t_air_noct"])*(param["G_sol"]/devs["PV"]["G_noct"])*((devs["PV"]["gamma"]*devs["PV"]["eta_el_stc"])/devs["PV"]["eta_opt"])))    
#    devs["PV"]["eta_el"] = devs["PV"]["eta_el_stc"] * (1 + devs["PV"]["gamma"] * (t_cell - devs["PV"]["t_cell_stc"]))
#    
    

    # Calculate annualized investment of every device
    devs = calc_annual_investment(devs, param)
    
    return dem, param, devs



    
#%%
def calc_annual_investment(devs, param):
    """
    Calculation of total investment costs including replacements (based on VDI 2067-1, pages 16-17).

    Parameters
    ----------
    dev : dictionary
        technology parameter
    param : dictionary
        economic parameters

    Returns
    -------
    annualized fix and variable investment
    """

    observation_time = param["observation_time"]
    interest_rate = param["interest_rate"]
    q = 1 + param["interest_rate"]

      
    # Calculate capital recovery factor
    CRF = ((q**observation_time)*interest_rate)/((q**observation_time)-1)
    
    # Calculate annuity factor for each device
    for device in devs.keys():
        
        # Get device life time
        life_time = devs[device]["life_time"]

        # Number of required replacements
        n = int(math.floor(observation_time / life_time))
        
        # Inestment for replcaments
        invest_replacements = sum((q ** (-i * life_time)) for i in range(1, n+1))

        # Residual value of final replacement
        res_value = ((n+1) * life_time - observation_time) / life_time * (q ** (-observation_time))

        # Calculate annualized investments       
        if life_time > observation_time:
            devs[device]["ann_factor"] = (1 - res_value) * CRF 
        else:
            devs[device]["ann_factor"] = ( 1 + invest_replacements - res_value) * CRF 
       

    return devs



#%% COP model for ammonia-heat pumps by Jensen et al.
# Heat pump COP, part 2: Generalized COP estimation of heat pump processes
# DOI: 10.18462/iir.gl.2018.1386
    
def calc_COP(devs, param, device, temperatures):
    
    # temperatures: array containing temperature information
    # temperatures = [heat source inlet temperature, heat source temperature difference, heat sink inlet temperature, heat sink temperature difference]
    # each array element can be an array itself or a single value
    # Temperatures must be given in Kelvin !
    
    
    # Get temperature parameters
    t_c_in = temperatures[0]
    dt_c = temperatures[1]
    t_h_in = temperatures[2]
    dt_h = temperatures[3]
    
    
    # Device parameters
    dt_pp = devs[device]["dT_pinch"]                # pinch point temperature difference
#    dt_pp = 50
    eta_is = devs[device]["eta_compr"]              # isentropic compression efficiency
    f_Q = devs[device]["heatloss_compr"]            # heat loss rate during compression
    
    # Entropic mean temperautures
    t_h_s = dt_h/np.log((t_h_in + dt_h)/t_h_in)
    t_c_s = dt_c/np.log(t_c_in/(t_c_in - dt_c))
    
    # Prevent numerical issues
    for d in range(param["n_clusters"]):
        for t in range(24):
            if t_h_s[d][t] == t_c_s[d][t]:
                t_h_s[d][t] += 1e-5
    
    #Lorenz-COP
    COP_Lor = t_h_s/(t_h_s - t_c_s)
    
    # Linear model equations
    dt_r_H = 0.2*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) + 0.2*dt_h + 0.016        # mean entropic heat difference in condenser deducting dt_pp
    w_is = 0.0014*(t_h_in + dt_h - (t_c_in - dt_c) + 2*dt_pp) - 0.0015*dt_h + 0.039    # ratio of isentropic expansion work to isentropic compression work
    
    
    # Help values
    num = 1 + (dt_r_H + dt_pp)/t_h_s
    denom = 1 + (dt_r_H + 0.5*dt_c + 2*dt_pp)/(t_h_s - t_c_s)
    
    # COP
    COP = COP_Lor * num/denom * eta_is * (1 - w_is) + 1 - eta_is - f_Q

    if device == "CC":
        COP = COP - 1   # consider COP definition for compression chillers (COP_CC = Q_0/P_el = (Q - P_el)/P_el = COP_HP - 1)
    
    # Limit COPs to avoid unrealistic large values
    COP_max = devs[device]["COP_max"]
    for d in range(param["n_clusters"]):
        for t in range(24):
            if COP[d,t] > COP_max or COP[d,t] < 0:
                COP[d,t] = COP_max

    return COP
