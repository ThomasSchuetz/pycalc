# -*- coding: utf-8 -*-

import os
import numpy as np

def calc_results(input_params):  # (request, context):
    type_demands = ("heat", "cool")
    
    days_sum = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365]

    param = {"floor_area": input_params["building"]["floor_area"]}  # m2

    # Heating demand
    param["heat"] = {
        "spec_total_demand": input_params["heating"]["spec_total_heat_demand"]
        * 1000,  # Wh/m2/a
        "spec_peak_demand": input_params["heating"]["spec_peak_heat_demand"],  # W/m2
        "full_year": input_params["heating"]["heat_full_year"],
        "end_period_month": input_params["heating"]["end_heating_period_month"],  # month of year: 1..12
        "begin_period_month": input_params["heating"]["begin_heating_period_month"],  # month of year: 1..12
    }

    param["cool"] = {
        "spec_total_demand": input_params["cooling"]["spec_total_cool_demand"] * 1000,  # Wh/m2/a
        "spec_peak_demand": input_params["cooling"]["spec_peak_cool_demand"],  # W/m2
        "full_year": input_params["cooling"]["cool_full_year"],
        "end_period_month": input_params["cooling"]["end_cooling_period_month"],  # month of year: 1..12
        "begin_period_month": input_params["cooling"]["begin_cooling_period_month"],  # month of year: 1..12
    }

    for m in type_demands:
        param[m]["annual_demand"] = param["floor_area"] * param[m]["spec_total_demand"]
        param[m]["peak_demand"] = param["floor_area"] * param[m]["spec_peak_demand"]

    path_file = str(os.path.dirname(os.path.realpath(__file__)))
    T_air, GHI, wind_speed = np.loadtxt(path_file + "/" + input_params["weather_file"], 
                                        delimiter=",", 
                                        skiprows=8, 
                                        usecols=[6,13,21], 
                                        unpack=True)

    param["air_temp"] = T_air  # degC
    
    if param["heat"]["full_year"]:
        param["heat"]["end_period"] = 8760 + 1
        param["heat"]["begin_period"] = -1
    else:
        param["heat"]["end_period"] = (
            24 * days_sum[param["heat"]["end_period_month"]]
        )  # hour of year: 0..8759
        param["heat"]["begin_period"] = (
            24 * days_sum[param["heat"]["begin_period_month"] - 1]
        )  # hour of year: 0..8759

    if param["cool"]["full_year"]:
        param["cool"]["begin_period"] = -1
        param["cool"]["end_period"] = 8760 + 1
    else:
        param["cool"]["begin_period"] = (
            24 * days_sum[param["cool"]["begin_period_month"] - 1]
        )  # hour of year: 0..8759
        param["cool"]["end_period"] = (
            24 * days_sum[param["cool"]["end_period_month"]]
        )  # hour of year: 0..8759

    course = {}
    sum_deg_day = {}
    peak = {}
    best_temp_set = {}
    for m in type_demands:
        deg_day = {}
        sum_deg_day[m] = {}
        peak[m] = {}
        course[m] = {}
        best_eps = 1e10

        for temp_set in [-20 + 0.5 * k for k in range(200)]:
            sum_deg_day[m][temp_set] = 0

            # Calculate degree days
            deg_day[temp_set] = np.zeros(8760)
            for t in range(8760):
                if m == "heat":
                    if (
                        (t + 1 < param["heat"]["end_period"])
                        or (t + 1 > param["heat"]["begin_period"])
                    ) and (param["air_temp"][t] < temp_set):
                        deg_day[temp_set][t] += temp_set - param["air_temp"][t]

                if m == "cool":
                    if (
                        (t + 1 < param["cool"]["end_period"])
                        and (t + 1 > param["cool"]["begin_period"])
                    ) and (param["air_temp"][t] > temp_set):
                        deg_day[temp_set][t] += param["air_temp"][t] - temp_set

            sum_deg_day[m][temp_set] = np.sum(deg_day[temp_set])

            # Calculate course over year
            course[m][temp_set] = (
                deg_day[temp_set] / sum_deg_day[m][temp_set] * param[m]["annual_demand"]
            )

            peak[m][temp_set] = np.max(course[m][temp_set])

            if abs(peak[m][temp_set] - param["heat"]["peak_demand"]) < best_eps:
                best_temp_set[m] = temp_set
                best_eps = abs(peak[m][temp_set] - param["heat"]["peak_demand"])
    
    hourly_dem = {}
    hour_in_minute = {}
    for m in type_demands:
        hour_in_minute[m] = np.zeros(8761)
        hourly_dem[m] = np.zeros(8761)
        for hour in range(8760):
            if hour==8760:
                hour_in_minute[m][hour+1] = hour_in_minute[m][0]
            else:    
                hourly_dem[m][hour] = np.round_(course[m][best_temp_set[m]][hour], decimals=2, out=None)
                hour_in_minute[m][hour] = course[m][best_temp_set[m]][hour]/60
            
    
    
    daily_dem = {}
    for m in type_demands:
        daily_dem[m] = np.zeros(365)
        for day in range (365):
            daily_dem[m][day] = np.round_(sum(
                course[m][best_temp_set[m]][t] for t in range(day * 24, (day +1) *24)
                ), decimals=2, out=None)

    
    minutes_dem = {}
    for m in type_demands:
        minutes_dem[m] = np.zeros(525600)
        for minute in range(525600):
            if minute/60==int:
                minutes_dem[m][minute] = np.round_(hour_in_minute[m][int(minute/60)], decimals=2, out=None)
            else:
                x = (hour_in_minute[m][int(minute/60)],hour_in_minute[m][int(minute/60 + 1)])
                y = (int(minute/60)*60,int(minute/60 + 1)*60)
                minutes_dem[m][minute] = np.round_(np.interp(minute, y, x), decimals=2, out=None)
           
                
    month_tuple = (
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    )

    month_dem = {}
    for m in type_demands:
            month_dem[m] = np.zeros(12)
            for month in range (12):
                month_dem[m][month]=np.round_(sum(
                    course[m][best_temp_set[m]][t] for t in range(days_sum[month] * 24, days_sum[month + 1] * 24)
                    ), decimals=2, out=None)


#%% EVALUATION

    monthly_dem = {}
    for m in type_demands:
        monthly_dem[m] = {}
        for month in range(12):
            monthly_dem[m][month_tuple[month]] = (
                sum(
                    course[m][best_temp_set[m]][t]
                    for t in range(days_sum[month] * 24, days_sum[month + 1] * 24)
                )
                / 1000
            )

    dem_dict = {}
    for demand_type in type_demands:
        dem_dict[demand_type] = [course[demand_type][best_temp_set[demand_type]][t] / 1000 
                 for t in range(8760)]
        

    result_dict = {
        "dem_dict": dem_dict,
        "total_dem": {},
        "monthly_dem": {},
        "real_peak_demand": {},
    }

    for m in type_demands:
        result_dict["total_dem"][m] = round(
            sum(course[m][best_temp_set[m]]) / 1000000, 0
        )  # MWh/a
        result_dict["monthly_dem"][m] = monthly_dem[m]
        result_dict["real_peak_demand"][m] = round(
            max(course[m][best_temp_set[m]]) / 1000, 0
        )  # kW

    return result_dict