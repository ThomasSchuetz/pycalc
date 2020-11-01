# -*- coding: utf-8 -*-

from demgen import calc_results

building = {"floor_area": 100,}
heating_demand = {"spec_total_heat_demand": 50,
                  "spec_peak_heat_demand": 100, 
                  "heat_full_year": 50000,
                  "end_heating_period_month": 4,
                  "begin_heating_period_month": 10}
cooling_demand = {"spec_total_cool_demand": 10,
                  "spec_peak_cool_demand": 30, 
                  "cool_full_year": 5000,
                  "end_cooling_period_month": 8,
                  "begin_cooling_period_month": 6}
weather_file = "DEU_Dusseldorf.104000_IWEC.epw"
params = {"building": building, 
          "heating": heating_demand, 
          "cooling": cooling_demand, 
          "weather_file": weather_file}

result = calc_results(params)

import matplotlib.pyplot as plt
plt.plot(result["dem_dict"]["heat"], color="red")
plt.plot(result["dem_dict"]["cool"], color="blue")