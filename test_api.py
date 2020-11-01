# -*- coding: utf-8 -*-
import requests

data = {"lhs": 10, "rhs": 15}
base_url = "https://python-calc.herokuapp.com/"
base_url = "http://127.0.0.1:5000/"
headers = {"Accept": "application/json"}

for operation in ("add", "subtract", "multiply", "divide"):
    print()
    response = requests.post(base_url + operation,json = data)

    print(f"operation: {operation}")
    print(f"inputs: {data}")
    print(f"result: {response.json()}")


#print("Sizing results")
#response_sizing = requests.post(base_url + "optimize_sizing",json = {})
#print(f"Capacities: {response_sizing.json()}")

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
response_demgen = requests.post(base_url + "demgen",
                                json = {"building": building, 
                                        "heating": heating_demand, 
                                        "cooling": cooling_demand, 
                                        "weather_file": weather_file}
                                )
print(f"demgen: {response_demgen.json()}")