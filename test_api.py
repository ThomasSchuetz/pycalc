# -*- coding: utf-8 -*-
import requests

data = {"lhs": 10, "rhs": 15}
base_url = 'http://localhost:5000/'
headers = {"Accept": "application/json"}

for operation in ("add", "subtract", "multiply", "divide"):
    print()
    response = requests.post(base_url + operation,json = data)

    print(f"operation: {operation}")
    print(f"inputs: {data}")
    print(f"result: {response.json()}")