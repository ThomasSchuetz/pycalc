# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/optimize_sizing', methods = ['POST'])
def optimize_sizing():
    from optimization.run_optimization import run
    params, capacities = run()
    return jsonify( { "capacities": capacities } )


@app.route("/get_demands", methods = ["POST"])
def get_demands():
    from numpy import loadtxt
    import os
    path_file = str(os.path.dirname(os.path.realpath(__file__)))
    path_demands = path_file + "/optimization/input_data/demands/"
    heating_demand = loadtxt(path_demands + "15.1_heating.txt")
    cooling_demand = loadtxt(path_demands + "15.1_cooling.txt")
    return jsonify( {"heating": heating_demand, "cooling": cooling_demand})


@app.route('/add', methods = ['POST'])
def add():
    inputs = request.get_json()
    return jsonify( { "result": calculate(inputs["lhs"], inputs["rhs"], "+") } )


@app.route('/subtract', methods = ['POST'])
def subtract():
    inputs = request.get_json()
    return jsonify( { "result": calculate(inputs["lhs"], inputs["rhs"], "-") } )


@app.route('/multiply', methods = ['POST'])
def multiply():
    inputs = request.get_json()
    return jsonify( { "result": calculate(inputs["lhs"], inputs["rhs"], "*") } )


@app.route('/divide', methods = ['POST'])
def divide():
    inputs = request.get_json()
    return jsonify( { "result": calculate(inputs["lhs"], inputs["rhs"], "/") } )


def calculate(lhs, rhs, operation):
    calc = {
        "+": (lambda lhs, rhs: lhs + rhs),
        "-": (lambda lhs, rhs: lhs - rhs),
        "*": (lambda lhs, rhs: lhs * rhs),
        "/": (lambda lhs, rhs: lhs / rhs)
    }
    
    return calc[operation](lhs, rhs)