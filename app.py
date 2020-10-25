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