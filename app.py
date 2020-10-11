# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
app = Flask(__name__)

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