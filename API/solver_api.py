import math
from time import perf_counter
import pandas as pd
import numpy as np
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
import uvicorn

# importing solver functions.
from Finite_Differences.model import HeatTransfer
from Finite_Differences.explicit_euler import ExplicitEuler
from Finite_Differences.implicit_euler import ImplicitEuler
from Finite_Differences.crank_nicolson import CrankNicolson
from Finite_Differences.theta import Theta


""" Multi-Thread Options """

executor = ThreadPoolExecutor(5)


""" Validator """

class Problem(BaseModel):
    equation: dict
    setting: dict
    solver: dict

    @validator("equation")
    def validate_equation(cls, equation):
        return equation
    
    @validator("setting")
    def validate_equation(cls, setting):
        return setting
    
    @validator("solver")
    def validate_equation(cls, solver):
        return solver


""" Encoder Class """

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


""" Helper Functions """

def initialize_class(problem: Problem) -> HeatTransfer:
    # defining setting variables.
    dim = int(problem.setting["dim"])
    h_const = float(problem.setting["h"])
    k_const = float(problem.setting["k"])
    time_range = problem.setting["time"]
    omega = problem.setting["omega"]

    # defining equation variables.
    f_eq = problem.equation["goal"]
    v_eq = problem.equation["initial"]
    g_eq = problem.equation["boundary"]

    # creating the class.
    if problem.solver["solver"] == "explicit":
        model = ExplicitEuler(dim, f_eq, v_eq, g_eq, h_const, k_const, omega, time_range)
    elif problem.solver["solver"] == "implicit":
        model = ImplicitEuler(dim, f_eq, v_eq, g_eq, h_const, k_const, omega, time_range)
    elif problem.solver["solver"] == "crank-nicolson":
        model = CrankNicolson(dim, f_eq, v_eq, g_eq, h_const, k_const, omega, time_range)
    elif problem.solver["solver"] == "theta":
        model = Theta(dim, f_eq, v_eq, g_eq, h_const, k_const, omega, time_range, problem.solver["theta"])
    return model


""" App """

app = FastAPI()
app.add_middleware(
    CORSMiddleware, 
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"]
)

@app.get("/")
async def home():
    return {"message": "hello to the solver FastAPI!"}

@app.get("/version")
async def get_version():
    return {"solver": "1.0"}

@app.get("/solvers")
async def get_solvers():
    return {
        "1": "exact",
        "2": "explicit_euler",
        "3": "implicit_euler",
        "4": "crank_nicolson",
        "5": "theta"
    }

@app.post("/solve")
async def solver_equation(problem: Problem) -> dict:
    start_time = perf_counter()
    print(f"SOLVER ==> Problem: {problem}")

    # initializing the problem.
    model = initialize_class(problem)
    model.initialize_equations()
    model.initialize_omega()
    model.initialize_matrices()
    print(f"SOLVER ==> Initialized everything. ({perf_counter() - start_time})")

    # calculating the results.
    results = model.calculate_solution(float(problem.setting["time"]))
    end_time = perf_counter() - start_time
    print(f"SOLVER ==> Done calculating Results. ({end_time})")

    return {
        "message": "received data",
        "data": json.dumps(results, cls=NumpyEncoder),
        "duration": end_time
    }


""" Testing """

if __name__ == "__main__":
    print("--> running.")
    uvicorn.run("solver_api:app", port=8001, host="127.0.0.1")
