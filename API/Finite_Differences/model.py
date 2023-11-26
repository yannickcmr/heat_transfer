import numpy as np
import math
from abc import ABC, abstractmethod
import itertools
import matplotlib.pyplot as plt


""" Helper Functions """

def convert_equation(equation: str):
    # replacing math functions
    equation = equation.replace("sin", "math.sin")
    equation = equation.replace("cos", "math.cos")
    equation = equation.replace("tan", "math.tan")

    # potential functions.
    equation = equation.replace("sqrt", "math.sqrt")
    equation = equation.replace("pow", "math.pow")

    # replacing logarithms.
    equation = equation.replace("log", "math.log")
    equation = equation.replace("log_2", "math.log2")
    equation = equation.replace("log_10", "math.log10")

    # replacing exponential functions.
    equation = equation.replace("exp", "math.exp")
    equation = equation.replace("exp_2", "math.exp2")

    # replacing constants.
    equation = equation.replace("pi", "math.pi")

    return equation

def convert_omega_boundary(value: float) -> float:
    value = str(value)
    value = convert_equation(value)
    return eval(value)

def convert_omega(omega: dict, dim: int):
    x = (convert_omega_boundary(omega["x"]["start"]), convert_omega_boundary(omega["x"]["end"]))
    y = (convert_omega_boundary(omega["y"]["start"]), convert_omega_boundary(omega["y"]["end"]))
    space = [x, y]
    return tuple(space[0:dim])

def calculate_inner_omega(omega: tuple, const: float, dim: int) -> list:
    # calculate the number of points per dimension.
    points = []
    for i in range(0, dim):
        delta = omega[i][1] - omega[i][0]
        m = int(np.around(delta/const))
        points.append(range(0, m -2, 1))

    # calculating all tuples.
    points = list(itertools.product(*points))
    return points

def calculate_outer_omega(omega: tuple, const: float, dim: int) -> list:
    # calculate the number of points per dimension.
    points = []
    for i in range(0, dim):
        delta = omega[i][1] - omega[i][0]
        m = int(np.around(delta/const))
        points.append(range(0, m, 1))

    # calculating all tuples.
    points = list(itertools.product(*points))
    return points

def plot_result(results: list, omega: dict, dim: int, y_range: list = [0,0], title: str = "test"):
    if dim == 2:
        ax = plt.figure().add_subplot(projection='3d')

        # plotting surface.
        x_surface = np.linspace(start=omega[0][0], stop=omega[0][1], num=results.shape[0])
        y_surface = np.linspace(start=omega[1][0], stop=omega[1][1], num=results.shape[1])
        x_surface, y_surface = np.meshgrid(x_surface, y_surface)

        ax.plot_surface(x_surface, y_surface, results)
        ax.set_title(title)

    else:
        x = np.linspace(start=omega[0][0], stop=omega[0][1], num=results.shape[0])
        plt.scatter(x, results)
        plt.plot(x, results)
        plt.title(title)
        if y_range != [0,0]:
            ax = plt.gca()
            ax.set_ylim(y_range)
    plt.show()


""" Classes """

class HeatTransfer(ABC):
    def __init__(self,dim: int, goal: str, initial: str, boundary: str,
                 h_const: float, k_const: float, omega: str, time_range: float) -> None:
        self.dim = dim
        self.f = goal
        self.v = initial
        self.g = boundary
        self.h = h_const
        self.k = k_const
        self.omega = omega
        self.time = time_range
    
    def get_lambda(self) -> float:
        return self.k / (self.h**2)
    
    def initialize_equations(self) -> None:
        self.f = convert_equation(self.f)
        self.v = convert_equation(self.v)
        self.g = convert_equation(self.g)

    def initialize_omega(self) -> None:
        self.omega = convert_omega(self.omega, self.dim)

    @abstractmethod
    def initialize_matrices(self):
        pass

    @abstractmethod
    def calculate_solution(self):
        pass

    @abstractmethod
    def calculate_initial(self):
        pass
