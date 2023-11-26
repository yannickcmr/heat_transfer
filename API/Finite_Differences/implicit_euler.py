import numpy as np
import io
import sys
import copy
import math

# importing model.
sys.path.append("Finite_Differences")
from model import HeatTransfer, calculate_inner_omega, plot_result


""" Helper Functions """

def save_matrix(matrix):
    SAVE_PATH = "Differential_Equations/API/Finite_Differences/test.txt"
    with io.open(SAVE_PATH, "w+", encoding="utf-8") as f:
        for row in matrix:
            f.write(f"{str(row)}\n")


""" Class """

class ImplicitEuler(HeatTransfer):
    def __init__(self, dim: int, goal: str, initial: str, boundary: str, h_const: float, k_const: float, omega: str, time_range: float) -> None:
        super().__init__(dim, goal, initial, boundary, h_const, k_const, omega, time_range)
        self.B = None
        self.inv_B = None

    def initialize_matrices(self):
        if self.dim == 1:
            l_const = self.get_lambda()
            space = np.linspace(start=self.omega[0][0], stop=self.omega[0][1], num=int((self.omega[0][1] - self.omega[0][0])/self.h)-2)
            matrix = np.zeros((len(space), len(space)))

            # adding main diagonal values.
            np.fill_diagonal(matrix, 1+2*l_const)

            # adding offset diagonal values.
            offset_matrix = np.array([-l_const for _ in range(0, matrix.shape[0] - 1)])
            matrix +=np.diag(offset_matrix, k=1)
            matrix +=np.diag(offset_matrix, k=-1)
            self.B = np.around(matrix, decimals=6)

            # calculating the inverse.
            if np.linalg.det(matrix) != 0:
                inv_B = np.linalg.inv(matrix)
                self.inv_B = np.around(inv_B, decimals=6)

        else:
            l_const = self.get_lambda()
            x_space = np.linspace(start=self.omega[0][0], stop=self.omega[0][1], num=int((self.omega[0][1] - self.omega[0][0]) / self.h))
            y_space = np.linspace(start=self.omega[1][0], stop=self.omega[1][1], num=int((self.omega[1][1] - self.omega[1][0]) / self.h))

            space = np.zeros((len(x_space)-2, len(y_space)-2))
            n, m = space.shape
            matrix = np.eye(m, m)

            # defining differential matrix (two center diagonals).
            differential_matrix = np.eye(n, n)
            np.fill_diagonal(differential_matrix, 1+4*l_const)

            # adding offset 1's.
            offset_main = np.array([-l_const for _ in range(0, differential_matrix.shape[0] -1)])
            differential_matrix += np.diag(offset_main, k=1)
            differential_matrix += np.diag(offset_main, k=-1)

            # creating kronecker product.
            matrix = np.kron(matrix, differential_matrix)
            
            # adding secondary diagonals
            offset_secondary = np.array([-l_const for _ in range(0, matrix.shape[0] - space.shape[0])])
            matrix += np.diag(offset_secondary, k=space.shape[0])
            matrix += np.diag(offset_secondary, k=-space.shape[0])
            self.B = np.around(matrix, decimals=6)

            # inverting matrix if possible.
            if np.linalg.det(matrix) != 0:
                inv_B = np.linalg.inv(matrix)
                self.inv_B = np.around(inv_B, decimals=6)

    def calculate_initial_1d(self, points: list):
        # defining range and vector.
        x_range = int((self.omega[0][1] - self.omega[0][0]) / self.h) - 2
        values = np.zeros((x_range))

        # calculating initial values.
        for point in points:
            x = self.omega[0][0] + (point[0] + 1)*self.h
            values[point] = eval(self.v, {"x": x, "math": math})
            values[point] += self.k * eval(self.f, {"x": x, "t": 0, "math": math})
        return values

    def calculate_initial_nd(self, points: list):
        x_range = int((self.omega[0][1] - self.omega[0][0]) / self.h) - 2
        y_range = int((self.omega[1][1] - self.omega[1][0]) / self.h) - 2
        values = np.zeros((x_range, y_range))

        # calculating the initial values.
        for point in points:
            x = self.omega[0][0] + (point[0]+1)*self.h
            y = self.omega[1][0] + (point[1]+1)*self.h
            values[point] = eval(self.v, {"x": x, "y": y, "math": math})
            values[point] += self.k * eval(self.f, {"x": x, "t": 0, "y": y, "math": math})
        return values

    def calculate_initial(self, plot: bool = False):
        self.initialize_matrices()
        points = calculate_inner_omega(self.omega, self.h, self.dim)

        if self.dim == 1:
            values = self.calculate_initial_1d(points)
        else:
            values = self.calculate_initial_nd(points) 

        # plotting the results.
        if plot:
            title = f"Initial Values ({self.dim}d)"
            plot_result(values, self.omega, self.dim, title)
        return values

    def calculate_solution_1d(self, initial_values: np.ndarray, time_steps: np.ndarray):
        # defining points.
        points = np.linspace(start=self.omega[0][0]+self.h,
                             stop=self.omega[0][1]-self.h,
                             num=int((self.omega[0][1] - self.omega[0][0])/self.h)-2)

        # calculating next steps.
        results = [(0, initial_values)]
        for t in time_steps:
            cache_values = copy.deepcopy(results[-1][1])

            # calculating next step.
            new_values = self.inv_B.dot(cache_values)
            f_values = np.array([self.k * eval(self.f, {"x": point, "t": t, "math": math}) for point in points])
            new_values += f_values

            results.append((t, new_values))
        
        # creating result matrix.
        results = [x[1] for x in results]
        results = np.array(results)
        return results

    def calculate_solution_nd(self, initial_values: np.ndarray, time_steps: np.ndarray):
        # defining points.
        x_points = np.linspace(start=self.omega[0][0]+self.h,
                               stop=self.omega[0][1]-self.h,
                               num=int((self.omega[0][1] - self.omega[0][0]) / self.h)-2)
        y_points = np.linspace(start=self.omega[1][0]+self.h,
                               stop=self.omega[1][1]-self.h,
                               num=int((self.omega[1][1] - self.omega[1][0]) / self.h)-2)
        original_shape = (len(x_points), len(y_points))

        # creating tuples of points.
        x_points, y_points = np.meshgrid(y_points, x_points)
        x_points = np.matrix.flatten(x_points)
        y_points = np.matrix.flatten(y_points)
        points = list(zip(x_points, y_points))

        # calculating next steps.
        results = [(0, initial_values)]
        for t in time_steps:
            cache_values = copy.deepcopy(results[-1][1])
            cache_values = np.matrix.flatten(cache_values)

            # calculating next step.
            new_values = self.inv_B.dot(cache_values)
            f_values = np.matrix.transpose(np.array([self.k * eval(self.f, {"x": x, "y": y, "t": t+self.k, "math": math}) for (x,y) in points]))
            new_values += f_values
            #print(f"max f: {np.around(np.max(f_values), decimals=4)}\tmax v: {np.around(np.max(new_values), decimals=4)}")

            new_values = np.reshape(new_values, original_shape)
            results.append((t, new_values))
        
        # creating result matrix.
        results = [x[1] for x in results]
        results = np.array(results)
        return results

    def calculate_solution(self, time_end: float, plot: bool = False) -> np.ndarray:
        # defining time steps.
        time_steps = np.around(time_end / self.k)
        time_steps = np.linspace(self.k, time_end, int(time_steps))

        # calculating the initial values v.
        initial_values = self.calculate_initial()

        if self.dim == 1:
            results = self.calculate_solution_1d(initial_values, time_steps)
        else:
            results = self.calculate_solution_nd(initial_values, time_steps)

        # padding boundary
        results = np.pad(results, pad_width=1, mode="constant", constant_values=0)
        results = results[1:-1]

        # plotting results.
        if plot:
            y_range = [np.min(results)-1, np.max(results)+1]
            for i in range(0, results.shape[0]):
                title = f"Equation at time: {np.around(i*self.k, decimals=5)} ({self.dim}d)"
                plot_result(results[i], self.omega,  self.dim, y_range, title)
        return results


""" Testing """

if __name__ == "__main__":
    print("==> Running Model")

    # defining the functions.
    f_equation = "0"
    v_equation = "sin(2*pi*x*y)"
    g_equation = "0"

    # defining the space.
    dim = 2
    omega = {
        "x": {"start": 0, "end": 1},
        "y": {"start": 0, "end": 1}
    }
    time_range = 0.01
    k, h = 0.001, 0.1

    # defining test point.
    point = {
        "t": 0.5,
        "x": 0.1,
        "y": 0
    }

    # creating the model.
    model = ImplicitEuler(dim, f_equation, v_equation, g_equation, h, k, omega, time_range)
    model.initialize_equations()
    model.initialize_omega()
    model.initialize_matrices()
    #results = model.calculate_initial(True)
    model.calculate_solution(time_range, True)
