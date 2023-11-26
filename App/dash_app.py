import numpy as np
import pandas as pd
import requests
import json
from flask import Flask
import dash
from dash import html, dcc, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objs as go


""" Constants """

complete_functions = {
    "Functions": {
        "1": "Sine (sin)",
        "2": "Cosine (cos)",
        "3": "Tangent (tan)",
        "4": "Squart Root",
        "5": "Power (x^y)",
        "6": "Logarithm (log)",
        "7": "Logarithm 2 (log_2)",
        "8": "Logarithm 10 (log_10)",
        "9": "Exponential (e)",
        "10": "Exponential with 2 (2^x)",
        "11": "Pi (3.141592...)"
    },
    "Input": {
        "1": "sin(...)",
        "2": "cos(...)",
        "3": "tan(...)",
        "4": "sqrt(...)",
        "5": "pow(x, y)",
        "6": "log(...)",
        "7": "log_2(...)",
        "8": "log_10 (...)",
        "9": "exp(...)",
        "10": "exp_2(...)",
        "11": "pi"
    },
    "Model": {
        "1": "math.sin(...)",
        "2": "math.cos(...)",
        "3": "math.tan(...)",
        "4": "math.sqrt(...)",
        "5": "math.pow(x,y)",
        "6": "math.log(...)",
        "7": "math.log_2(...)",
        "8": "math.log_10 (...)",
        "9": "math.exp(...)",
        "10": "math.exp_2(...)",
        "11": "math.pi"
    }
}
df_complete_functions = pd.DataFrame.from_dict(complete_functions)


""" Settings """

#SOLVER_URL = "http://localhost:8001/" # needed when running locally.
SOLVER_URL = "http://Solver_FastAPI:8001/"
TIMEOUT = 60


""" App """

server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div(id="main-container", children=[
    dcc.Store(id="solution-store"),
    dcc.Interval(id="animate", disabled=True, interval=100),
    html.Div(id="modal-container", children=[
        dbc.Modal(id="error-message-1", is_open=False),
        dbc.Modal(id="error-message-2", is_open=False),
        dbc.Modal(id="error-message-3", is_open=False)
    ]),
    html.Br(),
    html.Div(id="greetings-container", children=[
        html.Div(className="col-3"),
        html.Div(children=[
            html.H2("Finite Differences Solver", style={"text-align": "center"})
        ], className="col-6"),
        html.Div(children=[
            html.Abbr("How To",
                      title="When you want to approximate a PDE, scroll down to the How To Guide.\n\
To see all the available functions to choose from, please go to the bottom of the page.",
                      style={"text-align": "right"})
        ], className="col-2", style={"text-align": "right"}),
        html.Div(className="col-1")
    ] , className="row"),
    html.Br(),
    html.Br(),
    html.Div(id="solver-main-container", children=[
        dbc.Col(width=1),
        dbc.Col(id="graph-column", children=[
            dbc.Card(id="graph-card-container", children=[
                dbc.CardHeader(children=[
                    html.H3("Graph")
                ], style={"text-align": "center"}),
                dbc.CardBody(children=[
                    dcc.Loading(children=[
                        dbc.Row(id="graph-row", children=[
                            dcc.Graph(id="solution-graph", figure=go.Figure(go.Scatter(x=[0,1,2], y=[0,0,0])))
                        ])
                    ]),
                    html.Br(),
                    dbc.Row(id="time-slider-row", children=[
                        dbc.Col(width=3),
                        dbc.Col(id="time-slider-column", children=[
                            dcc.Slider(id="time-graph-input",
                                        min=0.00, max=10,
                                        step=0.001,
                                        value=0,
                                        marks=None,
                                        tooltip={
                                            "placement": "bottom",
                                            "always_visible": False
                                        }
                            )
                        ], width=6),
                        dbc.Col(width=1),
                        dbc.Col(children=[
                            dbc.Button("play", id="toggle-animation")
                        ], width=2)
                    ]),
                    html.Br(),
                    dbc.Row(id="results-row")
                ])
            ])
        ], width=8),
        dbc.Col(id="options-column", children=[
            dbc.Row(children=[
                dbc.Col(width=1),
                dbc.Col(children=[
                    dbc.Card(id="options-card-container", children=[
                        dbc.CardHeader(children=[
                            html.H3("Options")
                        ], style={"text-align": "center"}),
                        dbc.CardBody(children=[
                            html.H5("Space", style={"text-align": "center"}),
                            dbc.Row(id="dimension-container", children=[
                                dbc.Label(children=[
                                    html.H6("Dimension:", style={"text-align": "right"})
                                ], width=5),
                                dbc.Col(children=[
                                    dcc.Dropdown(id="dimension-input",
                                                options=["1", "2"],
                                                 value="2",
                                                 clearable=False,
                                                 style={'zIndex': '3'})
                                ], width=7)
                            ]),
                            dbc.Row(id="x-omega-container", children=[
                                dbc.Col(width=2),
                                dbc.Col(children=[
                                    dbc.Input(id="x-start-input",
                                              value=0,
                                              type="number",
                                              placeholder="x0")
                                ], width=3),
                                dbc.Label(children=[
                                    html.H6("< x < ", style={"text-align": "center"})
                                ], width=3),
                                dbc.Col(children=[
                                    dbc.Input(id="x-end-input",
                                              value=2,
                                              type="number",
                                              placeholder="x1")
                                ], width=3)
                            ]),
                            dbc.Collapse(id="y-omega-collapse", children=[
                                dbc.Row(id="y-omega-container", children=[
                                    dbc.Col(width=2),
                                    dbc.Col(children=[
                                        dbc.Input(id="y-start-input",
                                                  value=0,
                                                  type="number",
                                                placeholder="y0")
                                    ], width=3),
                                    dbc.Label(children=[
                                        html.H6("< y < ", style={"text-align": "center"})
                                    ], width=3),
                                    dbc.Col(children=[
                                        dbc.Input(id="y-end-input",
                                                  value=2,
                                                  type="number",
                                                placeholder="y1")
                                    ], width=3)
                                ])
                            ], is_open=False),
                            html.Br(),
                            html.H5("Equations", style={"text-align": "center"}),
                            dbc.Row(id="equation-container", children=[
                                dbc.Label(children=[
                                    html.H6("f:", style={"text-align": "right"})
                                ], width=2),
                                dbc.Col(children=[
                                    dbc.Input(id="equation-input",
                                              value="3*pow(x, 3) - exp(-y)",
                                              placeholder="Enter as LaTeX")
                                ], width=10)
                            ]),
                            dbc.Row(id="initial-condition-container", children=[
                                dbc.Label(children=[
                                    html.H6("v:", style={"text-align": "right"})
                                ], width=2),
                                dbc.Col(children=[
                                    dbc.Input(id="initial-condition-input",
                                              value="0.1*x*y",
                                              placeholder="Enter as LaTeX")
                                ], width=10)
                            ]),
                            dbc.Collapse(children=[
                                dbc.Row(id="border-condition-container", children=[
                                    dbc.Label(children=[
                                        html.H6("g:", style={"text-align": "right"})
                                    ], width=2),
                                    dbc.Col(children=[
                                        dbc.Input(id="boundary-input",
                                                value="0",
                                                placeholder="Enter as LaTeX")
                                    ], width=10)
                                ])
                            ], is_open=False),
                            html.Br(),
                            html.H5("Discretization Constants", style={"text-align": "center"}),
                            dbc.Row(id="constants-container", children=[
                                dbc.Label(children=[
                                    html.H6("k: ", style={"text-align": "right"})
                                ], width=2),
                                dbc.Col(children=[
                                    dbc.Input(id="k-const-input",
                                              value=0.001)
                                ], width=4),
                                dbc.Label(children=[
                                    html.H6("h: ", style={"text-align": "right"})
                                ], width=2),
                                dbc.Col(children=[
                                    dbc.Input(id="h-const-input",
                                              value=0.01)
                                ], width=4),
                            ]),
                            html.Br(),
                             dbc.Row(id="time-container", children=[
                                dbc.Label(children=[
                                    html.H6("Time Range: ", style={"text-align": "right"})
                                ], width=4),
                                dbc.Col(children=[
                                    dcc.Slider(id="time-input",
                                               min=0.01, max=5,
                                               step=0.01,
                                               value=0.3,
                                               marks=None,
                                               tooltip={
                                                   "placement": "top",
                                                   "always_visible": False
                                                }
                                    )
                                ], width=8, style={"padding-top": "10%"})
                            ]),
                            html.Br(),
                            html.H5("Approximation Schemes", style={"text-align": "center"}),
                            dbc.Row(id="solver-container", children=[
                                dbc.Label(children=[
                                    html.H6("Solver:", style={"text-align": "right"})
                                ], width=4),
                                dbc.Col(children=[
                                    dcc.Dropdown(id="solver-input",
                                                 options=["explicit", "implicit", "crank-nicolson", "theta", "exact"],
                                                 value="explicit",
                                                 clearable=False
                                                 )
                                ], width=8)
                            ]),
                            dbc.Collapse(id="theta-collapse-container", children=[
                                dbc.Row(id="theta-input-container", children=[
                                    dbc.Label(children=[
                                        html.H6("Theta:", style={"text-align": "right"})
                                    ], width=4),
                                    dbc.Col(children=[
                                        dbc.Input(id="theta-input",
                                                  value=0.1,
                                                  type="number")
                                    ], width=8)
                                ]),
                            ], is_open=False),
                            html.Br(),
                            dbc.Row(id="submit-container", children=[
                                dbc.Col(width=3),
                                dbc.Col(children=[
                                    dbc.Button("Submit", id="submit-button")
                                ], width=6, style={"text-align": "center"}),
                                dbc.Col(width=3)
                            ]),
                            html.Br(),
                        ])
                    ])
                ], width=10),
                dbc.Col(width=1)
            ])
        ], width=3),
    ], className="row"),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(id="how-to-use-manual-guide-container", children=[
        html.Div(className="col-2"),
        html.Div(children=[
            html.H2("How To Manual", style={"text-align": "center"}),
            html.Hr(),
            html.Div(children=[
                html.Div(children=[
                    html.H2("1.", style={"text-align": "right"})
                ], className="col-2"),
                html.Div(children=[
                    html.P("Start the approximation by choosing your desired dimension. \
Once you selected one, you can edit the space by entering the axis ranges. Note that you can only choose rectangular spaces in this version of the app.")
                ], className="col-10")
            ], className="row"),
            html.Br(),
            html.Hr(),
            html.Div(children=[
                html.Div(children=[
                    html.H2("2.", style={"text-align": "right"})
                ], className="col-2"),
                html.Div(children=[
                    html.P("Now you can enter the functions. If you want to know what kind of functions you can select, \
please have a look at the table down below in order to see how you should enter these functions. \
If you selected the 2-dim case, please keep in mind, that you have to also add the constant 'y' in the equations in order for the approximation to work properly."),
                    html.P("\t -> f(x,t): Describes the right side of the heat transfer equation."),
                    html.P("\t -> v(x): Represents the initial condition."),
                    html.P("\t -> g(x,t): Describes the boundary condition for the equation.")
                ], className="col-10")
            ], className="row"),
            html.Br(),
            html.Hr(),
            html.Div(children=[
                html.Div(children=[
                    html.H2("3.", style={"text-align": "right"})
                ], className="col-2"),
                html.Div(children=[
                    html.P("When you are done entering the equations, you can now set the discretization constants and the time range."),
                    html.P("\t -> k: The discrete time constants (time step per iteration)"),
                    html.P("\t -> h: The spacial discretization for the approximation")
                ], className="col-10")
            ], className="row"),
            html.Br(),
            html.Hr(),
            html.Div(children=[
                html.Div(children=[
                    html.H2("4.", style={"text-align": "right"})
                ], className="col-2"),
                html.Div(children=[
                    html.P("Finally you can select the desired solver for this problem. For each solver you may have to select a different lambda \
value in order for the approximation to run. If you selected a sub-optimal lambda value, some approximations will run poorly since the calculations are done via matrix multiplication.")
                ], className="col-10")
            ], className="row")
        ], className="col-8"),
        html.Div(className="col-2")
    ], className="row"),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Div(id="complete-function-list", children=[
        html.Div(className="col-3"),
        html.Div(children=[
            html.H2("Available Functions and Constants", style={"text-align": "center"}),
            html.Br(),
            dash_table.DataTable(data=df_complete_functions.to_dict("records"),
                                                style_as_list_view=True,
                                                style_cell={
                                                    "text-align": "center"
                                                },
                                                style_header={
                                                    "backgroundColor": "rgb(150, 239, 246)",
                                                    "fontWeight": "bold"
                                                })
        ], className="col-6"),
        html.Div(className="col-3")
    ], className="row"),
    html.Br(),
    html.Br(),
    html.Br()
])


""" Helper Functions """

def plot_result(data: np.ndarray, setting: dict, dim: int, y_lim: list):
    # defining base constants.
    omega = setting["setting"]["omega"]
    f_eq = setting["equation"]["goal"]

    if dim == 1:
        x_data = np.linspace(start=float(omega["x"]["start"]), stop=float(omega["x"]["end"]), num=len(data)+1)
        fig= go.Figure(data=[go.Scatter(x=x_data, y=data)], layout_yaxis_range=y_lim)

    else:
        x_data = np.linspace(start=float(omega["x"]["start"]), stop=float(omega["x"]["end"]), num=data.shape[0]+1)
        y_data = np.linspace(start=float(omega["y"]["start"]), stop=float(omega["y"]["end"]), num=data.shape[1]+1)

        fig = go.Figure(data=[go.Surface(x=x_data, y=y_data, z=data.transpose())])
        fig.update_layout(title=f"Approximation for f(x,t)={f_eq}", autosize=False)
        fig. update_layout(dict1={
                "scene": {
                    "zaxis": {"range": y_lim},
                    "aspectratio": {"x": float(omega["x"]["end"]), "y": float(omega["y"]["end"]), "z": 0.5}
                },
                "scene_camera_eye": {
                    "x": 1.3,
                    "y": 1.6,
                    "z": 0.4
                }
        }, height=550)

    return fig

def create_result_row(data: dict) -> list:
    print(data)
    # functions.
    function_row = dbc.Row(id="functions-results", children=[
        dbc.Col(width=3),
        dbc.Col(f"v(x)={data['equation']['initial']}", width=3, style={"text-align": "center"}),
        dbc.Col(f"g(x,t)={data['equation']['boundary']}", width=3, style={"text-align": "center"}),
        dbc.Col(width=3)
    ])
    # constants.
    constant_row = dbc.Row(id="constant_results", children=[
        dbc.Col(width=3),
        dbc.Col(f"h={data['setting']['h']}", width=2, style={"text-align": "center"}),
        dbc.Col(f"k={data['setting']['k']}", width=2, style={"text-align": "center"}),
        dbc.Col(f"lambda={np.around(float(data['setting']['k']) / (float(data['setting']['h']) ** 2), decimals=5)}", style={"text-align": "center"}, width=2),
        dbc.Col(width=3)
    ])

    return [html.Div(children=[
        html.H5("Input Data", style={"text-align": "center"}),
        function_row,
        html.Br(),
        constant_row
    ])]

def decode_error(error_msg: dict) -> list:
    return [
        dbc.ModalHeader(error_msg["title"], style={"text-align": "center"}),
        dbc.ModalBody(error_msg["message"])
    ]


""" Callback Functions """

@callback(
        output={
            "y-collapse": Output("y-omega-collapse", "is_open")
        },
        inputs={
            "dim": Input("dimension-input", "value")
        }
)
def show_omega(dim: str) -> dict:
    dim = int(dim)
    if dim == 1:
        return {
            "y-collapse": False
        }
    return {
            "y-collapse": True
        }

@callback(
        [Output("time-graph-input", "step"),
         Output("time-graph-input", "max")],
        Input("time-input", "value")
)
def set_graph_slider(time_value: float):
    return [0.01, time_value]

@callback(
        output={
            "store": Output("solution-store", "data"), 
            "animate": Output("animate", "max_intervals"),
            "modal": {
                "message": Output("error-message-1", "children"),
                "open": Output("error-message-1", "is_open")
            }
        },
        inputs={
            "button": Input("submit-button", "n_clicks"),
            "equations": {
                "goal": State("equation-input", "value"),
                "initial": State("initial-condition-input", "value"),
                "boundary": State("boundary-input", "value")
                },
            "settings": {
                "omega": {
                    "x": {
                        "start": State("x-start-input", "value"),
                        "end": State("x-end-input", "value")
                    },
                    "y": {
                        "start": State("y-start-input", "value"),
                        "end": State("y-end-input", "value")
                    }
                },
                "time": State("time-input", "value"),
                "h": State("h-const-input", "value"),
                "k": State("k-const-input", "value"),
                "dim": State("dimension-input", "value")
            },
            "solver": {
                "solver": State("solver-input", "value"),
                "theta": State("theta-input", "value")
            }
        },
        prevent_initial_call=True
)
def solver_problem(button, equations: dict, settings: dict, solver: dict, log_lvl: str = "debug") -> list:
    print("DASH ==> Solver Input:", button, equations, settings, solver)

    # defining relevant data.
    request_url = SOLVER_URL + "solve"
    request_body = {
        "equation": equations,
        "setting": settings,
        "solver": solver
    }
    requests_parameter = {
        "log_lvl": log_lvl
    }
    print(f"DASH ==> Sending Request: -> URL: {request_url}\n-> body: {request_body}\n-> parameter: {requests_parameter}")

    # sending the request to the fast api.
    try:
        response = requests.post(url=request_url,
                                 data=json.dumps(request_body),
                                 params=requests_parameter,
                                 timeout=TIMEOUT)
        print(f"DASH ==> Response: {response}")
    except Exception as e:
        print(f"DASH ==> Could not get a Response: {e}")
        return {
            "store" : None,
            "animate": 1,
            "modal": {
                "message": decode_error({"title": "Connection Error", "message": str(e)}),
                "open": True
            }
        }

    if response.status_code != 200:
        print(f"DASH ==> Could not solve this equation: {response.status_code}")
        message = f"Could not get data from Solver API. Received Response Code ({response.status_code})"
        return {
                "store" : None,
                "animate": 1,
                "modal": {
                    "message": decode_error({"title": "Connection Error", "message": message}),
                    "open": True
                }
            }
    
    # checking data.
    data = json.loads(response.text)
    print(data["duration"])
    data = json.loads(data["data"])
    data = np.asarray(data)
    print(data.shape)
    print(f"DASH ==> Data Shape: {data.shape}")
    return {
                "store" : {"body": request_body, "data": data},
                "animate": len(data)-1,
                "modal": {
                    "message": decode_error({"title": "", "message": ""}),
                    "open": False
                }
            }

@callback(
        Output("animate", "disabled"),
        Input("toggle-animation", "n_clicks"),
        State("animate", "disabled")
)
def play_animation(button, state):
    print(button, state)
    if button:
        return not state
    return state

@callback(
    [Output("graph-row", "children"),
     Output("results-row", "children"),
     Output("time-graph-input", "value")],
     inputs={
         "slider": Input("time-graph-input", "value"),
         "store": Input("solution-store", "data"),
         "animate": Input("animate", "n_intervals"),
         "k": State("k-const-input", "value")
     }, prevent_initial_call=True
)
def plot_solution(slider, store, animate, k) -> list:
    slider_value = 0
    if store is None:
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3, 4], y=[0,0,0,0]))
        return [[dcc.Graph(id="solution-graph", figure=fig)], [], slider_value]

    result_data = np.asarray(store["data"])
    array = result_data[0]
    for i in range(result_data.shape[0]):
        print(i, np.max(result_data[i]))
    
    # in case time was changed.
    print("checking time slider change.")
    trigger = dash.callback_context.triggered_id
    if trigger == "time-graph-input":
        print("slider changed")
        slider_value = slider
        print(f"taking slice: {int(np.floor(slider / float(k)))}")
        array = result_data[int(np.floor(slider / float(k)))]
    elif trigger == "animate":
        print("animate was pressed")
        slider_value = animate*float(k)
        array = result_data[int(animate)]

    print(f"start plotting for time step: {slider}")
    print(array.shape)
    dim = int(len(array.shape))
    print(np.min(result_data))
    y_lim = [np.min(result_data) - 1, np.max(result_data) + 1]
    fig = plot_result(array, store["body"], dim, y_lim)

    print("creating result row.")
    results = create_result_row(store["body"])

    return [[dcc.Graph(id="solution-graph", figure=fig)], results, slider_value]

@callback(
    Output("theta-collapse-container", "is_open"),
    Input("solver-input", "value")
)
def show_theta(solver: str) -> bool:
    if solver == "theta":
        return True
    return False


""" Testing """

if __name__ == "__main__":
    print("--> Starting App.")
    app.run(debug=True, port=8000, host="127.0.0.1")
