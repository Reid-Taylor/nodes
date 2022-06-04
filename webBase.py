import dash
from dash import dcc, callback_context
import dash_daq as daq
from dash import html
import networkx as nx 
import plotly.graph_objs as go

import numpy as np
import pandas as pd
from colour import Color
from datetime import datetime 
from textwrap import dedent as d 
import json

import nodes

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Graph Theory Network"

EPOCHS=50
NODES=12

G = nodes.Network(size=NODES)
graph = G.web_app()

# styles: for right side hover/click component
styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

app.layout = html.Div([
    # Title
    html.Div([html.H1("Transaction Network Graph")],
             className="row",
             style={'textAlign': "center"}),

    html.Div(
        className="row",
        children=[
            ############################################ Left side two input components
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d("""
                            **Dropout Rate**

                            Dropout rate for set of possible permutations (edges).
                            """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Slider(
                                0,1,
                                marks={
                                    0: {'label' : '0'},
                                    1: {'label' : '1'}
                                },
                                value=.65,
                                id='my-range-slider'
                            ),
                            html.Br(),
                            html.Div(id='output-container-range-slider')
                        ],
                        style={'height': '100px'}
                    ),
            html.Div(
                className="twelve columns",
                children=[
                    dcc.Markdown(d("""
                    **Size of Network**

                    Select the number of nodes in the network.
                    """)),
                    dcc.Slider(id='input1', value=10, min=0, max=30, step=1, marks={
                        0: {'label' : '0'},
                        30: {'label' : '30'},
                    }),
                    html.Br(),
                    html.Div(id="output")
                ],
                style={'height': '300px'}
            )
                ]
            ),

            ############################################ Middle graph component
            html.Div(
                className="eight columns",
                children=[dcc.Graph(id="my-graph",
                                    figure=graph)]
            ),

            ############################################ Right side two input components
            html.Div(
                className="two columns",
                children=[
                    dcc.Markdown(d(f"""
                            **Age of Network: {G.age}**

                            Select the number of cycles progressed from original layout.
                            """)),
                    html.Div(
                        className="twelve columns",
                        children=[
                            dcc.Slider(
                                0,100,
                                marks={
                                    0: {'label' : '0'},
                                    100: {'label' : '100'}
                                },
                                value=0,
                                step=1,
                                id='epoch-slider'
                            ),
                            html.Button('Prev', id='decrease-val', style={'width':'50%'}),
                            html.Button('Next', id='increase-val', style={'width':'50%'}),
                            html.Br(),
                            html.Div(id='epoch-container-range-slider'),
                        ],
                        style={'height': '100px'}
                    ),
            html.Div(
                className="twelve columns",
                children=[
                    dcc.Markdown(d("""
                    **Signal Amplification**

                    Odd Hyper-parameter.
                    """)),
                    dcc.Slider(id='input2', value=1, min=0, max=2, step=0.05, marks={
                        0: {'label' : '0'},
                        2: {'label' : '2'},
                    }),
                    html.Br(),
                    html.Div(id="output2")
                ],
                style={'height': '200px'}
            ),
            html.Div(
                className="twelve columns",
                children=[#TODO Add in download network info, then load from csv abilities
                    dcc.Markdown(d("""
                    **Activation Based Coloring**
                    """)),
                    daq.BooleanSwitch(id='my-boolean-switch', on=False),
                    html.Br(),
                    html.Div(id="button-node-activation-coloring")
                ],
                style={'height': '300px'}
            )
                ]
            ),
        ]
    )
])

###################################callback for side components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value'),
     dash.dependencies.Input('epoch-slider', 'value'), dash.dependencies.Input('my-boolean-switch','on'),
     dash.dependencies.Input('decrease-val', 'n_clicks'), dash.dependencies.Input('increase-val', 'n_clicks')
    ]
)
def update_output(value,input1, input2, input3, but1, but2):
    global G
    global graph
    if callback_context.triggered_id in ['my-range-slider', 'input1', 'input2']:
        if callback_context.triggered_id == 'my-range-slider':
            graph = G.web_app()
        elif callback_context.triggered_id == 'input1':
            graph = G.web_app()
        else:
            pass
    elif callback_context.triggered_id in ['epoch-slider', 'increase-val','decrease-val']:
        if callback_context.triggered_id == 'epoch-slider':
            pass
        elif callback_context.triggered_id == 'increase-val':
            G.forward(1)
            graph = G.web_app()
            print()
        else:
            pass
    
    YEAR = value
    ACCOUNT = input1
    print(f"Dropout: \t{value}\nNetwork Size:\t{input1}\nEpochs: \t{input2}\nColored: \t{input3}\n")

    return graph
################################callback for right side components
# @app.callback(
#     dash.dependencies.Output('hover-data', 'children'),
#     [dash.dependencies.Input('my-graph', 'hoverData')])
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)