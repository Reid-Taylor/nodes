import dash
from dash import html, dcc, callback_context
from textwrap import dedent as d 
import nodes

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Markovian Boltzmann Machine"

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
    html.Div([html.H1("Transaction Graph Network")],
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
                                value=G.age,
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
            )]
            ),
        ]
    )
])

###################################callback for side components
@app.callback(
    dash.dependencies.Output('my-graph', 'figure'),
    [dash.dependencies.Input('my-range-slider', 'value'), dash.dependencies.Input('input1', 'value'),
     dash.dependencies.Input('epoch-slider', 'value'),
     dash.dependencies.Input('decrease-val', 'n_clicks'), dash.dependencies.Input('increase-val', 'n_clicks')
    ]
)
def update_output(value,input1, input2, but1, but2):
    global G
    global graph
    if callback_context.triggered_id in ['my-range-slider', 'input1', 'input2']:
        if callback_context.triggered_id == 'my-range-slider':
            graph = G.web_app()
        elif callback_context.triggered_id == 'input1':
            graph = G.web_app()
        else:
            pass
    elif callback_context.triggered_id in ['epoch-slider','increase-val','decrease-val']:
        if callback_context.triggered_id == 'epoch-slider':
            G.forward(input2-G.age)
            G.age = input2
        elif callback_context.triggered_id == 'increase-val':
            G.forward(1)
            graph = G.web_app()
            G.age += 1
        elif callback_context.triggered_id == 'decrease-val':
            G.age -= 1

    return graph
################################callback for right side components
# @app.callback(
#     dash.dependencies.Output('hover-data', 'children'),
#     [dash.dependencies.Input('my-graph', 'hoverData')])
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)

if __name__ == '__main__':
    app.run_server(debug=True)