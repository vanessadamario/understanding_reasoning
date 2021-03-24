import dash 
from dash.dependencies import Output, Input
import dash_core_components as dcc 
import dash_html_components as html 
import plotly
import dash_table
import random 
import plotly.graph_objs as go 
import sys
import logging
from datetime import datetime
import pandas as pd
import json
from pathlib import Path

global progress_file

heading_style = {
    'padding' : '20px' ,
    'backgroundColor' : '#8B0000',
    'borderTop': '20px solid #3333cc',
    'borderBottom': '20px solid #3333cc',
    'color': "#ffffff",
    'font-size': '200%',
    'font-family': 'Calibri',
    'text-decoration':'underline overline dotted orange'
}

app = dash.Dash(__name__) 

app.layout = html.Div( 
    [ 
        dcc.Interval( 
            id = 'graph-update', 
            interval = 3000, 
            n_intervals = 0
        ),

        html.Div([
            html.H1("Live VQA Progress"),
            ], style = heading_style),

        html.H2(id='experiment_name'),

        html.H3(id='current_time'),

        html.H3(id='last_updated'),

        html.H3('Progress Table', style={"padding": '10px', 'textAlign':'center'}),

        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in ['Tag', 'Status', 'Progress', 'BestValAcc', 'Elapsed Time']],
            style_table={
                'width': '25%',
                # 'padding-left': '18%',
                'textAlign': 'center',
                'marginLeft': 'auto',
                'marginRight': 'auto'
            },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(248, 248, 248)'
                }
            ],
            style_header={
                'backgroundColor': 'rgb(230, 230, 230)',
                'fontWeight': 'bold'
            },
            style_cell={
                'textAlign': 'center',
                'minWidth': '100px', 
                'width': '200px', 
                'maxWidth': '380px',
            }
        )
    ]
)

@app.callback(
    [
        Output('table', 'data'),
        Output('experiment_name', 'children'),
        Output('current_time', 'children'),
        Output('last_updated', 'children'),
    ],
    [
        Input('graph-update', 'n_intervals'), 
    ]
)
def update_table(n):
    global progress_file

    with open(progress_file, 'r') as f:
        d = json.load(f)

    en = 'Case in progress: ' + d['current_case']
    ct = 'Current Time: ' + str(datetime.now().strftime("%Y.%m.%d - %H:%M:%S%p"))
    lu = 'Progress last updated at: ' + d['last_updated']

    df = pd.DataFrame(d['info'])

    return df.to_dict('records'), en, ct, lu


if __name__ == '__main__': 

    global progress_file
    progress_file = sys.argv[1] + '/progress.json'
    fp = Path(progress_file)
    while not fp.exists():
        time.sleep(1)

    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    dash_port = 42125
    app.run_server(port = dash_port)