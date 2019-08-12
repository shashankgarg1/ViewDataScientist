from datetime import datetime as dt
import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc

import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import DataLoader

dataLoader = DataLoader.DataLoader("./Sensor_Weather_Data_Challenge.csv")
df = dataLoader.getDf()
scalar = MinMaxScaler()

clusterDf = df.iloc[:, 0: 14].copy()
clusterDf["maxValue"] = clusterDf.iloc[:, 0:13].max(axis=1)
clusterDf.drop(columns=["d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8", "d9", "d10", "d11", "d12", "d13"],
               inplace=True)
x_scaled = scalar.fit_transform(clusterDf)
clusterDf = pd.DataFrame(data=x_scaled, index=clusterDf.index)
cDf = clusterDf.sample(frac=1)
nObs = len(cDf)
splitNo = round(0.7 * nObs)
cDf = cDf.head(splitNo)

gmm = GaussianMixture(n_components=5)
gmm.fit(cDf.values)

km = KMeans(
        n_clusters=5,
        init='random',
        n_init=30,
        max_iter=300,
        random_state=0)
km.fit(cDf.values)


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div([
    html.Title("View, Inc. Dashboard"),
    html.H1("View, Inc."),
    html.Div([
        html.H3("Statistics table"),
        html.Br(),
        "Select date range",
        html.Br(),
        dcc.DatePickerRange(
            id='my-date-picker-range',
            min_date_allowed=df.index.min().date(),
            max_date_allowed=df.index.max().date(),
            initial_visible_month=df.index.min().date(),
            start_date=str(df.index.min().date()),
            end_date=str(df.index.min().date()),
            minimum_nights=0
        ),
        html.Div(id='output-container-date-picker-single'),
    ]),

    html.Hr(),

    html.Div([
        html.H3("Comparison graph"),
        html.Br(),
        "Select date range",
        html.Br(),
        dcc.DatePickerRange(
            id='graph-date-picker-range',
            min_date_allowed=df.index.min().date(),
            max_date_allowed=df.index.max().date(),
            initial_visible_month=df.index.min().date(),
            start_date=str(df.index.min().date()),
            end_date=str(df.index.min().date()),
            minimum_nights=0
        ),
        html.Br(),
        dcc.Dropdown(
            id='xaxis-column',
            options=[{'label': i, 'value': i} for i in df.columns.values],
            value='d1'
        ),

        dcc.Dropdown(
            id='yaxis-column',
            options=[{'label': i, 'value': i} for i in df.columns.values],
            value='d1'
        ),

        # Graphs
        html.Div([
            html.Div(
                id='update_graph_1'
            ),
            html.Div([
                dcc.Graph(id='ga-category'),
            ], className=" twelve columns"
            ), ], className="row "
        )
    ]),

    html.Hr(),

    html.Div([
        html.H3("Trend"),
        html.Br(),
        "Select date range",
        html.Br(),
        dcc.DatePickerRange(
            id='trend-graph-date-picker-range',
            min_date_allowed=df.index.min().date(),
            max_date_allowed=df.index.max().date(),
            initial_visible_month=df.index.min().date(),
            start_date=str(df.index.min().date()),
            end_date=str(df.index.min().date()),
            minimum_nights=0
        ),
        html.Br(),
        dcc.Dropdown(
            id='trend-graph-xaxis-column',
            options=[{'label': i, 'value': i} for i in df.columns.values],
            value='d1'
        ),

        # Graphs
        html.Div([
            html.Div(
                id='trend-graph-update_graph_1'
            ),
            html.Div([
                dcc.Graph(id='trend-graph-category'),
            ], className=" twelve columns"
            ), ], className="row "
        )
    ]),

    html.Hr(),

    html.Div([
        html.H3("Clustering"),
        "The model has been trained before. Select dates",
        html.Br(),
        dcc.DatePickerRange(
            id='clustering-graph-date-picker-range',
            min_date_allowed=df.index.min().date(),
            max_date_allowed=df.index.max().date(),
            initial_visible_month=df.index.min().date(),
            start_date=str(df.index.min().date()),
            end_date=str(df.index.min().date()),
            minimum_nights=0
        ),
        html.Br(),
        "K-Means clustering",
        dcc.Graph(id="k-means-clustering-graph-category"),
        html.Br(),
        "GaussianMixture Clustering",
        dcc.Graph(id="gmm-clustering-graph-category"),
    ])
])


def normalize(series):
    arr = series.values.reshape(-1, 1)
    return pd.np.squeeze(scalar.fit_transform(arr))
    # normalize_series = (series - series.mean()) /  series.std()
    # return normalize_series

def generate_table(dataframe):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +
        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(len(dataframe))]
    )

@app.callback(
    Output('output-container-date-picker-single', 'children'),
    [Input('my-date-picker-range', 'start_date'),
     Input('my-date-picker-range', 'end_date')])
def showStatistics(start_date, end_date):
    if start_date is not None and end_date is not None:
        startDate = dt.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        endDate = dt.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        endDate = endDate.replace(hour=23, minute=59, second=59)
        dateDf = df.loc[startDate: endDate]
        newDf = pd.DataFrame(columns=dateDf["d1"].describe().index)
        obs = "There are a total of " + str(len(dateDf)) + " observations recorded between " + start_date + " and " + end_date
        for column in dateDf.columns.values:
            newDf = newDf.append(dateDf[column].describe())
        newDf.insert(2, "median", dateDf.median())
        newDf.insert(0, "colName", newDf.index)
        newDf.drop(labels="count", axis=1, inplace=True)
        newDf = newDf.round(decimals=2)
        return [obs, html.Br(), generate_table(newDf)]



@app.callback(
    Output("ga-category", "figure"),
    [Input("graph-date-picker-range", "start_date"),
     Input("graph-date-picker-range", "end_date"),
     Input("xaxis-column", "value"),
     Input("yaxis-column", "value")]
)
def plotLineGraph(start_date, end_date, xaxis_column, yaxis_column):
    if start_date is not None and end_date is not None:
        startDate = dt.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        endDate = dt.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        endDate = endDate.replace(hour=23, minute=59, second=59)
        if (startDate > endDate):
            return {"data":[]}
        dateDf = df.loc[startDate: endDate]
        return {
            'data': [
                go.Scatter(
                    x=pd.Series(dateDf.index),
                    y=normalize(dateDf[xaxis_column]),
                    mode='lines+markers',
                    marker={
                        'size': 5,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=xaxis_column
                ),

                go.Scatter(
                    x=pd.Series(dateDf.index),
                    y=normalize(dateDf[yaxis_column]),
                    mode='lines+markers',
                    marker={
                        'size': 5,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'black'}
                    },
                    name=yaxis_column
                ),
            ],
            'layout': go.Layout(
                xaxis={
                    'title': "Time"
                },
                yaxis={
                    'title': "Scaled values"
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }

@app.callback(
    Output("trend-graph-category", "figure"),
    [Input("trend-graph-date-picker-range", "start_date"),
     Input("trend-graph-date-picker-range", "end_date"),
     Input("trend-graph-xaxis-column", "value")]
)
def plotLineGraph(start_date, end_date, xaxis_column):
    if start_date is not None and end_date is not None:
        startDate = dt.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        endDate = dt.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        endDate = endDate.replace(hour=23, minute=59, second=59)
        if (startDate > endDate):
            return {"data":[]}
        dateDf = df.loc[startDate: endDate]
        return {
            'data': [
                go.Scatter(
                    x=pd.Series(dateDf.index),
                    y=dateDf[xaxis_column],
                    mode='lines+markers',
                    marker={
                        'size': 5,
                        'opacity': 0.5,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=xaxis_column
                )
            ],
            'layout': go.Layout(
                xaxis={
                    'title': "Time"
                },
                yaxis={
                    'title': xaxis_column
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }


@app.callback(
    Output("k-means-clustering-graph-category", "figure"),
    [Input("clustering-graph-date-picker-range", "start_date"),
     Input("clustering-graph-date-picker-range", "end_date")]
)
def plotKClusters(start_date, end_date):
    if start_date is not None and end_date is not None:
        startDate = dt.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        endDate = dt.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        endDate = endDate.replace(hour=23, minute=59, second=59)
        if (startDate > endDate):
            return {"data": []}
        cDf = clusterDf.loc[startDate: endDate].copy()
        labels = km.predict(cDf.values)
        return {
            'data': [
                go.Scatter(
                    x=cDf.values[:, 1],
                    y=cDf.values[:, 0],
                    mode='markers',
                    marker={
                        'size': 5,
                        'opacity': 0.5,
                        'color': labels
                    }
                )
            ],
            'layout': go.Layout(
                xaxis={
                    'title': "Sensor Illumination"
                },
                yaxis={
                    'title': "Radiation"
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }


@app.callback(
    Output("gmm-clustering-graph-category", "figure"),
    [Input("clustering-graph-date-picker-range", "start_date"),
     Input("clustering-graph-date-picker-range", "end_date")]
)
def plotKClusters(start_date, end_date):
    if start_date is not None and end_date is not None:
        startDate = dt.strptime(start_date.split(' ')[0], '%Y-%m-%d')
        endDate = dt.strptime(end_date.split(' ')[0], '%Y-%m-%d')
        endDate = endDate.replace(hour=23, minute=59, second=59)
        if (startDate > endDate):
            return {"data": []}
        cDf = clusterDf.loc[startDate: endDate].copy()
        labels = gmm.predict(cDf.values)
        return {
            'data': [
                go.Scatter(
                    x=cDf.values[:, 1],
                    y=cDf.values[:, 0],
                    mode='markers',
                    marker={
                        'size': 5,
                        'opacity': 0.5,
                        'color': labels
                    }
                )
            ],
            'layout': go.Layout(
                xaxis={
                    'title': "Sensor Illumination"
                },
                yaxis={
                    'title': "Radiation"
                },
                margin={'l': 40, 'b': 30, 't': 10, 'r': 0},
                height=450,
                hovermode='closest'
            )
        }

if __name__ == '__main__':
    app.run_server(debug=True)
