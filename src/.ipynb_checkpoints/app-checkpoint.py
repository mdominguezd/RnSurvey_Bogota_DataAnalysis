#!/usr/bin/env python
# coding: utf-8

# # Jupyter dashboard app for RC visualizations

# In[1]:


# Imports
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import dash
from dash import Dash, dcc, html, callback, callback_context
import dash_bootstrap_components as dbc
from dash_bootstrap_templates import load_figure_template
import dash_daq as daq
from dash.exceptions import PreventUpdate
from dash import dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output, State
from dash_iconify import DashIconify
import geopandas as gpd
from shapely.geometry import Polygon
from os import path
from os import remove
from os import listdir
import re
import warnings
import dash_mantine_components as dmc
import sys
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor as RFR


# ## Pre-processing for data distribution part

# In[3]:


DF_RC = pd.read_csv('Dataset for fitting/Processed_DataFrame.csv')

DF_data = pd.DataFrame(DF_RC['RC'])

DF_data['Exceed WHO'] = DF_data.RC.apply(lambda df :'Above WHO recommended level' if  (df >100) else 'Below WHO recmmended level')
DF_data['Exceed EPA'] = DF_data.RC.apply(lambda df :'Above EPA action level' if  (df >148) else 'Below EPA action level')


# In[4]:


X = DF_RC.iloc[:,2:]
X = sm.add_constant(X)

tab = pd.DataFrame()
tab["Features"] = X.columns[1:]
tab["VIF Factor"] = [round(variance_inflation_factor(X.values, i+1),2) for i in range(X.shape[1]-1)]
VIF_vars = list(tab[tab['VIF Factor'] < 4]['Features'])
tab = tab.sort_values(by = 'VIF Factor', ascending = False)
tab=tab.set_index('Features')


# ## Pre-processing for RC estimations map

# In[5]:


warnings.filterwarnings('ignore')

def read_data(variables, measurements_path = 'Dataset for fitting/Processed_DataFrame.csv'):
    DF_RC = pd.read_csv(measurements_path)
    
    X = DF_RC[variables]
    y = DF_RC['RC']

    X = sm.add_constant(X)
    
    msg = 'Reading data...'
    
    return X, y, msg

def fit_model(X, y, HQ, model = 'Log_Linear'):
    
    variables = list(X.columns)[1:]
    
    var_ID = ''
    for i in variables:
        var_ID += i[0]
    
    if model == 'Log_Linear':
        lin_reg = sm.OLS(np.log(y), X).fit(maxiter=1000)
        mod = lin_reg
        importance = lin_reg.params
        pval = lin_reg.pvalues
        importance = pd.DataFrame(importance[1:])
        importance['Features'] = variables
        importance['p'] = pval
        importance.columns = ['Weight', 'Features','p_value']
        importance = importance[['Features', 'Weight', 'p_value']]
        importance['Weight'] = importance.Weight.apply(lambda df : round(df,2))
        importance['p_value'] = importance.p_value.apply(lambda df : round(df,2))
        RMSE = np.sqrt(np.sum((np.exp(lin_reg.predict(X))-(y))**2)/(len(y)))
        RMSE = pd.DataFrame([np.round(RMSE,2)])
        RMSE.columns = ['RMSE']
        
    elif model == 'Random_Forest': 
        RF_reg = RFR().fit(X, y)
        mod = RF_reg
        importance = RF_reg.feature_importances_
        importance = pd.DataFrame(importance[1:])
        importance['Features'] = variables
        importance.columns = ['Importance', 'Features']
        importance = importance[['Features', 'Importance']]
        importance['Importance'] = importance.Importance.apply(lambda df : round(df,2))
        RMSE = np.sqrt(np.sum((RF_reg.predict(X)-y)**2)/(len(y)))
        RMSE = pd.DataFrame([RMSE])
        RMSE.columns = ['RMSE']
        
    importance = importance.to_dict('records')
    
    RMSE = RMSE.to_dict('records')
    
    name_file = model+var_ID
    
    if HQ:
        exists = path.exists('Regression results/Rn_estimations_'+name_file+'_pol_HQ.geojson')
    else:
        exists = path.exists('Regression results/Rn_estimations_'+name_file+'_pol.geojson')
        
    if exists:
        msg = 'Fitting model...\nEstimating RC values...'
    else:
        msg = 'Fitting model...\nEstimating RC values...\n\nThis will take some time...\n\n'
    
        
    return importance, RMSE, mod, exists, name_file, msg


def EstimatingValues(mod,
                     model,
                     X,
                     exists,
                     HQ,
                     name_file,
                     crs = '3116',
                     dataset_for_estimation_path = "Dataset for regression/Houses_for_Rn_estimation_processed_3116.txt",
                     res = 300,
                     ):
    
    if exists:
        
        if HQ:
            grid = gpd.read_file('Regression results/Rn_estimations_'+name_file+'_pol_HQ.geojson')
        else:
            grid = gpd.read_file('Regression results/Rn_estimations_'+name_file+'_pol.geojson')
    else:
        df_RnModel = pd.read_table(dataset_for_estimation_path, delimiter = ',')
        df_RnModel['const'] = np.ones(len(df_RnModel))

        x_range = df_RnModel['X'].max() - df_RnModel['X'].min()
        y_range = df_RnModel['Y'].max() - df_RnModel['Y'].min()

        df_RnModel['Cluster'] = np.zeros(len(df_RnModel))

        cols = np.arange(df_RnModel['X'].min() + res/2, df_RnModel['X'].max(), res)
        rows = np.arange(df_RnModel['Y'].min() + res/2, df_RnModel['Y'].max(), res)

        k = 0
        polygons = []
        for i in range(len(cols)-1):

            for j in range(len(rows)-1):

                k += 1
                df_RnModel.loc[(df_RnModel.X >= cols[i])&(df_RnModel.X < cols[i+1])&(df_RnModel['Y'] >= rows[j])&(df_RnModel['Y'] < rows[j+1]), 'Cluster'] = k
                polygons.append(Polygon([(cols[i],rows[j]), (cols[i]+res, rows[j]), (cols[i]+res, rows[j]+res), (cols[i], rows[j]+res)]))    

        df_RnModel = df_RnModel.groupby('Cluster').mean()

        df_RnModel_reg = df_RnModel[list(X.columns)]
        
        if model == 'Log_Linear':
            df_RnModel['Reg'] = np.exp(mod.predict(df_RnModel_reg))
        elif model == 'Random_Forest':
            df_RnModel['Reg'] = mod.predict(df_RnModel_reg)
            
        gdf = gpd.GeoDataFrame(df_RnModel['Reg'], geometry=gpd.points_from_xy(df_RnModel.X, df_RnModel.Y))
        gdf = gdf.set_crs('EPSG:'+crs)
        gdf = gdf.to_crs('EPSG:4326')
        grid = gpd.GeoDataFrame({'geometry':polygons})
        grid = grid.set_crs('EPSG:'+crs)
        grid = grid.to_crs('EPSG:4326')
        grid = gpd.sjoin(grid, gdf)
        
        if HQ:
            gdf.to_file('Regression results/Rn_estimations_'+name_file+'_HQ.geojson', driver="GeoJSON")
            grid.to_file('Regression results/Rn_estimations_'+name_file+'_pol_HQ.geojson', driver="GeoJSON")
        else:
            gdf.to_file('Regression results/Rn_estimations_'+name_file+'.geojson', driver="GeoJSON")
            grid.to_file('Regression results/Rn_estimations_'+name_file+'_pol.geojson', driver="GeoJSON")
    
    msg = 'Done :)'
    
    x_c = grid.dissolve().centroid.x.mean()
    y_c = grid.dissolve().centroid.y.mean()

    return grid, x_c, y_c, msg


# In[6]:


def plot_figure(Organization):
    x = np.arange(0,440,25)
    rc = DF_data['RC']
    y = np.histogram(rc, bins = x)

    hist = px.histogram(DF_data, x = 'RC', range_x = [0,425])

    fig=make_subplots(specs=[[{'secondary_y': True}, {"type": "pie"}]],
                      cols = 2)

    fig.update_layout(template = 'morph',
                     );

    fig.add_trace(
        go.Histogram(x=hist.data[0].x,
               y=hist.data[0].y,
               name="Percentage of<br>RC measurements", 
               histnorm = 'percent', marker_color = 'rgb(55,100,200)',
               hoverinfo = 'x+y',
              ), secondary_y=False)
    
    ref_levs = [100,148]
    
    if Organization[-3:] == 'WHO':
        fig.add_vline(ref_levs[0], annotation_text = Organization[-3:] + ' recommended level',
                      annotation_position = 'top',
                      line_dash="dash", row = 1, col =1)
    else:
        fig.add_vline(ref_levs[1], annotation_text = Organization[-3:] + ' recommended level',
                      annotation_position = 'top',
                      line_dash="dash", row = 1, col =1)

    fig.update_traces(xbins=dict( # bins used for histogram
            start=0.0,
            end=425.0,
            size=25
            ))
    
    fig.update_xaxes(range = [0, max(rc) + max(rc)*0.05])

    fig.add_trace(
        go.Scatter(x = (x[1:]),
                   y = np.round(100*np.cumsum(y[0]/30),2),
                   name="Accumulated<br>percentage of<br>RC measurements",
                   line_color="#ee0000", hoverinfo="x+y"), secondary_y=True)

    fig.update_layout(title_text = 'Residential RC measurements distribution', 
                      title_font_family = 'bahnschrift',
                      font_family = 'bahnschrift',
                      title_font_size = 30, xaxis_title_text='Residential RC [Bq/m^3]', # xaxis label
                      yaxis_title_text='Percentage of RC measurements' # yaxis label
                     )

    labels = DF_data.groupby(Organization).count().iloc[:,0].index
    values = DF_data.groupby(Organization).count().iloc[:,0].values

    fig.add_trace(go.Pie(labels = labels,
                         values = values,
                         textinfo = 'percent',
                         hoverinfo = 'label+value', 
                         marker = dict(colors = ['rgb(255,0,0)', 'rgb(55,100,200)']),
                         showlegend = False,
                         title = 'Comparison with ' + Organization[-3:] + ' recommedation',
                         titleposition = 'bottom center',
                         titlefont = dict(size = 20)
                        ),
                  row = 1, col = 2 
                 )


    fig.update_layout(title_font_size = 30)
    
    return fig


# # App 

# In[7]:


app = Dash(__name__, external_stylesheets=[dbc.themes.MORPH])
server = app.server

load_figure_template(["morph"])
color = 'lightsteelblue'

printing = html.P(["printing", html.Br()])

app.layout = dmc.NotificationsProvider(html.Div([
                                                 html.Div([html.Div([],style = {'width':20}), html.H1('Residential RC modeling', style={'font-family' : 'bahnschrift'})],  style = {'display':'flex'}),
                                                 html.Div([''], style = {'height':20, 'background-color':color}),
                                                 html.Div(
                                                   [   html.Div([''], style = {'width':20}),
                                                       html.Div([
                                                           html.H5('Compare measurements with recommendation levels', style={'font-family' : 'bahnschrift'}),
                                                           dcc.Dropdown(list(DF_data.iloc[:,-2:].columns),'Exceed WHO',id='Organization', style={'font-family' : 'bahnschrift','width':440}),
                                                           html.H5('Feature and model selection for RC modeling', style={'font-family' : 'bahnschrift'}),
                                                               html.Div([
                                                                   html.Plaintext('   Feature selection information: ', style={'font-family' : 'bahnschrift', 'width' : 250}),
                                                                   dcc.Dropdown(['Correlation matrix', 'Variance Inflation Factor'],'Correlation matrix', id = 'FS', style={'font-family' : 'bahnschrift', 'width' : 200})
                                                               ], style=dict(display='flex', width = 440)),
                                                               html.Div([
                                                                   html.Plaintext('   Model: ', style={'font-family' : 'bahnschrift', 'width' : 100}),
                                                                   dcc.Dropdown(['Log_Linear', 'Random_Forest'],'Log_Linear', id = 'model', style={'font-family' : 'bahnschrift', 'width' : 340})
                                                               ], style=dict(display='flex', width = 440)),
                                                               html.Div([
                                                                   html.Plaintext('   Features: ', style={'font-family' : 'bahnschrift', 'width' : 100}),
                                                                   dcc.Dropdown(list(DF_RC.iloc[:,2:].columns), VIF_vars, id = 'vars_', style={'font-family' : 'bahnschrift' , 'width' : 340}, multi = True)
                                                               ], style=dict(display='flex')),
                                                               html.Div([],style = {'height': 10}),
                                                               html.Div([
                                                                   html.Plaintext('   High quality model:',  style={'font-family' : 'bahnschrift', 'width' : 170}),
                                                                   html.Div([
                                                                       html.Div([''], style = {'height':15}),
                                                                       daq.BooleanSwitch(id='HQ_model', on=False),
                                                                   ]),
                                                                   ], style=dict(display='flex', width = 440)),
                                                               html.Div([
                                                                   html.Div([
                                                                               html.Plaintext('   Run model: ', style={'font-family' : 'bahnschrift', 'width' : 100}),
                                                                               html.Button('RUN', style={'font-family' : 'bahnschrift','background-color':'steelblue','font-size':'20px', 'border' : '0px',
                                                                                                         'color': 'white','border-radius':'12px','width' : 340, 'height' :50},  id='Predict_Rn', n_clicks=0),
                                                                            ], style=dict(display='flex', width = 465)), 
                                                               ], style=dict(display='flex', width = 450)),
                                                       ], style = {'width' : 450}
                                                       ), 
                                                       html.Div([''], style = {'width':20, 'background-color':color}),
                                                       html.Div([
                                                                    dcc.Graph(id='RC-histogram', style = {'height' : 450, 'width' : 995}),

                                                                    html.Div([''], style = {'width':20, 'background-color':color}),
                                                                    dcc.Graph(id = 'FS_out',  style = {'height' : 450, 'width' : 400})
                                                       ], style=dict(display='flex'))
                                                    ],
                                                        style=dict(display='flex')),

                                                     html.Div([''],style = {'height':20, 'background-color':color}),
                                               html.Div([
                                                   html.Div([''], style = {'width':20}),
                                                   html.Div([
                                                             html.H5(' Regression results: ', style={'font-family' : 'bahnschrift'}),
                                                             dash_table.DataTable(id= 'imp', style_table={'width' : 440}),
                                                             html.H6(' '),
                                                             dash_table.DataTable(id= 'RMSE', style_table={'width' : 440})
                                                            ]),
                                                   html.Div([''],style={'width':10}),
                                                   html.Div([''],style={'width':20, 'background-color':color}),
                                                           dcc.Graph(id='RC-model-map', style = {'height' : 540, 'width' : 1460}, config = {'displayModeBar': False})
                                                       ], style=dict(display='flex')
                                               ),
                                               html.Div([''], style = {'height':20,'background-color':color}),
                                               html.Div([''], style = {'height':20}),
                                               html.Div([
                                                           html.Div([html.P('Dashboard created by:'), 
                                                                     html.A('Martín Domínguez Durán', href='https://www.linkedin.com/in/mart%C3%ADn-dom%C3%ADnguez-dur%C3%A1n-54b4681b6/', target="_blank")], style = {'width':1460}),
                                                           html.Div([
                                                               html.Plaintext('   Reset modeling environment: ', style={'font-family' : 'bahnschrift'}),
                                                               html.Button('RESET', style={'font-family' : 'bahnschrift','background-color':'darkred','font-size':'14px', 'border' : '0px',
                                                                                                         'color': 'white','border-radius':'12px','width' : 100, 'height' :60},  id='RestartModel', n_clicks=0)
                                                           ], style=dict(display='flex',width = 400))
                                                        ],
                                                           style=dict(display='flex',width = 1900)
                                                       ),
                                               html.P(id='none'),
                                               html.Div(id = 'message'),
                                               html.Div(id = 'notif')
                                            ])
                                          )

@app.callback(
    Output('RC-histogram', 'figure'),
    Input('Organization', 'value'))
def update_graph(Organization):
    
    fig = plot_figure(Organization)
    
    return fig

lst_clicks_rstrt = []
@app.callback(
    Output('none', 'children'),
    Input('RestartModel','n_clicks')
)
def Restart(RestartModel):
    
    lst_clicks_rstrt.append(RestartModel)
    
    if len(lst_clicks_rstrt) == 1:
        PreventUpdate
    elif lst_clicks_rstrt[-1] > lst_clicks_rstrt[-2]:
        for f in listdir('Regression results'):
            if re.search('^RC_regression_estimations', f):
                remove(path.join('Regression results', f))
            if re.search('^Rn_estimations', f):
                remove(path.join('Regression results', f))
                
    return ''


@app.callback(
    Output('FS_out', 'figure'),
    Input('FS','value')
)
def feature_sel(info_FS):
    if info_FS == 'Correlation matrix':
        cor = DF_RC.corr().iloc[1:,1:]

        for i in range(len(cor)):
            for j in range(len(cor)):
                if i < j:
                    cor.iloc[i,j] = np.nan


        fig = px.imshow(cor, color_continuous_scale='RdBu_r', zmin = -1, zmax = 1)

        fig.update_traces(hoverinfo = 'z', hovertemplate = "r_pearson: %{z:.2f}")
    
    elif info_FS == 'Variance Inflation Factor':
        
        fig = px.imshow(tab, color_continuous_scale='RdBu_r', zmax = 4, zmin = 0)
        fig.update_coloraxes(showscale=False)
        fig.update_traces(hoverinfo = 'z', hovertemplate = "VIF: %{z:.2f}")
        
    fig.update_layout(title_text = 'Information for feature selection', 
                      title_font_family = 'bahnschrift',
                      font_family = 'bahnschrift')
            
    return fig

lst_clicks = []

@app.callback(
    Output('message', 'children'),
    [Input('Predict_Rn', 'n_clicks'), Input('model', 'value'), Input('vars_','value'), Input('HQ_model','on')]
)
def update_message(Predict_Rn, model, variables, HQ):
    
    lst_clicks.append(Predict_Rn)
    
    if len(lst_clicks) == 1:
        raise PreventUpdate
    elif lst_clicks[-1] == lst_clicks[-2]:
        raise PreventUpdate
    else:
        var_ID = ''
        for i in variables:
            var_ID += i[0]
        lst_clicks.append(Predict_Rn)
        
        if HQ:
            if path.exists('Regression results/Rn_estimations_'+model+var_ID+'_pol_HQ.geojson'):
                msg = 'This model has already been created. The results should be plotted in less than a minute.'
            else:
                msg = '"Modeling has started. Depending on the size of the dataset, this process will take more or less 30 minutes to plot the estimations..."'
        else:
            if path.exists('Regression results/Rn_estimations_'+model+var_ID+'_pol.geojson'):
                msg = 'This model has already been created. The results should be plotted in less than a minute.'
            else:
                msg = '"Modeling has started. Depending on the size of the dataset, this process will take more or less 5 minutes to plot the estimations..."'
        return dmc.Notification(title="Hey there!",
                         id="simple-notify",
                         action="show",
                         loading=True,
                         color="orange",
                         message=msg,
                         icon=DashIconify(icon="akar-icons:circle-check"),
                         autoClose = False, 
                         disallowClose = True
                        )
    

lst_clicks_mp = []

@app.callback(
    [Output('RC-model-map', 'figure'), Output('imp', 'data'), Output('RMSE', 'data'), Output('notif', 'children')],
    [Input('Predict_Rn','n_clicks'),Input('model', 'value'), Input('vars_','value'), Input('HQ_model','on')]
)

def update_map(Predict_Rn, model, vars_, HQ):
    
    lst_clicks_mp.append(Predict_Rn)
    
    if (Predict_Rn == 0):
        
        notif = ''
        df = pd.DataFrame([[0,-72]])
        df_ = pd.DataFrame([''])
        df_.columns = [' ']
        imp = df_.to_dict('records')
        RMSE = imp
        fig = px.scatter_mapbox(df, lat = 0, lon = 1, opacity = 0)
        
        fig.update_traces(hoverinfo = 'skip', hovertemplate = " ")
        fig.update_layout(mapbox_style="carto-positron",
                          mapbox_zoom = 1.5)
    elif (lst_clicks_mp[-1] == lst_clicks_mp[-2]):
        raise PreventUpdate
        
    elif (lst_clicks_mp[-1] > lst_clicks_mp[-2]):
        
        notif = dmc.Notification(title="Hey there!",
                         id="simple-notify",
                         action="update",
                         message="Model has finished",
                         color = 'green',
                         autoClose = 10*1000,
                         icon=DashIconify(icon="akar-icons:circle-check"),
                        )
        
        if  HQ:    
            print('HQ')
            X, y, msg = read_data(vars_)
            print(msg)
            imp, RMSE, mod, exists, name_file, msg = fit_model(X,y,HQ, model = model)
            print(msg)
            rc_pol, x_c, y_c, msg = EstimatingValues(mod, model, X, exists,HQ, name_file, res=100)
            print(msg)
            # rc_pol, y_c, x_c, msg = rasterize(gdf, name_file, exists, HQ, res = 100)
            # print(msg)
        else:
            print('LQ')
            X, y, msg = read_data(vars_)
            print(msg)
            imp, RMSE, mod, exists, name_file, msg = fit_model(X, y, HQ, model = model)
            print(msg)
            rc_pol, x_c, y_c, msg = EstimatingValues(mod, model, X, exists, HQ, name_file)
            print(msg)
            # rc_pol, y_c, x_c, msg = rasterize(gdf, name_file, exists, HQ)
            # print(msg)

        fig = px.choropleth_mapbox(rc_pol,
                                    geojson=rc_pol.geometry,
                                    locations = rc_pol.index,
                                    color='Reg',
                                    color_continuous_scale="Portland",
                                    opacity = 0.65,
                                    hover_data= ['Reg']
                                   )
        
        fig.update_traces(marker_line_width = 0, hoverinfo = 'z')

        fig.update_layout(mapbox_style="carto-positron",
                          mapbox_center = {'lat':y_c, 'lon':x_c},
                          mapbox_zoom = 10)
        
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
    return fig, imp, RMSE, notif


if __name__ == '__main__':
    app.run_server(debug=True, port = 8070)
    


# In[ ]:




