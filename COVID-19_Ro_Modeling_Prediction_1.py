#!/usr/bin/env python
# coding: utf-8

# #### Imports for downloading data set, data manipulation and plotting

# In[1]:


import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px


# #### Links to raw files for Covid-19 dataset provided by CSSEGIS JHU

# In[2]:


confirmed_cases_global_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
death_cases_global_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
recovered_cases_global_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv"
confirmed_cases_US_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv"
death_cases_US_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"
country_cases_file_link = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv"


# #### Downloading datasets into respective data frames

# In[3]:


confirmed_df = pd.read_csv(confirmed_cases_global_file_link)
print(confirmed_df.shape)
deaths_df = pd.read_csv(death_cases_global_file_link)
print(deaths_df.shape)
recovered_df = pd.read_csv(recovered_cases_global_file_link)
print(recovered_df.shape)
confirmed_US_df = pd.read_csv(confirmed_cases_US_file_link)
print(confirmed_US_df.shape)
deaths_US_df = pd.read_csv(death_cases_US_file_link)
print(deaths_US_df.shape)
cases_country_df = pd.read_csv(country_cases_file_link)
print(cases_country_df.shape)


# #### Imputing data

# In[4]:


confirmed_df = confirmed_df.replace(np.nan, '', regex=True)
deaths_df = deaths_df.replace(np.nan, '', regex=True)
recovered_df = recovered_df.replace(np.nan, '', regex=True)
confirmed_US_df = confirmed_US_df.replace(np.nan, '', regex=True)
deaths_US_df = deaths_US_df.replace(np.nan, '', regex=True)
cases_country_df = cases_country_df.replace(np.nan, '', regex=True)


# #### Viewing the columns in the dataframe - Timeseries

# In[5]:


confirmed_df.columns


# In[6]:


deaths_df.columns


# In[7]:


recovered_df.columns


# In[8]:


confirmed_US_df.columns


# In[9]:


deaths_US_df.columns


# In[10]:


cases_country_df.columns


# ## Exploratory Data Analysis

# #### For Chart 1 : Total Confirmed Coronavirus Cases (Globally)

# In[11]:


confirmed_ts = confirmed_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis =1)
confirmed_ts_summary = confirmed_ts.sum()


# In[12]:


fig_1 = go.Figure(data=go.Scatter(x=confirmed_ts_summary.index, y = confirmed_ts_summary.values, mode='lines+markers'))
fig_1.update_layout(title='Total Coronavirus Confirmed Cases (Globally)',
                  yaxis_title='Confirmed Cases', xaxis_tickangle = 315 )
fig_1.show()


# ### Defining a template plot function & color arrary

# In[13]:


# Initializing Color Array to be used across the analysis
color_arr = px.colors.qualitative.Dark24


# In[14]:


def draw_plot(ts_array, ts_label, title, colors, mode_size, line_size, x_axis_title , y_axis_title, tickangle = 0, yaxis_type = "", additional_annotations=[]):
    # initialize figure
    fig = go.Figure()
    # add all traces
    for index, ts in enumerate(ts_array):
        fig.add_trace(go.Scatter(x=ts.index,
                                 y = ts.values,
                                 name = ts_label[index],
                                 line=dict(color=colors[index], width=line_size[index]),connectgaps=True,))
    # base x_axis prop.
    x_axis_dict = dict(showline=True, 
                       showgrid=True, 
                       showticklabels=True, 
                       linecolor='rgb(204, 204, 204)', 
                       linewidth=2,
                       ticks='outside',
                       tickfont=dict(family='Arial',size=12,color='rgb(82, 82, 82)',))
    # setting x_axis params
    if x_axis_title:
        x_axis_dict['title'] = x_axis_title
    
    if tickangle > 0:
        x_axis_dict['tickangle'] = tickangle
    
    # base y_axis prop.
    y_axis_dict = dict(showline = True,
                       showgrid = True,
                       showticklabels=True,
                       linecolor='rgb(204, 204, 204)',
                       linewidth=2,)
    # setting y_axis params
    if yaxis_type != "":
        y_axis_dict['type'] = yaxis_type
    
    if y_axis_title:
        y_axis_dict['title'] = y_axis_title
        
#     # uncomment legend if you want to move the legend around
#     legend= dict(x=1,
#                  y=1,
#                  traceorder="normal",
#                  font=dict(family="sans-serif",size=12,color="black"),
#                  bgcolor="LightSteelBlue",
#                  bordercolor="Black",
#                  borderwidth=2)

#updating the layout
    fig.update_layout(xaxis = x_axis_dict,
                      yaxis = y_axis_dict,
                      autosize=True,
                      margin=dict(autoexpand=False,l=100,r=20,t=110,),
                      showlegend=True,
#                       legend = legend
                     )

    # base annotations for any graph
    annotations = []
    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05, xanchor='left', yanchor='bottom',
                                  text=title,
                                  font=dict(family='Arial',size=16,color='rgb(37,37,37)'),showarrow=False))
    # adding annotations in params
    if len(additional_annotations) > 0:
        annotations.append(additional_annotations)
    
    #updating the layout
    fig.update_layout(annotations=annotations)

    return fig


# #### For Chart 2: Covid-19 Case Status

# In[15]:


confirmed_agg_ts = confirmed_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis =1).sum()
death_agg_ts = deaths_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis =1).sum()
recovered_agg_ts = recovered_df.copy().drop(['Lat','Long','Country/Region','Province/State'],axis =1).sum()

#There is no timeseries data for Active cases, therefore it needs to be engineered separately
active_agg_ts = pd.Series(
    data=np.array(
        [x1 - x2 - x3  for (x1, x2, x3) in zip(confirmed_agg_ts.values,death_agg_ts.values, recovered_agg_ts.values)]),
    index= confirmed_agg_ts.index)

#Plot and add traces for all the aggregated timeseries


# In[16]:


ts_array = [confirmed_agg_ts, active_agg_ts, recovered_agg_ts, death_agg_ts]
labels = ['Confirmed', 'Active', 'Recovered', 'Deaths']
colors = [color_arr[5],  color_arr[0], color_arr[2], color_arr[3]]
mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]

# Calling the draw_plot function defined above
fig_2 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Covid-19 Case Status(22nd Jan to 18th November 2020)",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "", additional_annotations=[])

fig_2.show()


# #### For Country wise status

# In[17]:


cases_country_df.copy().drop(
    ['Lat','Long_','Last_Update'],axis =1).sort_values('Confirmed', ascending= False).reset_index(drop=True).style.bar(
    align="left",width=98,color='#d65f5f')


# ### USA Focus

# #### For Chart 3: "Covid-19 Case Trend in US"

# In[18]:


confirmed_US_ts = confirmed_df[confirmed_df['Country/Region']=="US"]
confirmed_US_ts = confirmed_US_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis =1).reset_index(drop=True).sum()

deaths_US_ts = deaths_df[deaths_df['Country/Region']=="US"]
deaths_US_ts = deaths_US_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis =1).reset_index(drop=True).sum()

recovered_US_ts = recovered_df[recovered_df['Country/Region']=="US"]
recovered_US_ts = recovered_US_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis =1).reset_index(drop=True).sum()

active_US_ts = pd.Series(
    data=np.array(
        [x1 - x2 - x3  for (x1, x2, x3) in zip(
            confirmed_US_ts.values,deaths_US_ts.values, recovered_US_ts.values)
        ] 
    ), 
    index= confirmed_US_ts.index
)


# In[19]:


ts_array = [confirmed_US_ts, active_US_ts, recovered_US_ts, deaths_US_ts]
labels = ['Confirmed', 'Active', 'Recovered', 'Deaths']
colors = [color_arr[5],  color_arr[0], color_arr[2], color_arr[3]]
mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]

# Calling the draw_plot function defined above
fig_3 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Covid-19 Case Trend in US",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "", additional_annotations=[])

fig_3.show()


# #### Chart 4: Covid-19 Transmission Timeline in US - 03/01/2020 Onwards

# In[20]:


#Just need to change the ts_array
ts_array = [confirmed_US_ts[39:], active_US_ts[39:], recovered_US_ts[39:], deaths_US_ts[39:]]
fig_4 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Covid-19 Transmission Timeline in US",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "", additional_annotations=[])

fig_4.show()


# #### Chart  5: Semi-Log Plot of Covid-19 Transmission Timeline in US - 03/01/2020 Onwards

# In[21]:


fig_5 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Semi-Log Plot of Covid-19 Transmission Timeline in US",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "log", additional_annotations=[])

fig_5.show()


# ### INDIA Focus

# #### For Chart 6: "Covid-19 Case Trend in India"

# In[22]:


confirmed_India_ts = confirmed_df[confirmed_df['Country/Region']=="India"]
confirmed_India_ts = confirmed_India_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis =1).reset_index(drop=True).sum()

deaths_India_ts = deaths_df[deaths_df['Country/Region']=="India"]
deaths_India_ts = deaths_India_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis =1).reset_index(drop=True).sum()

recovered_India_ts = recovered_df[recovered_df['Country/Region']=="India"]
recovered_India_ts = recovered_India_ts.drop(
    ['Lat','Long','Country/Region','Province/State'],axis =1).reset_index(drop=True).sum()

active_India_ts = pd.Series(
    data=np.array(
        [x1 - x2 - x3  for (x1, x2, x3) in zip(
            confirmed_India_ts.values,deaths_India_ts.values, recovered_India_ts.values)
        ] 
    ), 
    index= confirmed_India_ts.index
)


# In[23]:


ts_array = [confirmed_India_ts, active_India_ts, recovered_India_ts, deaths_India_ts]
labels = ['Confirmed', 'Active', 'Recovered', 'Deaths']
colors = [color_arr[5],  color_arr[0], color_arr[2], color_arr[3]]
mode_size = [8, 8, 12, 8]
line_size = [2, 2, 4, 2]

# Calling the draw_plot function defined above
fig_6 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Covid-19 Case Trend in INDIA",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "", additional_annotations=[])

fig_6.show()


# #### Chart 7: Covid-19 Transmission Timeline in INDIA - 03/01/2020 Onwards

# In[24]:


#Just need to change the ts_array
ts_array = [confirmed_India_ts[39:], active_India_ts[39:], recovered_India_ts[39:], deaths_India_ts[39:]]
fig_7 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Covid-19 Transmission Timeline in INDIA",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "", additional_annotations=[])

fig_7.show()


# #### Chart  8: Semi-Log Plot of Covid-19 Transmission Timeline in INDIA - 03/01/2020 Onwards

# In[25]:


fig_8 = draw_plot(ts_array = ts_array, 
                  ts_label = labels , 
                  title = "Semi-Log Plot of Covid-19 Transmission Timeline in INDIA",
                  colors = colors, mode_size = mode_size, 
                  line_size = line_size , 
                  x_axis_title = "Date" , 
                  y_axis_title = "Case Count", 
                  tickangle = 315, 
                  yaxis_type = "log", additional_annotations=[])

fig_8.show()


# ## Modeling & Prediction

# In[26]:


# Imports required
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from datetime import timedelta, datetime


# In[31]:


START_DATE = {
  'US': '12/25/20',
  'India': '12/25/20'
}
class Learner(object):
    def __init__(self, country, loss, start_date ='12/25/20', predict_range=150,s_0=100000, i_0=2, r_0=10):
        self.country = country
        self.loss = loss
        self.start_date = start_date
        self.predict_range = predict_range
        self.s_0 = s_0
        self.i_0 = i_0
        self.r_0 = r_0

    def load_confirmed(self, country):
        df = pd.read_csv(confirmed_cases_global_file_link)
        df = df.drop(['Province/State'],axis =1)
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_recovered(self, country):
        df = pd.read_csv(recovered_cases_global_file_link)
        df = df.drop(['Province/State'],axis =1)
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]


    def load_dead(self, country):
        df = pd.read_csv(death_cases_global_file_link)
        df = df.drop(['Province/State'],axis =1)
        country_df = df[df['Country/Region'] == country]
        return country_df.iloc[0].loc[self.start_date:]
    

    def extend_index(self, index, new_size):
        values = index.values
        current = datetime.strptime(index[-1], '%m/%d/%y')
        while len(values) < new_size:
            current = current + timedelta(days=1)
            values = np.append(values, datetime.strftime(current, '%m/%d/%y'))
        return values

    def predict(self, beta, gamma, data, recovered, death, country, s_0, i_0, r_0):
        """
        Predict how the number of people in each compartment can be changed through time toward the future.
        The model is formulated with the given beta and gamma.
        """
        new_index = self.extend_index(data.index, self.predict_range)
        size = len(new_index)
        def SIR(t, y):
            S = y[0]
            I = y[1]
            R = y[2]
            return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
        extended_actual = np.concatenate((data.values, [None] * (size - len(data.values))))
        extended_recovered = np.concatenate((recovered.values, [None] * (size - len(recovered.values))))
        extended_death = np.concatenate((death.values, [None] * (size - len(death.values))))
        return new_index, extended_actual, extended_recovered, extended_death, solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1))
    
    def train(self):
        """
        Run the optimization to estimate the beta and gamma fitting the given confirmed cases.
        """
        recovered = self.load_recovered(self.country)
        death = self.load_dead(self.country)
        data = (self.load_confirmed(self.country) - recovered - death)
        
        optimal = minimize(
            loss, 
            [0.001, 0.001], 
            args=(data, recovered, self.s_0, self.i_0, self.r_0), 
            method='L-BFGS-B', 
            bounds=[(0.00000001, 0.4), (0.00000001, 0.4)]
        )
        print(optimal)
        beta, gamma = optimal.x
        new_index, extended_actual, extended_recovered, extended_death, prediction = self.predict(beta, gamma, data, recovered, death, self.country, self.s_0, self.i_0, self.r_0)
        df = pd.DataFrame({'Infected data': extended_actual, 'Recovered data': extended_recovered, 'Death data': extended_death, 'Susceptible': prediction.y[0], 'Infected': prediction.y[1], 'Recovered': prediction.y[2]}, index=new_index)
        df.to_csv(f"{self.country}.csv")
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_title(self.country)
        df.plot(ax=ax)
        print(f"country={self.country}, beta={beta:.8f}, gamma={gamma:.8f}, r_0:{(beta/gamma):.8f}")
        fig.savefig(f"{self.country}.png")
        
        return df, fig


# In[32]:


def loss(point, data, recovered, s_0, i_0, r_0):
    size = len(data)
    beta, gamma = point
    def SIR(t, y):
        S = y[0]
        I = y[1]
        R = y[2]
        return [-beta*S*I, beta*S*I-gamma*I, gamma*I]
    solution = solve_ivp(SIR, [0, size], [s_0,i_0,r_0], t_eval=np.arange(0, size, 1), vectorized=True)
    l1 = np.sqrt(np.mean((solution.y[1] - data)**2))
    l2 = np.sqrt(np.mean((solution.y[2] - recovered)**2))
    alpha = 0.1
    return alpha * l1 + (1 - alpha) * l2


# ### For US

# In[33]:


us_learner = Learner(country="US", loss= loss )


# In[34]:


us_df, us_fig = us_learner.train()


# In[35]:


us_learner = Learner(country="US", loss= loss )


# In[36]:


us_df, us_fig = us_learner.train()


# ### For India

# In[37]:


india_learner = Learner(country="India", loss= loss )


# In[38]:


india_df, india_fig = india_learner.train()


# ### For Italy

# In[39]:


italy_learner = Learner(country="Italy", loss= loss )


# In[40]:


italy_df, italy_fig = italy_learner.train()


# ### US SIR Model

# In[41]:


us_learner = Learner(country="US", loss= loss, i_0= 3)


# In[42]:


us_sir , us_sir_fig = us_learner.train()


# In[43]:


us_sir = pd.read_csv('US.csv')


# In[44]:


us_sir = us_sir[:77]


# In[45]:


def plot_sir_prediction(title, df_sir, remove_series=[],yaxis_type="", yaxis_title=""):
    fig = go.Figure()
    title = title
    labels = ['Infected data','Recovered data', 'Death data', 'Susceptible','Infected','Recovered']
    colors = [color_arr[0], color_arr[8], color_arr[2], color_arr[3],color_arr[7], color_arr[16]]
    line_size = [2, 2, 2, 2, 2, 2]
    
    for index, data_series in enumerate(labels):
        if data_series not in remove_series:
            fig.add_trace(go.Scatter(x=df_sir.index, 
                                           y = df_sir[data_series], 
                                           name = labels[index],
                                           line=dict(
                                               color=colors[index], 
                                               width=line_size[index]),
                                           connectgaps=True,))
    xaxis= dict(
        title = "Date",
        showline=True,
        showgrid=True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,
        ticks='outside',
        tickangle = 280,
        tickfont=dict(
            family='Arial',
            size=12,
            color='rgb(82, 82, 82)',),)
    yaxis = dict(
        title = "Case Count",
        showline = True,
        showgrid = True,
        showticklabels=True,
        linecolor='rgb(204, 204, 204)',
        linewidth=2,)
    
    if yaxis_type!="":
        yaxis['type'] = yaxis_type
    
    if yaxis_title !="":
        yaxis['title'] = yaxis_title
    
    fig.update_layout(
        xaxis = xaxis,
        yaxis = yaxis,
        autosize=True,
        margin=dict(autoexpand=True,l=100,r=20,t=110,),
        showlegend=True)

    annotations = []

    # Title
    annotations.append(dict(xref='paper', yref='paper', x=0.0, y=1.05,
                              xanchor='left', yanchor='bottom',
                              text=title,
                              font=dict(family='Arial',
                                        size=16,
                                        color='rgb(37,37,37)'),
                              showarrow=False))

    fig.update_layout(annotations=annotations)
    
    return fig


# ### For Chart : SIR Model -- Covid-19 Transmission -- Prediction -- US

# In[46]:


fig_8 = plot_sir_prediction(title="SIR Model --  Covid-19 Transmission -- Prediction -- US", df_sir= us_sir)
fig_8.show()


# In[47]:


# to remove the "Susceptible" series just pass the series name to remove_series
fig_8 = plot_sir_prediction(
    title="SIR Model --  Covid-19 Transmission -- Prediction -- US", 
    df_sir= us_sir,
    remove_series=['Susceptible'])

fig_8.show()


# ### For Chart 9 : SIR Model -- Covid-19 Transmission -- Prediction -- US | Case Count (Log Scale)

# In[48]:


fig_9 = plot_sir_prediction(
    title="SIR Model --  Covid-19 Transmission -- Prediction -- US", 
    df_sir= us_sir,
    remove_series=['Susceptible'],
    yaxis_type="log",
    yaxis_title="Case Count (Log Scale)")

fig_9.show()

