# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:38:50 2020

NOTE: Purpose of this script is to forecast the next-day number of deaths
      due to COVID in the state of Florida. It accomplishes this by 
      pulling updated data from the NY times github page, filtering by state,
      then fitting a series of ARIMA and/or SARIMA models. The best model
      is retained, and then used for forecasting.

@author: JR Free
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import smtplib, ssl

from sklearn.preprocessing import MinMaxScaler
from scipy.stats import shapiro
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import Config
from datetime import date
plt.style.use('seaborn-darkgrid')


########################
# Function Definitions #
########################

def auto_SARIMA(series , p=3, d=3, q=3, P=2, D=2, Q=2, s=7, extract=5):
    '''
    Parameters
    ----------
    series : pandas Dataframe or Series
        This is the series you wish to which you want to fit an ARIMA model.
    p : int, optional
        Maximum AR parameter to try. The default is 3.
    d : int, optional
        Maximum difference to try. The default is 3.
    q : int, optional
        Maximum MA parameter to try. The default is 3.
    P : int, optional
        Maximum seasonal AR parameter to try. The default is 2.
    D : int, optional
        Maximum seasonal difference parameter to try. The default is 2.
    Q : int, optional
        Maximum seasonal MA parameter to try. The default is 2.
    s : int, optional
        Seasonal period. The default is 7.
    extract : int, optional
        Top number of models to extract. The default is 5.

    Returns
    -------
    models : List of tuples containing ARIMA and/or SARIMA models, BICs, and 
             model orders.

    '''

    # We want to fit either an ARIMA or SARIMA model, so initial range of
    # parameters to fit over.
    _p = range(0,p)
    _d = range(0,d)
    _q = range(0,q)
    _P = range(1,P)
    _D = range(0,D)
    _Q = range(1,Q)
    _s = [s] #weekly process, likely corresponds to reporting cycle time.
    
    # create lists of order combinations for ARIMA parameters and seasonal
    # parameters.
    order = list(itertools.product(_p,_d,_q))
    s_order = list(itertools.product(_P,_D,_Q,_s))
    
    # Initialize lists for holding candidate models, orders, and information
    # criteria.
    models = []
    bics = []
    orders = []
    # Start fitting candidate ARIMA models.
    for o in order:
        print("Fitting...ARIMA",o)
        try:
            # Fit ARIMA
            model = ARIMA(series,o).fit()
            models.append(model)
            bics.append(model.bic)
            orders.append(str(o))
            for s in s_order:
                # Using ARIMA order part from before, fit SARIMA model
                print("Fitting...SARIMA",o,"x",s)
                model= SARIMAX(series, order=o, seasonal_order = s).fit()
                models.append(model)
                bics.append(model.bic)
                orders.append(str(o)+'x'+str(s))
        except ValueError as e:
            print("Model error. Skipping")
            print(e)
        except np.linalg.LinAlgError as f:
            print("Model error. Skipping")
            print(f)
            
    # Extract top models based on BIC (the lower the better).
    models = list(zip(models,bics,orders))
    models.sort(reverse = True,key=lambda tup: tup[1])
    return models

def ADF_test(series, title="ADF Test for Stationarity"):
    '''
    Parameters
    ----------
    series : Pandas dataframe or series
        Series to run ADF test on.
    title : str
        title for output.

    Returns
    -------
    ad_results : str
        String giving results of the test.

    '''
    
    results = adfuller(series)
    ad_results = title + "\np-value: " \
                + str(round(results[1],4))
    if results[1] < 0.001:
        ad_results = ad_results + "\nReject at 0.001 level: series is stationary"
    elif results[1] < 0.01:
        ad_results = ad_results + "\nReject at 0.01 level: series is stationary"
    elif results[1] < 0.05:
        ad_results = ad_results + "\nReject at 0.05 level: series is stationary"
    else:
        ad_results = ad_results + "\nSeries is non-stationary"
    
    return ad_results

def shapiro_report(data, title='Shapiro-Wilk Test'):
    '''
    Parameters
    ----------
    data : Pandas dataframe or series.
        Data to run shapiro wilk test on.
    title : str, optional
        Title for output. The default is 'Shapiro-Wilk Test'.

    Returns
    -------
    shapiro_results: str
        String giving result of the test..

    '''
    
    results = shapiro(data)
    shapiro_results = title + '\nW: ' + \
        str(round(results[0],4)) + '\np-value: ' + str(round(results[1],4))
    if results[1] > 0.10:
        shapiro_results = shapiro_results + \
            '\nVerdict: Do not reject null hypothesis. Cannot claim non-normality at .1 level.'
    elif results[1] > 0.05:
        shapiro_results = shapiro_results + \
            '\nVerdict: Do not reject null hypothesis. Cannot claim non-normality at .05 level.'
    else:
        shapiro_results = shapiro_results + \
            '\nVerdict: Reject null. Residuals are non-normal.'
    
    return shapiro_results

########
# BODY #
########

# Pull data from NY times github.
url='https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv'
covid = pd.read_csv(url,parse_dates=[0],index_col='date')
covid = covid.dropna(thresh=3)

# Isolate Florida data.
# Change 'Florida' to any other state to run this analysis elsewhere.
florida = covid[covid['state']=='Florida']
deaths = florida['deaths'].diff().dropna()

# Extract deaths passed 30 days since recording 
# i.e. throw out transient start-up behavior and focus on stable process.   
deaths2 = deaths[30:]

# Apply minmaxscaler to series and apply sqrt transformation to 
# stablize variance.
scale = MinMaxScaler()
scale.fit(deaths2.values.reshape(-1,1))
deaths2 = (scale.transform(deaths2.values.reshape(-1,1)))**.5

# Find optimal S/ARIMA models.
results = auto_SARIMA(deaths2,extract=5)

# Get fitted model
fitted = results[-1][0]
# Hold on to retained model. Here being the one with lowest BIC.
retained_model = 'Retained model: ' + 'ARIMA' + str(results[-1][2]) + '\n' \
                  + 'BIC: ' + str(round(results[-1][1],3)) + '\n'

# Calculate one-day forecast and confidence interval.
forecast_arima = fitted.get_forecast()
fc = "Forecast: " + str(scale.inverse_transform(forecast_arima.predicted_mean.reshape(-1,1)**2))
ci = "Margin of error: " + str(scale.inverse_transform(forecast_arima.conf_int()**2))

# Conduct shapiro-wilk test on residuals to assess normality.
shapiro_results = shapiro_report(fitted.resid,
                                 title="Shapiro-Wilk Test for Error Process")

# Conduct ADF unit root test on residuals to assess stationarity.
ad_results = ADF_test(fitted.resid,
                      title="ADF Unit Root Test for Error Process")

# Set up report body.
subject = 'Subject: Florida COVID Death Forecast {}\n\n'.format(date.today())
title = "FLORIDA COVID DEATHS ESTIMATE FOR " + str(date.today()) +'\n\n'
output_str = subject + title + retained_model + fc +'\n' + ci +'\n\n' + shapiro_results \
                + '\n\n' + ad_results +'\n'

# Set up e-mail context and send report email.
port = 465  # For SSL

# Import your password from the Config package.
password = Config.password
# Create a secure SSL context
context = ssl.create_default_context()
with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
    server.login("SENDER", password)
    
    sender_email = 'SENDER'
    receiver_email = 'RECEIVER'

    server.sendmail(sender_email, receiver_email, output_str)