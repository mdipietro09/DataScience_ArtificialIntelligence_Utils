
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from sklearn import preprocessing
from tensorflow.keras import models, layers
import itertools
import tqdm



###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################
'''
'''
def create_y(dtf, col, shifts=1):
    dtf = dtf.sort_index(ascending=False)
    dtf["Y"] = dtf[col].shift(shifts)
    print("First row:", str(dtf.index[0]), "| NA:", dtf["Y"].isna()[0] == True)
    print("...")
    print("Last row:", str(dtf.index[-1]), "| NA:", dtf["Y"].isna()[-1] == True)
    return dtf



'''
'''
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(20,13)):
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title(ts.name)
    if plot_ma:
        plt.plot(rolling_mean, 'g', label='MA'+str(window))
    if plot_intervals:
        #mean_absolute_error = np.mean(np.abs((ts[window:] - rolling_mean[window:]) / ts[window:])) * 100
        #deviation = np.std(ts[window:] - rolling_mean[window:])
        #lower_bound = rolling_mean - (mean_absolute_error + 1.96 * deviation)
        #upper_bound = rolling_mean + (mean_absolute_error + 1.96 * deviation)
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        plt.plot(lower_bound, 'r--')
    plt.plot(ts[window:], label='Actual values', linewidth=3)
    plt.legend(loc='best')
    plt.grid(True)
        
    

'''
'''
def plot_2_ts(ts1, ts2, figsize=(20,13)):
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, sharey=False, figsize=figsize)
    ax[0].plot(ts1)
    ax[0].set_title(ts1.name)
    ax[0].grid(True)
    ax[1].plot(ts2)
    ax[1].set_title(ts2.name)
    ax[1].grid(True)
    


'''
'''
def plot_acf_pacf(ts, lags=30, figsize=(20,13)):
    with plt.style.context(style='bmh'):
        fig = plt.figure(figsize=figsize)
        layout = (2,2)
        ts_ax = plt.subplot2grid(layout, (0,0), colspan=2)
        acf_ax = plt.subplot2grid(layout, (1,0))
        pacf_ax = plt.subplot2grid(layout, (1,1))
        ts.plot(ax=ts_ax)
        p_value = sm.tsa.stattools.adfuller(ts)[1]
        ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.format(p_value))
        smt.graphics.plot_acf(ts, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(ts, lags=lags, ax=pacf_ax)
        plt.tight_layout()    
    


'''
'''
def diff_ts(ts, shifts=1, na="fill"):
    tsdiff = ts - ts.shift(-shifts)
    if na == "drop":
        tsdiff = tsdiff[(pd.notnull(tsdiff))]
    elif na == "fill":
        tsdiff = tsdiff.fillna(method="ffill")
    return tsdiff



'''
'''
def decompose_ts(ts, freq=250, figsize=(20,13)):
    decomposition = smt.seasonal_decompose(ts, freq=freq)
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid   
    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True, sharey=False, figsize=figsize)
    ax[0].plot(ts)
    ax[0].set_title('Original')
    ax[0].grid(True) 
    ax[1].plot(trend)
    ax[1].set_title('Trend')
    ax[1].grid(True)  
    ax[2].plot(seasonal)
    ax[2].set_title('Seasonality')
    ax[2].grid(True)  
    ax[3].plot(residual)
    ax[3].set_title('Residuals')
    ax[3].grid(True)
    return {"trend":trend, "seasonal":seasonal, "residual":residual}



###############################################################################
#                          FORECAST                                           #
###############################################################################
'''
Fits best Seasonal-ARIMA.
:parameter
    :param ts: padas - raw timeseries
    :param d: num - integration order
    :param D: num - seasonal integration order
    :param s: num - length of season
    :param max_integration - bound for params trials
:return
    best model
'''
def fit_arima(ts, d=1, D=1, s=60, max_integration=3, ahead=5, figsize=(20,13)):
    ## init params
    import warnings
    warnings.filterwarnings("ignore")
    ps, qs, Ps, Qs = range(0,max_integration), range(0,max_integration), range(0,max_integration), range(0,max_integration)
    lst_params = list(itertools.product(ps, qs, Ps, Qs))
    
    ## itera per best model
    best_aic = float('inf')
    for param in tqdm.tqdm(lst_params):
        try: 
            model = smt.statespace.SARIMAX(ts, order=(param[0], d, param[1]),
                                           seasonal_order=(param[2], D, param[3], s)
                                           ).fit(disp=-1)
        except:
            continue
        aic = model.aic
        if aic < best_aic:
            best_model = model
            best_aic = aic
            best_param = param
            
    ## predict
    preds = best_model.forecast(steps=ahead)
    
    ## plot
    print("best model --> p, q, P, Q:", best_param)
    print(best_model.summary())
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(ts.values)
    ax.plot(best_model.fittedvalues.values, color='red')
    ax.plot(preds, color='red', linewidth=3)
    ax.grid(True)
    return preds



'''
'''
def fit_prophet(ts, ahead=5, figsize=(20,13)):
    return 0



###############################################################################
#                            RNN                                              #
###############################################################################
'''
'''
def ts_preprocessing(ts, scaler=None, size=20):
    ts = ts.sort_index(ascending=True).values
    
    ## scale
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    ts_preprocessed = scaler.fit_transform(ts.reshape(-1,1))          
    
    ## create X y for train
    lst_X, lst_y = [], []
    for i in range(len(ts_preprocessed)):
        end_ix = i + size
        if end_ix > len(ts_preprocessed)-1:
            break
        Xi, yi = ts_preprocessed[i:end_ix], ts_preprocessed[end_ix]
        lst_X.append(Xi)
        lst_y.append(yi)
    X = np.array(lst_X)
    y = np.array(lst_y)
    X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return {"X":X, "y":y, "scaler":scaler}



'''
'''
def fit_lstm(X, y, batch_size=32, epochs=100, figsize=(20,13)):
    ## lstm
    model = models.Sequential()
    model.add( layers.LSTM(input_shape=X.shape[1:], units=50, activation='tanh', return_sequences=False) )
    model.add( layers.Dense(1) )
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    ## fit
    training = model.fit(x=X, y=y, batch_size=batch_size, epochs=epochs, shuffle=True)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(training.history['loss'], label='loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    print(training.model.summary())
    return training.model



'''
'''
def evaluate_lstm(model, X_test, Y_test):
    return 0



'''
'''
def predict_lstm(ts, model, scaler, size=20, ahead=5, figsize=(20,13)):
    ## preprocess
    ts = ts.sort_index(ascending=True).values
    ts_preprocessed = list(scaler.fit_transform(ts.reshape(-1,1)))
    
    ## validation
    lst_fitted = [np.nan]*size
    for i in range(len(ts_preprocessed)):
        end_ix = i + size
        if end_ix > len(ts_preprocessed)-1:
            break
        X = ts_preprocessed[i:end_ix]
        X = np.array(X)
        X = np.reshape(X, (1, 1, X.shape[0]))
        fit = model.predict(X)
        fit = scaler.inverse_transform(fit)[0][0]
        lst_fitted.append(fit)
         
    ## predict
    lst_preds = []
    for i in range(ahead):
        i += 1
        lst_X = ts_preprocessed[len(ts_preprocessed)-(size+1) : -1]
        X = np.array(lst_X)
        X = np.reshape(X, (1, 1, X.shape[0]))
        pred = model.predict(X)
        ts_preprocessed.append(pred)
        pred = scaler.inverse_transform(pred)[0][0]
        lst_preds.append({"actual":np.nan, "pred":pred})
        
    ## plot
    dtf = pd.DataFrame({'actual':ts, 'fitted':lst_fitted, 'pred':np.nan})
    dtf = pd.concat( [dtf, pd.DataFrame([dic for dic in lst_preds])] )
    dtf = dtf.reset_index(drop=True)
    ax = dtf.plot(title="pred "+str(ahead)+" ahead", figsize=figsize, linewidth=3)
    ax.grid(True)
    return dtf

