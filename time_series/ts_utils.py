
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
from fbprophet import Prophet
import fbprophet.plot as fbPlot
pd.plotting.register_matplotlib_converters()
from sklearn import preprocessing
from tensorflow.keras import models, layers
import itertools
import tqdm
import warnings
warnings.filterwarnings("ignore")



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



'''
'''
def compute_model_stats(dtf, columns=["ts","fitted","preds"], plot=True, figsize=(20,13)):
    try:
        col_ts, col_fitted, col_preds = columns[0], columns[1], columns[2]
        
        ## residuals
        dtf["error"] = dtf[col_ts] - dtf[col_fitted]
        dtf["error_pct"] =  dtf["error"] / dtf[col_ts]
        
        ## kpi
        error_mean = dtf["error"].mean()  #errore medio
        error_std = dtf["error"].std()  #standard dev dell'errore
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()  #mean absolute error
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  #mean absolute error %
        mse = dtf["error"].apply(lambda x: x**2).mean() # mean squared error
        rmse = np.sqrt(mse)  #root mean squared error
        print("error_mean:",np.round(error_mean), "error_std:",np.round(error_std),
              "mae:",np.round(mae), "mape:",np.round(mape*100), "mse:",np.round(mse), 
              "rmse:",np.round(rmse))
        
        ## intervals
        dtf["ci_lower_bound"] = dtf[col_preds] - 1.96*error_std
        dtf["ci_upper_bound"] = dtf[col_preds] + 1.96*error_std
        #dtf["pi_lower_bound"] = dtf[col_preds] - 1.96*mae
        #dtf["pi_upper_bound"] = dtf[col_preds] + 1.96*mae
        
        ## plot
        if plot==True:
            ### ts
            ax = dtf[columns].plot(figsize=figsize, color=["black","red","blue"], title="Fitted Model")
            #ax.fill_between(x=dtf.index, y1=dtf['pi_lower_bound'], y2=dtf['pi_lower_bound'], color='b', alpha=0.2)
            ax.fill_between(x=dtf.index, y1=dtf['ci_lower_bound'], y2=dtf['ci_lower_bound'], color='b', alpha=0.3)
            ax.grid(True)
            plt.show()
            
            ### residuals 
            fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle("Residuals", fontsize=20)
            dtf["error"].plot(ax=ax[0])
            ax[0].grid(True)
            dtf["error"].plot(ax=ax[1], kind='kde')
            ax[1].grid(True)
            plt.show()
        
        return dtf
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        


'''
'''
def split_train_test(dtf, column_split, first_split="2017-01-01"):
    first_dtf_training = dtf[dtf[column_split] < first_split]
    lst_training = [first_dtf_training]
    lst_splits = dtf[dtf[column_split] > first_split][column_split].unique()
    for split in lst_splits:
        print(split)
        dtf_training = dtf[dtf[column_split] <= split]
        lst_training.append(dtf_training)
    return lst_training
    
    

###############################################################################
#                             ARIMA                                           #
###############################################################################
'''
Fits SARIMAX (Seasonal ARIMA with External Regressors).
:parameter
    :param ts: pandas timeseries
    :param model: SARIMAX model or None
    :param order: tuple - ARIMA(p,d,q) --> p: lag order, d: degree of differencing, q: order of moving average 
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of time steps for a single seasonal (4 for quarterly data, 12 for monthly data, 250 for business days, 365 for daily data
    :param trend: str or None - "c" constant, "t" linear, "ct" both
    :param exog: array
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param pred_ahead: num - predictions ahead
    :param pred_exog: array
    :param pred_confidence: num - confidence interval
    :param figsize: tuple - matplotlib figure
:return
    ts with predictons
'''
def fit_arima(ts, model=None, order=(1,1,1), seasonal_order=(0,0,0,0), exog=None, trend=None, 
              pred_ahead=5, pred_exog=None, pred_confidence=0.95, figsize=(20,13)):
    ## fit model
    if model is None:
        model = smt.SARIMAX(ts, order=order, seasonal_order=seasonal_order, exog=exog, freq=None).fit(disp=-1, trend='c')
            
    ## forecast
    if pred_ahead != 0:
        forecast = model.get_forecast(steps=pred_ahead, alpha=1-pred_confidence, exog=pred_exog)
        preds = forecast.predicted_mean.values
        std_err = forecast.se_mean.values
        conf_int = forecast.conf_int().values
        # conf_int 95% = [pred - 1.96*std_err , pred + 1.96*std_err]
        
        ### dtf out
        dtf_preds = pd.DataFrame({'lower_confint':conf_int[:,0], 'preds':preds, 'upper_confint':conf_int[:,1]})
        dtf_ts = pd.DataFrame({'ts':ts.values, 'fitted':model.fittedvalues.values})
        dtf_ts = dtf_ts.append(dtf_preds, sort=False)
        dtf_ts = dtf_ts.reset_index(drop=True)
        
        ### plot ts
        ax = dtf_ts[["ts", "fitted", "preds"]].plot(figsize=figsize, color=["black","red","green"])
        ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower_confint'], y2=dtf_ts['upper_confint'], color='b', alpha=0.2)
        ax.grid(True)
        plt.show()
        
        ### plot residual
        dtf_residuals = pd.DataFrame(model.resid)
        fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=figsize)
        fig.suptitle("Residuals", fontsize=20)
        dtf_residuals.plot(ax=ax[0])
        ax[0].grid(True)
        dtf_residuals.plot(ax=ax[1], kind='kde')
        ax[1].grid(True)
        plt.show()     
        return dtf_ts
    
    else:
        return model


    
'''
Fits best Seasonal-ARIMAX.
:parameter
    :param ts: pandas timeseries
    :param d: num - degree of differencing
    :param D: num - seasonal integration order
    :param s: num - number of time steps for a single seasonal (4 for quarterly data, 12 for monthly data, 250 for business days, 365 for daily data
    :param max_integration - bound for params trials
:return
    best model
'''
def find_best_arima(ts, d=1, D=1, s=365, max_integration=3):
    ## init params
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
    print("best model --> p, q, P, Q:", best_param)
    return best_model



'''
Fits HoltWinters Exponential Smoothing.
:parameter
    :param ts: pandas timeseries
    :param trend: str or None - "additive", "multiplicative"
    :param s: num - number of time steps for a single seasonal (4 for quarterly data, 12 for monthly data, 250 for business days, 365 for daily data
    :param alpha: num - the alpha value of the simple exponential smoothing (ex 0.94)
    :param pred_ahead: num - predictions ahead
:return
    ts with predictons
'''
def fit_expsmooth(ts, trend=None, s=None, alpha=None, pred_ahead=5, figsize=(20,13)):
    ## fit model
    model = smt.ExponentialSmoothing(ts, trend=trend, seasonal_periods=s).fit(smoothing_level=alpha)
    preds = model.forecast(pred_ahead)
    #std_err = model.sse
    
    ## dtf out
    dtf_preds = pd.DataFrame({'preds':preds})
    #dtf_preds["lower_confint"] = dtf_preds["preds"] - 1.96*std_err
    #dtf_preds["higher_confint"] = dtf_preds["preds"] + 1.96*std_err
    dtf_ts = pd.DataFrame({'ts':ts.values, 'fitted':model.fittedvalues.values})
    dtf_ts = dtf_ts.append(dtf_preds, sort=False)
    dtf_ts = dtf_ts.reset_index(drop=True)
        
    ## plot ts
    ax = dtf_ts.plot(figsize=figsize, color=["black","red","green"])
    #ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower_confint'], y2=dtf_ts['higher_confint'], color='b', alpha=0.2)
    ax.grid(True)
    plt.show()
        
    ## plot residual
    dtf_residuals = pd.DataFrame(model.resid)
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Residuals", fontsize=20)
    dtf_residuals.plot(ax=ax[0])
    ax[0].grid(True)
    dtf_residuals.plot(ax=ax[1], kind='kde')
    ax[1].grid(True)
    plt.show()     
    return dtf_ts



###############################################################################
#                           PROPHET                                           #
###############################################################################
'''
Fits prophet.
:parameter
    :param ts: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param interval_width: num - the portion of dataset used to compute upper and lower bounds with MonteCarlo
    :param freq: str - "D" daily, "M" monthly, "Y" annual
    
    :param growth: str - 'linear' or 'logistic' trend. "logistic" is for forecasting growth and needs capacity (ex. total market size, total population size)
    :param changepoints: list or None - dates at which to include potential changepoints
    :param n_changepoints: num or None - number of potential automatic changepoints to include
    
    :param yearly_seasonality: str or bool - "auto", True or False
    :param weekly_seasonality: str or bool - "auto", True or False
    :param daily_seasonality: str or bool - "auto", True or False
    :param seasonality_mode: str - 'additive' or 'multiplicative'
    
    :param holidays: pandas - DataFrame with columns 'ds' (dates) and 'holiday' (string ex 'xmas')

    :param lst_exog: list - names of variables
    :paam pred_exog: array - values of exog variables
:return
    ts with predictons
'''
def fit_prophet(ts, freq="D", interval_width=0.80, preds_ahead=5, figsize=(20,13),
                growth="linear", changepoints=None, n_changepoints=25,
                yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto", seasonality_mode='additive',
                holidays=None, lst_exog=None, pred_exog=None):
    ## fit model
    model = Prophet(growth, changepoints=changepoints, n_changepoints=n_changepoints,
                    yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality,seasonality_mode=seasonality_mode,
                    holidays=holidays, interval_width=interval_width)
    if lst_exog != None:
        for regressor in lst_exog:
            model.add_regressor(regressor)
        
    model.fit(ts)
    
    ## predict
    dtf_prophet = model.make_future_dataframe(periods=preds_ahead, freq=freq, include_history=True)
    
    if growth == "logistic":
        dtf_prophet["cap"] = ts["cap"].unique()[0]
    
    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(ts[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-preds_ahead:][lst_exog] = pred_exog
    
    dtf_prophet = model.predict(dtf_prophet)
    
    ## plot prophet
    fbPlot.plot(model, dtf_prophet, figsize=figsize)
    fbPlot.plot_components(model, dtf_prophet, figsize=figsize)
    
    ## dtf
    fitted = dtf_prophet.iloc[:len(ts)][["ds", "yhat"]]
    forecast = dtf_prophet.iloc[len(ts):][["ds", "yhat_lower", "yhat", "yhat_upper"]]
    dtf_ts = fitted.merge(ts, how="left")
    dtf_ts = dtf_ts.rename(columns={"y":"ts", "yhat":"fitted"})
    dtf_ts = dtf_ts.append(forecast, sort=False)
    dtf_ts = dtf_ts.rename(columns={"yhat_lower":"lower_confint", "yhat":"preds", "yhat_upper":"upper_confint"})
    dtf_ts = dtf_ts.set_index("ds")

    ## plot ts
    ax = dtf_ts[["ts", "fitted", "preds"]].plot(figsize=figsize, color=["black","red","green"])
    ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower_confint'], y2=dtf_ts['upper_confint'], color='b', alpha=0.2)
    ax.grid(True)
    plt.show()
        
    ## plot residual
    dtf_residuals = dtf_ts["ts"] - dtf_ts["fitted"]
    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Residuals", fontsize=20)
    dtf_residuals.plot(ax=ax[0])
    ax[0].grid(True)
    dtf_residuals.plot(ax=ax[1], kind='kde')
    ax[1].grid(True)
    plt.show()
    
    return dtf_ts



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

