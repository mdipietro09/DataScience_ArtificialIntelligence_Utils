
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pmdarima
from fbprophet import Prophet, diagnostics, plot as fbPlot
pd.plotting.register_matplotlib_converters()
from sklearn import preprocessing
from tensorflow.keras import models, layers, preprocessing as kprocessing
import datetime
import warnings
warnings.filterwarnings("ignore")



###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################
'''
'''
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(20,13)):
    rolling_mean = ts.rolling(window=window).mean()
    rolling_std = ts.rolling(window=window).std()
    plt.figure(figsize=figsize)
    plt.title(ts.name)
    plt.plot(ts[window:], label='Actual values', color="black")
    if plot_ma:
        plt.plot(rolling_mean, 'g', label='MA'+str(window), color="red")
    if plot_intervals:
        #mean_absolute_error = np.mean(np.abs((ts[window:] - rolling_mean[window:]) / ts[window:])) * 100
        #deviation = np.std(ts[window:] - rolling_mean[window:])
        #lower_bound = rolling_mean - (mean_absolute_error + 1.96 * deviation)
        #upper_bound = rolling_mean + (mean_absolute_error + 1.96 * deviation)
        lower_bound = rolling_mean - (1.96 * rolling_std)
        upper_bound = rolling_mean + (1.96 * rolling_std)
        #plt.plot(upper_bound, 'r--', label='Upper bound / Lower bound')
        #plt.plot(lower_bound, 'r--')
        plt.fill_between(x=ts.index, y1=lower_bound, y2=upper_bound, color='lightskyblue', alpha=0.4)
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
        


'''
'''
def test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30, figsize=(20,13)):
    with plt.style.context(style='bmh'):
        ## set figure
        fig = plt.figure(figsize=figsize)
        ts_ax = plt.subplot2grid(shape=(2,2), loc=(0,0), colspan=2)
        pacf_ax = plt.subplot2grid(shape=(2,2), loc=(1,0))
        acf_ax = plt.subplot2grid(shape=(2,2), loc=(1,1))
        
        ## plot ts with mean/std of a sample from the first x% 
        dtf_ts = ts.to_frame(name="ts")
        sample_size = int(len(ts)*sample)
        dtf_ts["mean"] = dtf_ts["ts"].head(sample_size).mean()
        dtf_ts["lower"] = dtf_ts["ts"].head(sample_size).mean() + dtf_ts["ts"].head(sample_size).std()
        dtf_ts["upper"] = dtf_ts["ts"].head(sample_size).mean() - dtf_ts["ts"].head(sample_size).std()
        dtf_ts["ts"].plot(ax=ts_ax, color="black", legend=False)
        dtf_ts["mean"].plot(ax=ts_ax, legend=False, color="red", linestyle="--", linewidth=0.7)
        ts_ax.fill_between(x=dtf_ts.index, y1=dtf_ts['lower'], y2=dtf_ts['upper'], color='lightskyblue', alpha=0.4)
        dtf_ts["mean"].head(sample_size).plot(ax=ts_ax, legend=False, color="red", linewidth=0.9)
        ts_ax.fill_between(x=dtf_ts.head(sample_size).index, y1=dtf_ts['lower'].head(sample_size), y2=dtf_ts['upper'].head(sample_size), color='lightskyblue')
        
        ## test stationarity (Augmented Dickey-Fuller)
        adfuller_test = sm.tsa.stattools.adfuller(ts, maxlag=maxlag, autolag="AIC")
        adf, p, critical_value = adfuller_test[0], adfuller_test[1], adfuller_test[4]["5%"]
        p = round(p, 3)
        conclusion = "Stationary" if p < 0.05 else "Non-Stationary"
        ts_ax.set_title('Dickey-Fuller Test 95%: '+conclusion+' (p-value: '+str(p)+')')
        
        ## pacf (for AR) e acf (for MA) 
        smt.graphics.plot_pacf(ts, lags=maxlag, ax=pacf_ax, title="Partial Autocorrelation (for AR component)")
        smt.graphics.plot_acf(ts, lags=maxlag, ax=acf_ax, title="Autocorrelation (for MA component)")
        plt.tight_layout()    
   
    

'''
'''
def diff_ts(ts, lag=1, order=1, na="drop"):
    for i in range(order):
        ts = ts - ts.shift(lag)
    ts = ts[(pd.notnull(ts))] if na == "drop" else ts.fillna(method="bfill")
    return ts



'''
'''
def undo_diff(ts, first_y, lag=1, order=1):
    for i in range(order):
        (24168.04468 - 18256.02366) + a.cumsum()
        ts = np.r_[ts, ts[lag:]].cumsum()
    return ts



'''
'''
def test_2ts_casuality(ts1, ts2, maxlag=30, figsize=(20,13)):
    ## prepare
    dtf = ts1.to_frame(name=ts1.name)
    dtf[ts2.name] = ts2
    dtf.plot(figsize=figsize, grid=True, title=ts1.name+"  vs  "+ts2.name)
    plt.show()
    ## test casuality (Granger test) 
    granger_test = sm.tsa.stattools.grangercausalitytests(dtf, maxlag=maxlag, verbose=False)
    for lag,tupla in granger_test.items():
        p = np.mean([tupla[0][k][1] for k in tupla[0].keys()])
        p = round(p, 3)
        if p < 0.05:
            conclusion = "Casuality with lag "+str(lag)+" (p-value: "+str(p)+")"
            print(conclusion)
        


'''
'''
def decompose_ts(ts, s=250, figsize=(20,13)):
    decomposition = smt.seasonal_decompose(ts, freq=s)
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
#                 MODEL DESIGN & TESTING - FORECASTING                        #
###############################################################################
'''
'''
def utils_split_train_test(ts, exog=None, test=0.2):
    if type(test) is float:
        split = int(len(ts)*(1-test))
    elif type(test) is str:
        split = ts.index.tolist().index(test)
    else:
        split = test
    print("--- splitting at index: ", split, " ---")
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)  
    if exog is not None:
        exog_train = exog[0:split] 
        exog_test = exog[split:]
        return ts_train, ts_test, exog_train, exog_test
    else:
        return ts_train, ts_test
    


'''
'''
def utils_evaluate_forecast(dtf, title, columns=["ts","fitted","test","forecast"], plot=True, figsize=(20,13)):
    try:
        col_ts, col_fitted, col_test, col_preds = columns[0], columns[1], columns[2], columns[3]
        
        ## residuals
        dtf["fitting_error"] = dtf[col_ts] - dtf[col_fitted]
        dtf["error_pct"] =  dtf["fitting_error"] / dtf[col_ts]
        dtf["prediction_error"] = dtf[col_ts] - dtf[col_test]
        dtf["prediction_error_pct"] = dtf["prediction_error"] / dtf[col_ts]
        
        ## kpi
        ### fitting
        error_mean = dtf["fitting_error"].mean()  #errore medio
        error_std = dtf["fitting_error"].std()  #standard dev dell'errore
        mae = dtf["fitting_error"].apply(lambda x: np.abs(x)).mean()  #mean absolute error
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  #mean absolute error %
        mse = dtf["fitting_error"].apply(lambda x: x**2).mean() # mean squared error
        rmse = np.sqrt(mse)  #root mean squared error
        ### testing
        prediction_error_mean = dtf["prediction_error"].mean()
        prediction_error_std = dtf["prediction_error"].std()
        prediction_mae = dtf["prediction_error"].apply(lambda x: np.abs(x)).mean()
        prediction_mape = dtf["prediction_error_pct"].apply(lambda x: np.abs(x)).mean()
        prediction_mse = dtf["prediction_error"].apply(lambda x: x**2).mean()
        prediction_rmse = np.sqrt(prediction_mse)
        
        ## intervals
        dtf["conf_int_low"] = dtf[col_preds] - 1.96*error_std
        dtf["conf_int_up"] = dtf[col_preds] + 1.96*error_std
        dtf["pred_int_low"] = dtf[col_preds] - 1.96*prediction_error_std
        dtf["pred_int_up"] = dtf[col_preds] + 1.96*prediction_error_std
        
        ## plot
        if plot==True:
            fig, ax = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(title, fontsize=20)      
            ### fitted values
            dtf[pd.notnull(dtf[col_fitted])][[col_ts,col_fitted]].plot(color=["black","green"], title="Fitted Values", grid=True, ax=ax[0,0])      
            ### test preds
            dtf[pd.isnull(dtf[col_fitted])][[col_ts,col_test,col_preds]].plot(color=["black","red","blue"], title="Predictions", grid=True, ax=ax[0,1])
            ax[0,1].fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax[0,1].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)     
            ### residuals
            dtf[["fitting_error","prediction_error"]].plot(ax=ax[1,0], color=["green","red"], title="Residuals", grid=True)
            ### residuals distribution
            dtf[["fitting_error","prediction_error"]].plot(ax=ax[1,1], color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            plt.show()
            print("error_mean:",np.round(prediction_error_mean), " | error_std:",np.round(prediction_error_std),
              " | mae:",np.round(prediction_mae), " | mape:",np.round(prediction_mape*100), "%  | mse:",np.round(prediction_mse), 
              " | rmse:",np.round(prediction_rmse))
        
        return dtf
    
    except Exception as e:
        print("--- got error ---")
        print(e)
    
    

###############################################################################
#                             ARIMA                                           #
###############################################################################
'''
Fits Holt-Winters Exponential Smoothing: 
    y_t+i = (level_t + i*trend_t) * seasonality_t
:parameter
    :param ts: pandas timeseries
    :param trend: str - "additive", "multiplicative"
    :param seasonal: str - "additive", "multiplicative"
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param alpha: num - the alpha value of the simple exponential smoothing (ex 0.94)
    :param pred_ahead: num - predictions ahead
:return
    dtf with predictons and the model
'''
def fit_expsmooth(ts, trend=None, seasonal=None, s=None, alpha=0.94, test=0.2, pred_ahead=5, figsize=(20,13)):
    ## checks
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal is None) & (s is None) else "Seasonal parameters: "+str(seasonal)+" Seasonality every "+str(s)+" observations"
    print(check_seasonality)
    
    ## split train/test
    ts_train, ts_test = utils_split_train_test(ts, test=test)
    
    ## train
    #alpha = alpha if s is None else 2/(s+1)
    model = smt.ExponentialSmoothing(ts_train, trend=trend, seasonal=seasonal, seasonal_periods=s).fit(smoothing_level=alpha)
    dtf = ts.to_frame(name="ts")
    dtf["fitted"] = model.fittedvalues
    
    ## test
    dtf["test"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1)
    
    ## forecast
    model = smt.ExponentialSmoothing(ts, trend=trend, seasonal=seasonal, seasonal_periods=s).fit(smoothing_level=alpha)
    preds = model.forecast(pred_ahead)
    preds = preds.to_frame(name="forecast")
    dtf = dtf.append(preds, sort=False)
    
    ## evaluate
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Holt-Winters alpha:"+str(alpha))
    return dtf, model



'''
Fits SARIMAX (Seasonal ARIMA with External Regressors):  
    yt+1 = (c + a0*yt + a1*yt-1 +...+ ap*yt-p) + (et + b1*et-1 + b2*et-2 +...+ bq*et-q) + (B*Xt)
:parameter
    :param ts: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d: degree of differencing (to remove trend), q: order of moving average (MA)
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param trend: str or None - "c" constant, "t" linear, "ct" both
    :param exog: pandas dataframe or numpy array
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
    :param pred_ahead: num - predictions ahead
    :param pred_exog: pandas dataframe or numpy array
    :param pred_confidence: num - confidence interval
    :param figsize: tuple - matplotlib figure
:return
    dtf with predictons and the model
'''
def fit_sarimax(ts, order=(1,0,1), trend=None, seasonal_order=(0,0,0,0), exog=None, test=0.2, pred_exog=None, pred_ahead=5, figsize=(20,13)):
    ## checks
    check_trend = "Trend parameters: No trend and No differencing" if (order[1] == 0) & (trend==None) else "Trend parameters: trend is "+str(trend)+" and d="+str(order[1])
    print(check_trend)
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal_order[3] == 0) & (np.sum(seasonal_order[0:3]) == 0) else "Seasonal parameters: Seasonality every "+str(seasonal_order[3])+" observations"
    print(check_seasonality)
    check_exog = "Exog parameters: Not given" if (exog is None) & (pred_exog is None) else "Exog parameters: number of regressors="+str(exog.shape[1])
    print(check_exog)
    
    ## split train/test
    if exog is None:
        ts_train, ts_test = utils_split_train_test(ts, exog=exog, test=test)
        exog_train, exog_test = None, None
    else:
        ts_train, ts_test, exog_train, exog_test = utils_split_train_test(ts, exog=exog, test=test)
    
    ## train
    model = smt.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit(disp=-1, trend=trend)
    dtf = ts.to_frame(name="ts")
    dtf["fitted"] = model.fittedvalues
    
    ## test
    dtf["test"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1, exog=exog_test)
    
    ## forecast
    model = smt.SARIMAX(ts, order=order, seasonal_order=seasonal_order, exog=exog, enforce_stationarity=False, enforce_invertibility=False).fit(disp=-1, trend=trend)
    preds = model.forecast(pred_ahead, exog=pred_exog)
    preds = preds.to_frame(name="forecast")
    dtf = dtf.append(preds, sort=False)
    
    ## evaluate
    title = "ARIMA "+str(order) if exog is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model


    
'''
Fits best Seasonal-ARIMAX.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    best model
'''
def find_best_sarimax(ts, seasonal=True, stationary=False, s=1, exog=None,
                      max_p=10, max_d=3, max_q=10,
                      max_P=10, max_D=3, max_Q=10):
    best_model = pmdarima.auto_arima(ts, exogenous=exog,
                                     seasonal=seasonal, stationary=stationary, m=s, 
                                     information_criterion='aic', max_order=20,
                                     max_p=max_p, max_d=max_d, max_q=max_q,
                                     max_P=max_P, max_D=max_D, max_Q=max_Q,
                                     error_action='ignore')
    print("best model --> (p, d, q):", best_model.order, " and  (P, D, Q, s):", best_model.seasonal_order)
    return best_model.summary()



'''
'''
def fit_varmax():
    smt.VARMAX()
    return dtf



###############################################################################
#                           PROPHET                                           #
###############################################################################
'''
Fits prophet on Business Data:
    y = trend + seasonality + holidays
:parameter
    :param ts: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param freq: str - "D" daily, "M" monthly, "Y" annual, "MS" monthly start ...
    
    :param growth: str - 'linear' or 'logistic' trend. "logistic" is for forecasting growth and needs capacity (ex. total market size, total population size)
    :param changepoints: list or None - dates at which to include potential changepoints
    :param n_changepoints: num - number of potential automatic changepoints to include
    
    :param yearly_seasonality: str or bool - "auto", True or False
    :param weekly_seasonality: str or bool - "auto", True or False
    :param daily_seasonality: str or bool - "auto", True or False
    :param seasonality_mode: str - 'additive' or 'multiplicative'
    
    :param holidays: pandas - DataFrame with columns 'ds' (dates) and 'holiday' (string ex 'xmas')

    :param lst_exog: list - names of variables
    :paam pred_exog: array - values of exog variables
:return
    dtf with predictons and the model
'''
def fit_prophet(ts, freq="D", figsize=(20,13),
                growth="linear", changepoints=None, n_changepoints=25,
                yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto", seasonality_mode='additive',
                holidays=None, lst_exog=None, pred_exog=None,
                test=0.2, preds_ahead=5):
    ## split train/test
    ts_train, ts_test = utils_split_train_test(ts, exog=None, test=test)
     
    ## prophet
    model = Prophet(growth, changepoints=changepoints, n_changepoints=n_changepoints,
                    yearly_seasonality=yearly_seasonality, weekly_seasonality=weekly_seasonality, daily_seasonality=daily_seasonality,seasonality_mode=seasonality_mode,
                    holidays=holidays)
    if lst_exog != None:
        for regressor in lst_exog:
            model.add_regressor(regressor)
    
    ## train
    model.fit(ts_train)
    
    ## test
    dtf_prophet = model.make_future_dataframe(periods=len(ts_test), freq=freq, include_history=True)
    
    if growth == "logistic":
        dtf_prophet["cap"] = ts_train["cap"].unique()[0]
    
    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(ts_train[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-preds_ahead:][lst_exog] = ts_test[lst_exog].values
    
    dtf_prophet = model.predict(dtf_prophet)
    
    
    ## predict
    dtf_prophet = model.make_future_dataframe(periods=preds_ahead, freq=freq, include_history=True)
    
    if growth == "logistic":
        dtf_prophet["cap"] = ts["cap"].unique()[0]
    
    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(ts[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-preds_ahead:][lst_exog] = pred_exog
    
    dtf_prophet = model.predict(dtf_prophet)
    
    ## plot prophet
    #fig = fbPlot.plot(model, dtf_prophet, figsize=figsize)
    #fbPlot.add_changepoints_to_plot(fig.gca(), model, dtf_prophet)
    
    
    
    return dtf_prophet[["ds", "yhat_lower", "yhat", "yhat_upper"]], model



'''
'''
def evaluate_prophet(model, years_train, years_fold, days_forecast, figsize=(20,13)):
    ## initial training
    initial =  str(years_train * 365)+" days"
    ## period to test
    period = str(years_fold * 365)+" days"
    ## horizon to forecast
    horizon = str(days_forecast)+" days"
    
    dtf_cv = diagnostics.cross_validation(model, initial=initial, period=period, horizon=horizon)
    fbPlot.plot(model, dtf_cv, figsize=figsize)
    kpi = diagnostics.performance_metrics(dtf_cv)
    return kpi



###############################################################################
#                            RNN                                              #
###############################################################################
'''
Preprocess a ts partitioning into X and y.
:parameter
    :param ts: pandas timeseries
    :param scaler: sklearn scaler object - if None is fitted
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    X, y, scaler
'''
def ts_preprocessing(ts, scaler=None, exog=None, s=20):
    ts = ts.sort_index(ascending=True).values
    
    ## scale
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    ts_preprocessed = scaler.fit_transform(ts.reshape(-1,1)).reshape(-1)        
    
    ## create X,y for train
    ts_preprocessed = kprocessing.sequence.TimeseriesGenerator(data=ts_preprocessed, targets=ts_preprocessed, length=s, batch_size=1)
    lst_X, lst_y = [], []
    for i in range(len(ts_preprocessed)):
        xi, yi = ts_preprocessed[i]
        lst_X.append(xi)
        lst_y.append(yi)
    X = np.array(lst_X)
    y = np.array(lst_y)
    #X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
    return X, y, scaler



'''
'''
def predict_lstm(ts, model, scaler, exog=None):
    s = model.input_shape[-1]
    lst_fitted = [np.nan]*s
    for i in range(len(ts)):
        end_ix = i + s
        if end_ix > len(ts)-1:
            break
        X = ts[i:end_ix]
        X = np.array(X)
        X = np.reshape(X, (1, 1, X.shape[0]))
        fit = model.predict(X)
        fit = scaler.inverse_transform(fit)[0][0]
        lst_fitted.append(fit)
    return np.array(lst_fitted)



'''
'''
def forecast_lstm(ts, model, scaler, exog=None, ahead=5):
    ## preprocess
    s = model.input_shape[-1]
    ts = ts.sort_index(ascending=True).values
    ts_preprocessed = list(scaler.fit_transform(ts.reshape(-1,1)))
         
    ## predict
    lst_preds = []
    for i in range(ahead):
        i += 1
        lst_X = ts_preprocessed[len(ts_preprocessed)-(s+1) : -1]
        X = np.array(lst_X)
        X = np.reshape(X, (1, 1, X.shape[0]))
        pred = model.predict(X)
        ts_preprocessed.append(pred)
        pred = scaler.inverse_transform(pred)[0][0]
        lst_preds.append(pred)
    return np.array(lst_preds)



'''
Fits LSTM neural network.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    generator, scaler 
'''
def fit_lstm(ts, exog=None, s=20, neurons=50, batch_size=1, epochs=100, test=0.2, ahead=5, freq="D", figsize=(20,13)):
    ## split train/test
    if exog is None:
        ts_train, ts_test = utils_split_train_test(ts, exog=exog, test=test)
        exog_train, exog_test = None, None
    else:
        ts_train, ts_test, exog_train, exog_test = utils_split_train_test(ts, exog=exog, test=test)
    
    ## preprocess train
    X_train, y_train, scaler = ts_preprocessing(ts_train, scaler=None, exog=exog_train, s=s)
       
    ## lstm
    model = models.Sequential()
    model.add( layers.LSTM(input_shape=X_train.shape[1:], units=neurons, activation='tanh', return_sequences=False) )
    model.add( layers.Dense(1) )
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    
    ## train
    print(model.summary())
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
    print("--- training ---")
    fig, ax = plt.subplots()
    ax.plot(training.history['loss'], label='loss')
    ax.grid(True)
    plt.xlabel('epoch')
    plt.legend()
    plt.show()
    
    ## create dtf with fitted values
    dtf = ts.to_frame(name="ts")
    fitted_values = predict_lstm(ts_train, training.model, scaler, exog_train)
    dtf["fitted"] = pd.DataFrame(data=fitted_values, index=ts_train.index)
    
    ## test
    test_data = ts_train[-s:].append(ts_test) 
    preds = predict_lstm(test_data, training.model, scaler, exog_test) 
    dtf["test"] = pd.DataFrame(data=preds[-len(ts_test):], index=ts_test.index)
    
    ## forecast
    ### refit final model with all data
    print("--- testing ---")
    X_train, y_train, scaler = ts_preprocessing(ts, scaler=None, exog=exog, s=s)
    final_model = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0)
    forecast = forecast_lstm(ts, final_model.model, scaler, exog, ahead)
    ### predictions ahead
    if freq is not None:
        start = ts.index[-1] + datetime.timedelta(days=1)
        end = start + datetime.timedelta(days=ahead-1)
        forecast_index = pd.date_range(start=start, end=end, freq=freq)
    else:
        forecast_index = np.arange(ts.index[-1]+1, ts.index[-1]+1+ahead)  
    dtf = dtf.append(pd.DataFrame(data=forecast, index=forecast_index, columns={"forecast"}))
    
    ## evaluate
    dtf = utils_evaluate_forecast(dtf[s:], figsize=figsize, title="LSTM with "+str(neurons)+" Neurons")
    return dtf, final_model.model


  