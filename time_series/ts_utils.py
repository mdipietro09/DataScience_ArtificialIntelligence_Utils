
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.tsa.api as smt
import statsmodels.api as sm
import pmdarima
import arch
from fbprophet import Prophet, diagnostics, plot as fbPlot
pd.plotting.register_matplotlib_converters()
from sklearn import preprocessing, svm
from tensorflow.keras import models, layers, preprocessing as kprocessing
import datetime



###############################################################################
#                         TS ANALYSIS                                         #
###############################################################################
'''
Plot ts with rolling mean and 95% confidence interval with rolling std.
:parameter
    :param ts: pandas Series
    :param window: num for rolling stats
:return
    dtf: input dataframe with 2 new columns
'''
def plot_ts(ts, plot_ma=True, plot_intervals=True, window=30, figsize=(15,5)):
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
Test stationarity by:
    - running Augmented Dickey-Fuller test wiht 95%
    - plotting mean and variance of a sample from data
    - plottig autocorrelation and partial autocoreelation
'''
def test_stationarity_acf_pacf(ts, sample=0.20, maxlag=30, figsize=(15,10)):
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
Defferenciate ts.
:parameter
    :param ts: pandas Series
    :param lag: num - diff[t] = y[t] - y[t-lag]
    :param order: num - how many times it has to differenciate: diff[t]^order = diff[t] - diff[t-lag] 
    :param drop_na: logic - if True Na are dropped, else are filled with last observation
'''
def diff_ts(ts, lag=1, order=1, drop_na=True):
    for i in range(order):
        ts = ts - ts.shift(lag)
    ts = ts[(pd.notnull(ts))] if drop_na is True else ts.fillna(method="bfill")
    return ts



'''
'''
def undo_diff(ts, first_y, lag=1, order=1):
    for i in range(order):
        (24168.04468 - 18256.02366) + a.cumsum()
        ts = np.r_[ts, ts[lag:]].cumsum()
    return ts



'''
Run Granger test on 2 series
'''
def test_2ts_casuality(ts1, ts2, maxlag=30, figsize=(15,5)):
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
Decompose ts into
    - trend component = moving avarage
    - seasonality
    - residuals = y - (trend + seasonality)
:parameter
    :param s: num - number of observations per seasonality (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
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



'''
Find outliers using sklearn unsupervised support vetcor machine.
:parameter
    :param ts: pandas Series
    :param perc: float - percentage of outliers to look for
:return
    dtf with raw ts, outlier 1/0 (yes/no), numeric index
'''
def find_outliers(ts, perc=0.01, figsize=(15,5)):
    ## fit svm
    scaler = preprocessing.StandardScaler()
    ts_scaled = scaler.fit_transform(ts.values.reshape(-1,1))
    model = svm.OneClassSVM(nu=perc, kernel="rbf", gamma=0.01)
    model.fit(ts_scaled)
    ## dtf output
    dtf_outliers = ts.to_frame(name="ts")
    dtf_outliers["index"] = range(len(ts))
    dtf_outliers["outlier"] = model.predict(ts_scaled)
    dtf_outliers["outlier"] = dtf_outliers["outlier"].apply(lambda x: 1 if x==-1 else 0)
    ## plot
    fig, ax = plt.subplots(figsize=figsize)
    ax.set(title="Outliers detection: found "+str(sum(dtf_outliers["outlier"]==1)))
    ax.plot(dtf_outliers["index"], dtf_outliers["ts"], color="black")
    ax.scatter(x=dtf_outliers[dtf_outliers["outlier"]==1]["index"], y=dtf_outliers[dtf_outliers["outlier"]==1]['ts'], color='red')
    ax.grid(True)
    plt.show()
    return dtf_outliers



'''
Interpolate outliers in a ts.
'''
def remove_outliers(ts, outliers_idx, figsize=(15,5)):
    ts_clean = ts.copy()
    ts_clean.loc[outliers_idx] = np.nan
    ts_clean = ts_clean.interpolate(method="linear")
    ax = ts.plot(figsize=figsize, color="red", alpha=0.5, title="Remove outliers", label="original", legend=True)
    ts_clean.plot(ax=ax, grid=True, color="black", label="interpolated", legend=True)
    plt.show()
    return ts_clean



###############################################################################
#                 MODEL DESIGN & TESTING - FORECASTING                        #
###############################################################################
'''
Split train/test from .
:parameter
    :param ts: pandas Series
    :param exog: array len(ts) x n regressors
    :param test: num or str - test size (ex. 0.20) or index position (ex. "yyyy-mm-dd", 1000)
:return
    dtf: input dataframe with 2 new columns
'''
def split_train_test(ts, exog=None, test=0.20, plot=True, figsize=(15,5)):
    ## define splitting point
    if type(test) is float:
        split = int(len(ts)*(1-test))
        perc = test
    elif type(test) is str:
        split = ts.reset_index()[ts.reset_index().iloc[:,0]==test].index[0]
        perc = round(len(ts[split:])/len(ts), 2)
    else:
        split = test
        perc = round(len(ts[split:])/len(ts), 2)
    print("--- splitting at index: ", split, "|", ts.index[split], "| test size:", perc, " ---")
    
    ## split ts
    ts_train = ts.head(split)
    ts_test = ts.tail(len(ts)-split)
    if plot is True:
        fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=figsize)
        ts_train.plot(ax=ax[0], grid=True, title="Train", color="black")
        ts_test.plot(ax=ax[1], grid=True, title="Test", color="black")
        ax[0].set(xlabel=None)
        ax[1].set(xlabel=None)
        plt.show()
        
    ## split exog
    if exog is not None:
        exog_train = exog[0:split] 
        exog_test = exog[split:]
        return ts_train, ts_test, exog_train, exog_test
    else:
        return ts_train, ts_test
    


'''
Evaluation metrics for predictions.
:parameter
    :param dtf: DataFrame with columns raw values, fitted training values, predicted test values
:return
    dataframe with raw ts and forecast
'''
def utils_evaluate_forecast(dtf, title, plot=True, figsize=(20,13)):
    try:
        ## residuals
        dtf["residuals"] = dtf["ts"] - dtf["model"]
        dtf["error"] = dtf["ts"] - dtf["forecast"]
        dtf["error_pct"] = dtf["error"] / dtf["ts"]
        
        ## kpi
        residuals_mean = dtf["residuals"].mean()  #errore medio nel training
        residuals_std = dtf["residuals"].std()    #standard dev dell'errore nel training
        error_mean = dtf["error"].mean()   #errore medio nel test
        error_std = dtf["error"].std()     #standard dev dell'errore nel test
        mae = dtf["error"].apply(lambda x: np.abs(x)).mean()  #mean absolute error
        mape = dtf["error_pct"].apply(lambda x: np.abs(x)).mean()  #mean absolute error %
        mse = dtf["error"].apply(lambda x: x**2).mean() # mean squared error
        rmse = np.sqrt(mse)  #root mean squared error
        
        ## intervals
        dtf["conf_int_low"] = dtf["forecast"] - 1.96*residuals_std
        dtf["conf_int_up"] = dtf["forecast"] + 1.96*residuals_std
        dtf["pred_int_low"] = dtf["forecast"] - 1.96*error_std
        dtf["pred_int_up"] = dtf["forecast"] + 1.96*error_std
        
        ## plot
        if plot==True:
            fig = plt.figure(figsize=figsize)
            fig.suptitle(title, fontsize=20)   
            ax1 = fig.add_subplot(2,2, 1)
            ax2 = fig.add_subplot(2,2, 2, sharey=ax1)
            ax3 = fig.add_subplot(2,2, 3)
            ax4 = fig.add_subplot(2,2, 4)
            ### training
            dtf[pd.notnull(dtf["model"])][["ts","model"]].plot(color=["black","green"], title="Model", grid=True, ax=ax1)      
            ax1.set(xlabel=None)
            ### test
            dtf[pd.isnull(dtf["model"])][["ts","forecast"]].plot(color=["black","red"], title="Forecast", grid=True, ax=ax2)
            ax2.fill_between(x=dtf.index, y1=dtf['pred_int_low'], y2=dtf['pred_int_up'], color='b', alpha=0.2)
            ax2.fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3)     
            ax2.set(xlabel=None)
            ### residuals
            dtf[["residuals","error"]].plot(ax=ax3, color=["green","red"], title="Residuals", grid=True)
            ax3.set(xlabel=None)
            ### residuals distribution
            dtf[["residuals","error"]].plot(ax=ax4, color=["green","red"], kind='kde', title="Residuals Distribution", grid=True)
            ax4.set(ylabel=None)
            plt.show()
            print("Training --> Residuals mean:", np.round(residuals_mean), " | std:", np.round(residuals_std))
            print("Test --> Error mean:", np.round(error_mean), " | std:", np.round(error_std),
                  " | mae:",np.round(mae), " | mape:",np.round(mape*100), "%  | mse:",np.round(mse), " | rmse:",np.round(rmse))
        
        return dtf[["ts","model","residuals","conf_int_low","conf_int_up", 
                    "forecast","error","pred_int_low","pred_int_up"]]
    
    except Exception as e:
        print("--- got error ---")
        print(e)
    


'''
Generate dates to index predictions.
:parameter
    :param start: str - "yyyy-mm-dd"
    :param end: str - "yyyy-mm-dd"
    :param n: num - length of index
    :param freq: None or str - 'B' business day, 'D' daily, 'W' weekly, 'M' monthly, 'A' annual, 'Q' quarterly
'''
def utils_generate_indexdate(start, end=None, n=None, freq="D"):
    if end is not None:
        index = pd.date_range(start=start, end=end, freq=freq)
    else:
        index = pd.date_range(start=start, periods=n, freq=freq)
    index = index[1:]
    print("--- generating index date --> start:", index[0], "| end:", index[-1], "| len:", len(index), "---")
    return index



'''
Plot unknown future forecast.
'''
def utils_plot_forecast(dtf, zoom=30, figsize=(15,5)):
    ## interval
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    dtf["conf_int_low"] = dtf["forecast"] - 1.96*dtf["residuals"].std()
    dtf["conf_int_up"] = dtf["forecast"] + 1.96*dtf["residuals"].std()
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    
    ## entire series
    dtf[["ts","forecast"]].plot(color=["black","red"], grid=True, ax=ax[0], title="History + Future")
    ax[0].fill_between(x=dtf.index, y1=dtf['conf_int_low'], y2=dtf['conf_int_up'], color='b', alpha=0.3) 
          
    ## focus on last
    first_idx = dtf[pd.notnull(dtf["forecast"])].index[0]
    first_loc = dtf.index.tolist().index(first_idx)
    zoom_idx = dtf.index[first_loc-zoom]
    dtf.loc[zoom_idx:][["ts","forecast"]].plot(color=["black","red"], grid=True, ax=ax[1], title="Zoom on the last "+str(zoom)+" observations")
    ax[1].fill_between(x=dtf.loc[zoom_idx:].index, y1=dtf.loc[zoom_idx:]['conf_int_low'], 
                       y2=dtf.loc[zoom_idx:]['conf_int_up'], color='b', alpha=0.3)
    plt.show()
    return dtf[["ts","model","residuals","conf_int_low","forecast","conf_int_up"]]



###############################################################################
#                           RANDOM WALK                                       #
###############################################################################
'''
Generate a Random Walk process.
:parameter
    :param y0: num - starting value
    :param n: num - length of process
    :param ymin: num - limit
    :param ymax: num - limit
'''
def utils_generate_rw(y0, n, sigma, ymin=None, ymax=None):
    rw = [y0]
    for t in range(1, n):
        yt = rw[t-1] + np.random.normal(0,sigma)
        if (ymax is not None) and (yt > ymax):
            yt = rw[t-1] - abs(np.random.normal(0,sigma))
        elif (ymin is not None) and (yt < ymin):
            yt = rw[t-1] + abs(np.random.normal(0,sigma))
        rw.append(yt)
    return rw
        

 
'''
Simulate Random Walk from params of a given ts: 
    y[t+1] = y[t] + wn~(0,σ)
'''
def simulate_rw(ts_train, ts_test, figsize=(15,10)):
    ## simulate train
    diff_ts = ts_train - ts_train.shift(1)
    rw = utils_generate_rw(y0=ts_train[0], n=len(ts_train), sigma=diff_ts.std(), ymin=ts_train.min(), ymax=ts_train.max())
    dtf_train = ts_train.to_frame(name="ts").merge(pd.DataFrame(rw, index=ts_train.index, columns=["model"]), how='left', left_index=True, right_index=True)
    
    ## test
    rw = utils_generate_rw(y0=ts_train[-1], n=len(ts_test), sigma=diff_ts.std(), ymin=ts_train.min(), ymax=ts_train.max())
    dtf_test = ts_test.to_frame(name="ts").merge(pd.DataFrame(rw, index=ts_test.index, columns=["forecast"]), 
                                                 how='left', left_index=True, right_index=True)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Random Walk Simulation")
    return dtf



'''
Forecast unknown future.
'''
def forecast_rw(ts, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    diff_ts = ts - ts.shift(1)
    sigma = diff_ts.std()
    rw = utils_generate_rw(y0=ts[0], n=len(ts), sigma=sigma, ymin=ts.min(), ymax=ts.max())
    dtf = ts.to_frame(name="ts").merge(pd.DataFrame(rw, index=ts.index, columns=["model"]), 
                                       how='left', left_index=True, right_index=True)
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = utils_generate_rw(y0=ts[-1], n=len(index), sigma=sigma, ymin=ts.min(), ymax=ts.max())
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf
    


###############################################################################
#                        AUTOREGRESSIVE                                       #
###############################################################################
'''
Fits Holt-Winters Exponential Smoothing: 
    y[t+i] = (level[t] + i*trend[t]) * seasonality[t]
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param trend: str - "additive" (linear), "multiplicative" (non-linear)
    :param seasonal: str - "additive" (ex. +100 every 7 days), "multiplicative" (ex. x10 every 7 days)
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param alpha: num - the alpha value of the simple exponential smoothing (ex 0.94)
:return
    dtf with predictons and the model
'''
def fit_expsmooth(ts_train, ts_test, trend="additive", seasonal="multiplicative", s=None, alpha=0.94, figsize=(15,10)):
    ## checks
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal is None) & (s is None) else "Seasonal parameters: "+str(seasonal)+" Seasonality every "+str(s)+" observations"
    print(check_seasonality)
    
    ## train
    #alpha = alpha if s is None else 2/(s+1)
    model = smt.ExponentialSmoothing(ts_train, trend=trend, seasonal=seasonal, seasonal_periods=s).fit(smoothing_level=alpha)
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Holt-Winters ("+str(alpha)+")")
    return dtf, model



'''
Fits SARIMAX (Seasonal ARIMA with External Regressors):  
    y[t+1] = (c + a0*y[t] + a1*y[t-1] +...+ ap*y[t-p]) + (e[t] + b1*e[t-1] + b2*e[t-2] +...+ bq*e[t-q]) + (B*X[t])
:parameter
    :param ts_train: pandas timeseries
    :param ts_test: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p: lag order (AR), d: degree of differencing (to remove trend), q: order of moving average (MA)
    :param seasonal_order: tuple - (P,D,Q,s) --> s: number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param exog_train: pandas dataframe or numpy array
    :param exog_test: pandas dataframe or numpy array
:return
    dtf with predictons and the model
'''
def fit_sarimax(ts_train, ts_test, order=(1,0,1), seasonal_order=(0,0,0,0), exog_train=None, exog_test=None, figsize=(15,10)):
    ## checks
    check_trend = "Trend parameters: No differencing" if order[1] == 0 else "Trend parameters: d="+str(order[1])
    print(check_trend)
    check_seasonality = "Seasonal parameters: No Seasonality" if (seasonal_order[3] == 0) & (np.sum(seasonal_order[0:3]) == 0) else "Seasonal parameters: Seasonality every "+str(seasonal_order[3])+" observations"
    print(check_seasonality)
    check_exog = "Exog parameters: Not given" if (exog_train is None) & (exog_test is None) else "Exog parameters: number of regressors="+str(exog_train.shape[1])
    print(check_exog)
    
    ## train
    model = smt.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit()
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.fittedvalues
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.predict(start=len(ts_train), end=len(ts_train)+len(ts_test)-1, exog=exog_test)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "ARIMA "+str(order) if exog_train is None else "ARIMAX "+str(order)
    title = "S"+title+" x "+str(seasonal_order) if np.sum(seasonal_order) > 0 else title
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model


    
'''
Find best Seasonal-ARIMAX parameters.
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
Fits GARCH (Generalized Autoregressive Conditional Heteroskedasticity):  
    y[t+1] = m + e[t+1]
    e[t+1] = σ[t+1] * wn~(0,1)
    σ²[t+1] = c + (a0*σ²[t] + a1*σ²[t-1] +...+ ap*σ²[t-p]) + (b0*e²[t] + b1*e[t-1] + b2*e²[t-2] +...+ bq*e²[t-q])
:parameter
    :param ts: pandas timeseries
    :param order: tuple - ARIMA(p,d,q) --> p:lag order (AR), d:degree of differencing (to remove trend), q:order of moving average (MA)
'''
def fit_garch(ts_train, ts_test, order=(1,0,1), seasonal_order=(0,0,0,0), exog_train=None, exog_test=None, figsize=(15,10)):
    ## train
    arima = smt.SARIMAX(ts_train, order=order, seasonal_order=seasonal_order, exog=exog_train, enforce_stationarity=False, enforce_invertibility=False).fit()
    garch = arch.arch_model(arima.resid, p=order[0], o=order[1], q=order[2], x=exog_train, dist='StudentsT', power=2.0, mean='Constant', vol='GARCH')
    model = garch.fit(update_freq=s)
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = model.conditional_volatility
    
    ## test
    dtf_test = ts_test.to_frame(name="ts")
    dtf_test["forecast"] = model.forecast(horizon=len(ts_test))

    ## evaluate
    dtf = dtf_train.append(dtf_test)
    title = "GARCH ("+str(order[0])+","+str(order[2])+")" if order[0] != 0 else "ARCH ("+str(order[2])+")"
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title=title)
    return dtf, model



'''
Forecast unknown future.
'''
def forecast_arima(ts, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    model = model.fit()
    dtf = ts.to_frame(name="ts")
    dtf["model"] = model.fittedvalues
    dtf["residuals"] = dtf["ts"] - dtf["model"]
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = model.forecast(len(index))
    dtf = dtf.append(preds.to_frame(name="forecast"))
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf



###############################################################################
#                            RNN                                              #
###############################################################################
'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metric = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)][0]
    fig, ax = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(15,3))
    ax[0].set(title="Training")
    ax11 = ax[0].twinx()
    ax[0].plot(training.history['loss'], color='blue')
    ax11.plot(training.history[metric], color='green')
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss', color='blue')
    ax11.set_ylabel(metric.capitalize(), color='green')
    ax[1].set(title="Validation")
    ax22 = ax[1].twinx()
    ax[1].plot(training.history['val_loss'], color='blue')
    ax22.plot(training.history['val_'+metric], color='green')
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Loss', color='blue')
    ax22.set_ylabel(metric.capitalize(), color='green')
    plt.show()
    
    
    
'''
Preprocess a ts partitioning into X and y.
:parameter
    :param ts: pandas timeseries
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
    :param scaler: sklearn scaler object - if None is fitted
    :param exog: pandas dataframe or numpy array
:return
    X, y, scaler
'''
def utils_preprocess_ts(ts, s, scaler=None, exog=None):
    ## scale
    if scaler is None:
        scaler = preprocessing.MinMaxScaler(feature_range=(0,1))
    ts_preprocessed = scaler.fit_transform(ts.values.reshape(-1,1)).reshape(-1)        
    
    ## create X,y for train
    ts_preprocessed = kprocessing.sequence.TimeseriesGenerator(data=ts_preprocessed, 
                                                               targets=ts_preprocessed, 
                                                               length=s, batch_size=1)
    lst_X, lst_y = [], []
    for i in range(len(ts_preprocessed)):
        xi, yi = ts_preprocessed[i]
        lst_X.append(xi)
        lst_y.append(yi)
    X = np.array(lst_X)
    y = np.array(lst_y)
    return X, y, scaler



'''
Get fitted values.
'''
def utils_fitted_lstm(ts, model, scaler, exog=None):
    ## scale
    ts_preprocessed = scaler.fit_transform(ts.values.reshape(-1,1)).reshape(-1) 
    
    ## create Xy, predict = fitted
    s = model.input_shape[-1]
    lst_fitted = [np.nan]*s
    for i in range(len(ts_preprocessed)):
        end_ix = i + s
        if end_ix > len(ts_preprocessed)-1:
            break
        X = ts_preprocessed[i:end_ix]
        X = np.array(X)
        X = np.reshape(X, (1,1,X.shape[0]))
        fit = model.predict(X)
        fit = scaler.inverse_transform(fit)[0][0]
        lst_fitted.append(fit)
    return np.array(lst_fitted)



'''
Predict ts using previous predictions.
'''
def utils_predict_lstm(ts, model, scaler, pred_ahead, exog=None):
    ## scale
    s = model.input_shape[-1]
    ts_preprocessed = list(scaler.fit_transform(ts[-s:].values.reshape(-1,1))) 
    
    ## predict, append, re-predict
    lst_preds = []
    for i in range(pred_ahead):
        X = np.array(ts_preprocessed[len(ts_preprocessed)-s:])
        X = np.reshape(X, (1,1,X.shape[0]))
        pred = model.predict(X)
        ts_preprocessed.append(pred)
        pred = scaler.inverse_transform(pred)[0][0]
        lst_preds.append(pred)
    return np.array(lst_preds)



'''
Fit Long short-term memory neural network.
:parameter
    :param ts: pandas timeseries
    :param exog: pandas dataframe or numpy array
    :param s: num - number of observations per seasonal (ex. 7 for weekly seasonality with daily data, 12 for yearly seasonality with monthly data)
:return
    generator, scaler 
'''
def fit_lstm(ts_train, ts_test, model, exog=None, s=20, figsize=(15,5)):
    ## check
    print("Seasonality: using the last", s, "observations to predict the next 1")
    
    ## preprocess train
    X_train, y_train, scaler = utils_preprocess_ts(ts_train, scaler=None, exog=exog, s=s)
    
    ## lstm
    if model is None:
        model = models.Sequential()
        model.add( layers.LSTM(input_shape=X_train.shape[1:], units=50, activation='relu', return_sequences=False) )
        model.add( layers.Dense(1) )
        model.compile(optimizer='adam', loss='mean_absolute_error')
    
    ## train
    print(model.summary())
    training = model.fit(x=X_train, y=y_train, batch_size=1, epochs=100, shuffle=True, verbose=0, validation_split=0.3)
    utils_plot_keras_training(training)
    
    dtf_train = ts_train.to_frame(name="ts")
    dtf_train["model"] = utils_fitted_lstm(ts_train, training.model, scaler, exog)
    dtf_train["model"] = dtf_train["model"].fillna(method='bfill')
    
    ## test
    preds = utils_predict_lstm(ts_train[-s:], training.model, scaler, pred_ahead=len(ts_test), exog=None)
    dtf_test = ts_test.to_frame(name="ts").merge(pd.DataFrame(data=preds, index=ts_test.index, columns=["forecast"]),
                                                 how='left', left_index=True, right_index=True)
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="LSTM (memory:"+str(s)+")")
    return dtf, training.model



'''
Forecast unknown future.
'''
def forecast_lstm(ts, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    s = model.input_shape[-1]
    X, y, scaler = utils_preprocess_ts(ts, scaler=None, exog=None, s=s)
    training = model.fit(x=X, y=y, batch_size=1, epochs=100, shuffle=True, verbose=0, validation_split=0.3)
    dtf = ts.to_frame(name="ts")
    dtf["model"] = utils_fitted_lstm(ts, training.model, scaler, None)
    dtf["model"] = dtf["model"].fillna(method='bfill')
    
    ## index
    index = utils_generate_indexdate(start=ts.index[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    preds = utils_predict_lstm(ts[-s:], training.model, scaler, pred_ahead=len(index), exog=None)
    dtf = dtf.append(pd.DataFrame(data=preds, index=index, columns=["forecast"]))
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf



###############################################################################
#                           PROPHET                                           #
###############################################################################
'''
Fits prophet on Business Data:
    y = trend + seasonality + holidays
:parameter
    :param dtf_train: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param dtf_test: pandas Dataframe with columns 'ds' (dates), 'y' (values), 'cap' (capacity if growth="logistic"), other additional regressor
    :param lst_exog: list - names of variables
    :param freq: str - "D" daily, "M" monthly, "Y" annual, "MS" monthly start ...
:return
    dtf with predictons and the model
'''
def fit_prophet(dtf_train, dtf_test, lst_exog=None, model=None, freq="D", figsize=(15,10)):
    ## setup prophet
    if model is None:
        model = Prophet(growth="linear", changepoints=None, n_changepoints=25, seasonality_mode="multiplicative",
                yearly_seasonality="auto", weekly_seasonality="auto", daily_seasonality="auto",
                holidays=None)
    if lst_exog != None:
        for regressor in lst_exog:
            model.add_regressor(regressor)
    
    ## train
    model.fit(dtf_train)
    
    ## test
    dtf_prophet = model.make_future_dataframe(periods=len(dtf_test), freq=freq, include_history=True)
    
    if model.growth == "logistic":
        dtf_prophet["cap"] = dtf_train["cap"].unique()[0]
    
    if lst_exog != None:
        dtf_prophet = dtf_prophet.merge(dtf_train[["ds"]+lst_exog], how="left")
        dtf_prophet.iloc[-len(dtf_test):][lst_exog] = dtf_test[lst_exog].values
    
    dtf_prophet = model.predict(dtf_prophet)
    dtf_train = dtf_train.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    dtf_test = dtf_test.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'forecast', 'y':'ts'}).set_index("ds")
    
    ## evaluate
    dtf = dtf_train.append(dtf_test)
    dtf = utils_evaluate_forecast(dtf, figsize=figsize, title="Prophet")
    return dtf, model
    


'''
Forecast unknown future.
'''
def forecast_prophet(dtf, model, pred_ahead=None, end=None, freq="D", zoom=30, figsize=(15,5)):
    ## fit
    model.fit(dtf)
    
    ## index
    index = utils_generate_indexdate(start=dtf["ds"].values[-1], end=end, n=pred_ahead, freq=freq)
    
    ## forecast
    dtf_prophet = model.make_future_dataframe(periods=len(index), freq=freq, include_history=True)
    dtf_prophet = model.predict(dtf_prophet)
    dtf = dtf.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'model', 'y':'ts'}).set_index("ds")
    preds = pd.DataFrame(data=index, columns=["ds"])
    preds = preds.merge(dtf_prophet[["ds","yhat"]], how="left").rename(columns={'yhat':'forecast'}).set_index("ds")
    dtf = dtf.append(preds)
    
    ## plot
    dtf = utils_plot_forecast(dtf, zoom=zoom)
    return dtf