
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, discriminant_analysis, cluster
import itertools
from lime import lime_tabular



###############################################################################
#                       DATA ANALYSIS                                         #
###############################################################################
'''
Counts Nas for every column of a dataframe.
:parameter
    :param dtf: dataframe - input data
    :param plot: str or None - "freq", "map" 
    :param top: num - plot setting
    :param fontsize: num - plot setting
'''
def check_Nas(dtf, plot="freq", top=20, fontsize=10):
    try:
        ## print
        len_dtf = len(dtf)
        print("len dtf: "+str(len_dtf))
        for col in dtf.columns:
            print(col+" --> Nas: "+str(dtf[col].isna().sum())+" ("+str(np.round(dtf[col].isna().mean(), 3)*100)+"%)")
            if dtf[col].nunique() == len_dtf:
                print("    # possible pk")
                
        ## plot
        if plot == "freq":
            ax = dtf.isna().sum().head(top).sort_values().plot(kind="barh")
            totals= []
            for i in ax.patches:
                totals.append(i.get_width())
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=fontsize, color='black')
            plt.title("NAs count")
            plt.show()
        
        elif plot == "map":
            sns.heatmap(dtf.isnull(), cbar=False).set_title('Missings Map')
        
    except Exception as e:
        print("--- got error ---")
        print(e)
        
        

'''
Plots the frequency distribution of a dtf column.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column name
    :param max_cat: num - max number of uniques to consider a numeric variable categorical
    :param top: num - plot setting
    :param show_perc: logic - plot setting
    :param fontsize: num - plot setting
    :param bins: num - plot setting
    :param quantile_breaks: tuple - plot distribution between these quantiles (to exclude outilers)
    :param figsize: tuple - plot settings
'''
def freqdist_plot(dtf, x, max_cat=20, top=20, show_perc=False, fontsize=10, bins=100, quantile_breaks=(0,10), figsize=(10,10)):
    try:
        ## categorical
        if (dtf[x].dtype == "O") | (dtf[x].nunique() < max_cat):        
            ax = dtf[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals= []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=fontsize, color='black')
            else:
                total= sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=fontsize, color='black')
            plt.suptitle(x, fontsize=20)
            plt.show()
            
        ## numeric
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            ### distribution
            ax[0].title.set_text('distribution')
            variable = dtf[x].fillna(dtf[x].mean())
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[ (variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]]) ]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
            ### boxplot 
            ax[1].title.set_text('outliers')
            tmp_dtf = pd.DataFrame(dtf[x])
            tmp_dtf[x] = np.log(tmp_dtf[x])
            tmp_dtf.boxplot(column=x, ax=ax[1])
            plt.show()   
        
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Plots a bivariate analysis.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column
    :param y: str - column
    :param analysis_type: str - "timeseries", "Nas", "distribution"
    :param max_cat: num - max number of uniques to consider a numeric variable categorical
'''
def bivariate_plot(dtf, x, y, analysis_type="distribution", max_cat=20):
    
    def utils_recognize_type(dtf, col, max_cat):
        if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
            return "cat"
        else:
            return "num"
    
    try:
        if analysis_type == "distribution":
            ## numeric vs numeric
            if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
                sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='scatter')
            
            ## numeric vs categorical 
            elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "num"):
                sns.catplot(x=x, y=y, data=dtf, kind="box")    
            elif (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
                sns.catplot(x=y, y=x, data=dtf, kind="box")

            ## categorical vs categorical
            elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):  
                sns.catplot(x=x, hue=y, data=dtf, kind='count')
        
        ## timeserie
        elif analysis_type == "timeserie":
            sns.lineplot(x=x, y=y, data= dtf, linewidth=1)
        
        ## Nas
        #elif analysis_type == "Nas":
        
        else:
            print('choose one analysis type: "timeseries", "Nas", "distribution"')
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Plots the correlation matrix with seaborn.
:parameter
    :param dtf: dataframe - input data
    :param method: str - "pearson", "spearman" ...
    :param annotation: logic - plot setting
    :param figsize: tuple - plot setting
'''
def corrmatrix_plot(dtf, method="pearson", annotation=True, figsize=(10,10)):    
    corr_matrix = dtf.corr(method= method)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, annot=annotation, cmap="YlGnBu", ax=ax)
    plt.title(method + " correlation")
        


'''
Checks the primary key of a table and saves it as csv.
:parameter
    :param dtf: dataframe - input data
    :param pk: str - column name
    :param save: logic - want to save the file?
    :param path: str - dirpath
    :param dtf_name: str - csv name
:return
    the duplicated keys (in case pk is not unique)
'''
def CheckAndSave(dtf, pk, save=False, path=None, dtf_name=None):
    try:
        if len(dtf.index) == dtf[pk].nunique():
            print("Rows:", len(dtf.index), " = ", "Pk:", dtf[pk].nunique(), "--> OK")
            if save == True:
                dtf.to_csv(path_or_buf= path+dtf_name+".csv", sep=',', decimal=".", header=True, index=False)
                print("saved.")
        else:
            print("Rows:", len(dtf.index), " != ", "Pk:", dtf[pk].nunique(), "--> SHIT")
            ERROR = dtf.groupby(pk).size().reset_index(name= "count").sort_values(by="count", ascending= False)
            print("Prendo ad esempio: ", pk+"==", ERROR.iloc[0,0])
            return dtf[ dtf[pk]==ERROR.iloc[0,0] ]
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        
        

'''
Moves columns into a dtf.
:parameter
    :param dtf: dataframe - input data
    :param lst_cols: list - names of the columns that must be moved
    :param where: str - "front" or "end"
:return
    dtf with moved columns
'''
def pop_columns(dtf, lst_cols, where="front"):
    current_cols = dtf.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        dtf = dtf[lst_cols + current_cols]
    elif where == "end":
        dtf = dtf[current_cols + lst_cols]
    else:
        print('choose where "front" or "end"')
    return dtf



###############################################################################
#                FEATURES ENGINEERING & SELECTION                             #
###############################################################################
'''
Transforms a categorical column into dummy columns
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param x: str - column name
    :param dropx: logic - whether the x column should be dropped
    :param dummy_na: logic - remove Nas or treat as category
:return
    dtf with dummy columns added
'''
def add_dummies(dtf, x, dropx=False, dummy_na=False):
    try:
        dtf_dummy = pd.get_dummies(dtf[x], prefix=x, drop_first=True, dummy_na=dummy_na)
        dtf = pd.concat([dtf, dtf_dummy], axis=1)
        if dropx == True:
            dtf = dtf.drop(x, axis=1)
        return dtf
    
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Rebalances a dataset.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable 
    :param balance: str or None - "up", "down"
    :param replace: logic - resampling with replacement
    :param size: num - 1 for same size of the other class, 0.5 for half of the other class
:return
    rebalanced dtf
'''
def rebalance(dtf, y, balance=None,  replace=True, size=1):
    try:
        ## check
        check = dtf[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        major = check.index[0]
        minor = check.index[1]
        dtf_major = dtf[dtf[y]==major]
        dtf_minor = dtf[dtf[y]==minor]
        
        ## up-sampling
        if balance == "up":
            dtf_minor = utils.resample(dtf_minor, replace=replace, random_state=123,
                                       n_samples=int(round(size*len(dtf_major), 0)) )
            dtf_balanced = pd.concat([dtf_major, dtf_minor])
        ## down-sampling
        elif balance == "down":
            dtf_major = utils.resample(dtf_major, replace=replace, random_state=123,
                                       n_samples=int(round(size*len(dtf_minor), 0)) )
            dtf_balanced = pd.concat([dtf_major, dtf_minor])
        else:
            print("select up or down resampling")
            return dtf
        
        print("")
        check = dtf_balanced[y].value_counts().to_frame()
        check["%"] = (check[y] / check[y].sum() *100).round(1).astype(str) + '%'
        print(check)
        print("tot:", check[y].sum())
        return dtf_balanced
    
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Performs features selections: by correlation (keeping the lowest p-value) and by lasso.
:prameter
    :param dtf: dataframe - feature matrix dtf
    :param pk: str - name of the primary key
    :param y: str - name of the dependent variable
    :param corr_threshold: num or None- threshold to considere features high correlated (if None skip the step)
    :param lasso_alpha: num or None - Constant that multiplies the L1 term (if None skip the step)
:return
    dic with lists of features to keep.
'''     
def features_selection(dtf, pk, y, corr_threshold=0.7, lasso_alpha=0.5):
    try:
        ## correlation and p-value
        if corr_threshold is not None:
            ### compute
            corr_matrix = dtf.drop([pk, y], axis=1).corr(method="spearman")
            dtf_stats = pd.DataFrame(columns=['variable', 'p_value'])
            for col in dtf.drop([pk, y], axis=1).columns.to_list():
                slope, intercept, corr_value, p_value, std_err = scipy.stats.linregress(x=dtf[col].values, y=dtf[y].values)
                dtf_stats = dtf_stats.append([{"variable":col, 'p_value':p_value}], ignore_index=False)     
            ### selection
            stats_selected_features = []
            for col in corr_matrix.index.unique():
                dtf_filtered = corr_matrix.drop(col, axis=0)[col].to_frame()
                dtf_filtered = dtf_filtered[(dtf_filtered[col] >= corr_threshold) | (dtf_filtered[col] <= -corr_threshold)]
                if len(dtf_filtered) > 0:
                    for i in range(0, len(dtf_filtered)):
                        varA = col
                        varB = dtf_filtered.index[0]
                        corrAB = dtf_filtered.iloc[i,0]
                        pvalueA = dtf_stats[dtf_stats["variable"]==varA]["p_value"].values[0]
                        pvalueB = dtf_stats[dtf_stats["variable"]==varB]["p_value"].values[0]
                        if pvalueA < pvalueB:
                            stats_selected_features.append(varA)
                        else:
                            stats_selected_features.append(varB)
                else:
                    next
            stats_selected_features = list(set(stats_selected_features))
            print("stats: features from", len(corr_matrix.columns.to_list()), "--> to", len(stats_selected_features))
        else:
            stats_selected_features = 0
                        
        ## lasso regression
        if lasso_alpha is not None:
            ### compute
            dtf_X = dtf.drop([pk, y], axis=1)
            lasso_selection = feature_selection.SelectFromModel( linear_model.Lasso(alpha=lasso_alpha, normalize=False, random_state=123) )
            lasso_selection.fit(X=dtf_X, y=dtf[y])
            ### selection
            lasso_selected_features = dtf_X.columns[ (lasso_selection.estimator_.coef_ != 0).ravel().tolist() ]
            lasso_selected_features = list(set(lasso_selected_features))
            print("lasso: features from", len(dtf_X.columns.to_list()), "--> to", len(lasso_selected_features))
        else:
            lasso_selected_features = 0
        
        return { "stats":stats_selected_features, "lasso":lasso_selected_features, 
                 "join":list(set(stats_selected_features).intersection(lasso_selected_features)) }
    
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Plots the binomial test of y over x.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column name
    :param y: str - column name
    :param bins: num - plot setting
    :param figsize: tuple - plot setting
'''
def binomial_test(dtf, x, y, bins=10, figsize=(10,10)):
    try:
        if dtf[x].dtype != "O":
            ## prepare data
            data_nonan = dtf.dropna(subset=[x])
            dtf['bin'] = bins
            dtf.loc[data_nonan.index, 'bin'] = pd.qcut(data_nonan[x], bins, duplicates='drop', labels=False)
            num_bins = len(dtf.bin.unique())-1
            dic = {v:k for k,v in enumerate(sorted(dtf.bin.unique()))}
            dtf["bin"] = dtf["bin"].apply(lambda x: dic[x])
        
            ## fig setup 
            plt.rcParams['axes.facecolor'] = 'silver'
            plt.figure(figsize=figsize)
            plt.grid(True,color='white')
            plt.suptitle(x, fontsize=20)
        
            ## barchart
            h = np.histogram(dtf["bin"], bins=num_bins+1)[1]
            hb = [(h[i]+h[i+1])/2 for i in range(len(h)-1)]
            col_h = 1.*np.histogram(dtf["bin"], bins=num_bins+1)[0]/len(dtf)    
            plt.bar(hb, col_h, color='deepskyblue')
            
            ## plot
            y_vals = (dtf.groupby('bin')[y].sum()*1.0 / dtf.groupby('bin')[y].count())
            p0 = (dtf[y].sum()*1.0 / dtf[y].count())
            plt.axhline(y=(dtf[y].sum()*1.0 / dtf[y].count()), color='k', linestyle='--', zorder=1, lw=3) 
            plt.plot(hb, y_vals, '-', c='black', zorder=5, lw=0.1)     
            
            ## binomial test
            l = []
            for b, g in dtf.groupby('bin'):
                l.append((g[y].sum(), g[y].count()))
            l = pd.Series(l).apply(pd.Series)
            lower = (l[0]/l[1])-l.apply(lambda x: scipy.stats.beta.ppf(0.05/2, x[0], x[1]-x[0]+1), axis=1)
            upper = l.apply(lambda x: scipy.stats.beta.ppf(1-0.05/2, x[0]+1, x[1]-x[0]), axis=1)-(l[0]/l[1])
            plt.scatter(hb, y_vals, s=250, c=['red' if x-lower[i]<=p0<=x+upper[i] else 'green' for i,x in enumerate(y_vals)], zorder=10)
            plt.errorbar(hb, y_vals, yerr=[lower,upper], c='k', zorder=15, lw=2.5)
            
            ## ticks
            plt.xticks(hb, np.array(dtf.groupby('bin')[x].mean()), rotation='vertical')
            ytstep = (dtf[y].sum()*1.0 / dtf[y].count())
            yticks = np.arange(0,max(col_h)+0.001,ytstep)
            plt.yticks(yticks,yticks*100)
            ax1_max = plt.ylim()[1]
            ax2 = plt.twinx()
            ax2.set_ylim([0.,ax1_max])
            ax2.set_yticks(yticks)
            plt.yticks([min(col_h),max(col_h)])
            plt.xlim([0.3, num_bins+0.5])
            plt.show()
            
            dtf = dtf.drop("bin", axis=1)
        else:
            print("chosen X aint numeric")
    
    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                   MODEL DESING & TESTING                                    #
###############################################################################
'''
Computes all the required data preprocessing.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param pk: str - name of the primary key
    :param y: str - name of the dependent variable 
    :param processNas: str or None - "mean", "median", "most_frequent"
    :param processCategorical: str or None - "dummies"
    :param split: num or None - test_size (example 0.2)
    :param scale: str or None - "standard", "minmax"
    :param task: str - "classification" or "regression"
:return
    dictionary with dtf, X_names lsit, (X_train, X_test), (Y_train, Y_test), scaler
'''
def data_preprocessing(dtf, pk, y, processNas=None, processCategorical=None, split=None, scale="standard", task="classification"):
    try:
        ## 0.tidying columns
        dtf = pop_columns(dtf, lst_cols=[pk, y], where="front")
        
        ## 1.missing
        ### check
        print(" ")
        print("1. check missing...")
        if dtf.isna().sum().sum() != 0:
            cols_with_missings = []
            for col in dtf.columns.to_list():
                if dtf[col].isna().sum() != 0:
                    print("WARNING:", col, "-->", dtf[col].isna().sum(), "Nas")
                    cols_with_missings.append(col)
            ### treat
            if processNas is not None:
                print("...treating Nas...")
                cols_with_missings_numeric = []
                for col in cols_with_missings:
                    if dtf[col].dtype == "O":
                        print(col, "categorical --> replacing Nas with label 'missing'")
                        dtf[col] = dtf[col].fillna('missing')
                    else:
                        cols_with_missings_numeric.append(col)
                if len(cols_with_missings_numeric) != 0:
                    print("replacing Nas in the numerical variables:", cols_with_missings_numeric)
                imputer = impute.SimpleImputer(strategy=processNas)
                imputer = imputer.fit(dtf[cols_with_missings_numeric])
                dtf[cols_with_missings_numeric] = imputer.transform(dtf[cols_with_missings_numeric])
        else:
            print("   OK: No missing")
                
        ## 2.categorical data
        ### check
        print(" ")
        print("2. check categorical data...")
        cols_with_categorical = []
        for col in dtf.drop(pk, axis=1).columns.to_list():
            if dtf[col].dtype == "O":
                print("WARNING:", col, "-->", dtf[col].nunique(), "categories")
                cols_with_categorical.append(col)
        ### treat
        if len(cols_with_categorical) != 0:
            if processCategorical is not None:
                print("...trating categorical...")
                for col in cols_with_categorical:
                    print(col)
                    dtf = pd.concat([dtf, pd.get_dummies(dtf[col], prefix=col)], axis=1).drop([col], axis=1)
        else:
            print("   OK: No categorical")
        
        ## 3.split train/test
        print(" ")
        print("3. split train/test...")
        X = dtf.drop([pk, y], axis=1).values
        Y = dtf[y].values
        if split is not None:
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=split, random_state=123)
            print("X_train shape:", X_train.shape, " | X_test shape:", X_test.shape)
            if task=="classification":
                print("1s in y_train:", round(sum(Y_train)/Y_train.shape[0]), " | 1s in y_test:", round(sum(Y_test)/Y_test.shape[0]))
            print(X_train.shape[1], "features:", dtf.drop([pk, y], axis=1).columns.to_list())
        else:
            print("   OK: skipped this step")
            X_train = X
            Y_train = Y
            X_test = Y_test = 0
        
        ## 4.scaling
        print(" ")
        print("4. scaling...")
        if scale is not None:
            if scale == "standard":
                scalerX = preprocessing.StandardScaler()
                scalerY = preprocessing.StandardScaler()
            elif scale == "minmax":
                scalerX = preprocessing.MinMaxScaler()
                scalerY = preprocessing.MinMaxScaler()
            else:
                print("select a scaler: 'standard' or 'minmax'")
            X_train = scalerX.fit_transform(X_train)
            if X_test != 0:
                X_test = scalerX.transform(X_test)
            if task == "regression":
                Y_train = scalerY.fit_transform(Y_train)
            else:
                scalerY = 0
            print("   OK: scaled all features")
        else:
            print("   OK: skipped this step")
            scalerX = scalerY = 0
        
        return {"dtf":dtf, "X_names":dtf.drop([pk, y], axis=1).columns.to_list(), 
                "X":(X_train, X_test), "Y":(Y_train, Y_test), "scaler":[scalerX, scalerY]}
    
    except Exception as e:
        print("--- got error ---")
        print(e)
    
    

'''
Tunes the hyperparameters of a sklearn model.
:parameter
    :param model_base: model object - model istance to tune (before fitting)
    :param param_dic: dict - dictionary of parameters to tune
    :param X_train: array - feature matrix
    :param y_train: array - y vector
    :param scoring: string - evaluation metrics like "roc_auc"
    :param searchtype: string - "RandomSearch" or "GridSearch"
:return
    model with hyperparams tuned
'''
def tuning_model(model_base, param_dic, X_train, Y_train, scoring="roc_auc", searchtype="RandomSearch"):
    try:
        ## Search
        print(searchtype.upper()+" :")
        if searchtype == "RandomSearch":
            random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=1000, scoring=scoring).fit(X_train, Y_train)
            print("Best Parameters: ", random_search.best_params_)
            print("Best Accuracy: ", random_search.best_score_)
            model = random_search.best_estimator_
            
        elif searchtype == "GridSearch":
            grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=scoring).fit(X_train, Y_train)
            print("Best Parameters: ", grid_search.best_params_)
            print("Best Accuracy: ", grid_search.best_score_)
            model = grid_search.best_estimator_
        
        ## K fold validation
        print("")
        Kfold_accuracy_base = model_selection.cross_val_score(estimator=model_base, X=X_train, y=Y_train, cv=10)
        Kfold_accuracy_model = model_selection.cross_val_score(estimator=model, X=X_train, y=Y_train, cv=10)
        print("K-FOLD VALIDATION :")
        print("accuracy mean = from", Kfold_accuracy_base.mean(), " ----> ", Kfold_accuracy_model.mean() )
        print("accuracy variance = from", Kfold_accuracy_base.std(), " ----> ", Kfold_accuracy_model.std() )
        return model
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        
        
        
'''
Fits a sklearn model.
:parameter
    :param model: model object - model to fit (before fitting)
    :param X_train: array
    :param Y_train: array
    :param X_test: array
    :param Y_test: array
    :param scalerY: scaler object (only for regression)
    :param task: str - "classification" or "regression"
    :param Y_threshold: num - predictions > threshold are 1, otherwise 0 (only for classification)
:return
    model fitted and predictions
'''
def fit_model(model, X_train, Y_train, X_test, Y_test, scalerY=None, task="classification", Y_threshold=0.5):
    if task == "classification":
        classes = ( str(np.unique(Y_train)[0]), str(np.unique(Y_train)[1]) )
        model.fit(X_train, Y_train)
        predicted_prob = model.predict_proba(X_test)[:,1]
        predicted = (predicted_prob > Y_threshold)
        print( "accuracy =", model.score(X_test, Y_test) )
        print( "auc =", metrics.roc_auc_score(Y_test, predicted_prob) )
        print( "log_loss =", metrics.log_loss(Y_test, predicted_prob) )
        print( " " )
        print( metrics.classification_report(Y_test, predicted, target_names=classes) )
    
    elif task == "regression":
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        predicted = scalerY.inverse_transform(predicted)
        print("r2 =", model.score(X_test, Y_test) )
        print("explained variance =", metrics.explained_variance_score(Y_test, predicted))
        print("mean absolute error =", metrics.mean_absolute_error(Y_test, predicted))
        print("mean squared error =", metrics.mean_squared_error(Y_test, predicted))
        predicted_prob = 0
        
    return {"model":model, "predicted_prob":predicted_prob, "predicted":predicted}



'''
Evaluates a model performance.
:parameter
    :param Y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param task: str - "classification" or "regression"
    :param figsize: tuple - plot setting
'''
def evaluate_model(Y_test, predicted, predicted_prob, task="classification", figsize=(10,10)):
    try:
        if task == "classification":
            classes = ( str(np.unique(Y_test)[0]), str(np.unique(Y_test)[1]) )
            
            ## confusion matrix
            cm = metrics.confusion_matrix(Y_test, predicted)
            plt.figure(figsize=figsize)
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion matrix')
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
            fmt = 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.show()
            
            ## roc
            fpr, tpr, thresholds = metrics.roc_curve(Y_test, predicted_prob)
            roc_auc = metrics.auc(fpr, tpr)
            plt.figure(figsize=figsize)
            plt.plot(fpr, tpr, color='darkorange', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.show()
        
        elif task == "regression":
            from statsmodels.graphics.api import abline_plot
            fig, ax = plt.subplots()
            ax.scatter(predicted, Y_test)
            abline_plot(intercept=0, slope=1, horiz=None, vert=None, model_results=None, ax=ax)
            ax.set_ylabel('Y True')
            ax.set_xlabel('Y Predicted')
            plt.show()
            
    except Exception as e:
        print("--- got error ---")
        print(e)
        

    
'''
Computes features importance.
:parameter
    :param X_train: array
    :param X_names: list
    :param Y_train: array
    :param model: model istance (after fitting)
    :param figsize: tuple - plot setting
:return
    dtf with features importance
'''
def features_importance(X_train, X_names, Y_train, model, figsize=(10,10)):
    ## importance dtf
    importances = model.feature_importances_
    dtf_importances = pd.DataFrame({"IMPORTANCE": importances, "VARIABLE": X_names}).sort_values("IMPORTANCE", ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")
    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0])
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()
    return dtf_importances.reset_index()



'''
Plots a 2-features classification model result.
:parameter
    :param X_test: array
    :param X_names: list
    :param Y_test: array
    :param model: model istance (after fitting)
    :param colors: tuple - plot setting
    :param figsize: tuple - plot setting
'''
def plot2D_classification(X_test, X_names, Y_test, model, colors={0:"black",1:"green"}, figsize=(10,10)):
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, Y_test
    X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, 
                                   stop= X_set[:, 0].max() + 1, 
                                   step=0.01),
                         np.arange(start=X_set[:, 1].min() - 1, 
                                   stop=X_set[:, 1].max() + 1, 
                                   step=0.01)
                         )
    plt.figure(figsize=figsize)
    plt.contourf(X1, X2, model.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                 alpha=0.75, cmap=ListedColormap(list(colors.values())))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i,j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c=colors[j], label=j)    
    plt.title('Classification Model')
    plt.xlabel(X_names[0])
    plt.ylabel(X_names[1])
    plt.legend()
    plt.show()



'''
Uses lime to build an a explainer.
:parameter
    :param X_train: array
    :param X_names: list
    :param model: model instance (after fitting)
    :param Y_train: array
    :param X_test_instance: array of size n x 1 (n,)
    :param task: string - "classification", "regression"
    :param top: num - top features to display
:return
    dtf with explanations
'''
def explainer(X_train, X_names, model, Y_train, X_test_instance, task="classification", top=10):
    if task=="classification":
        expainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(Y_train), mode=task)
        expainer = expainer.explain_instance(X_test_instance, model.predict_proba, num_features=top)
        dtf_explainer = pd.DataFrame(expainer.as_list(), columns=['reason','effect'])
        expainer.as_pyplot_figure()
    else:
        dtf_explainer = 0
    return dtf_explainer
    

    
###############################################################################
#                         OTHER MODELS                                        #
###############################################################################
'''
Decomposes the feture matrix of train and test.
:parameter
    :param X_train: array
    :param X_test: array
    :param algo: string - 'PCA', 'KernelPCA', 'SVD', 'LDA'
    :param Y_train: array or None - only for algo="LDA"
    :param n_features: num - how many dimensions you want
:return
    dict with new train and test, and the model 
'''
def dimensionality_reduction(X_train, X_test, algo="PCA", Y_train=None, n_features=2):
    if algo == "PCA":
        dimred_model = decomposition.PCA(n_components=n_features)
        X_train_dimred = dimred_model.fit_transform(X_train)
        X_test_dimred = dimred_model.transform(X_test)
    elif algo == "KernelPCA":
        dimred_model = decomposition.KernelPCA(n_components=n_features, kernel='rbf')
        X_train_dimred = dimred_model.fit_transform(X_train)
        X_test_dimred = dimred_model.transform(X_test)
    elif algo == "SVD":
        dimred_model = decomposition.TruncatedSVD(n_components=n_features)
        X_train_dimred = dimred_model.fit_transform(X_train)
        X_test_dimred = dimred_model.transform(X_test)
    elif algo == "LDA":
        if Y_train is not None:
            dimred_model = discriminant_analysis.LinearDiscriminantAnalysis(n_components=n_features)
            X_train_dimred = dimred_model.fit_transform(X_train, Y_train)
            X_test_dimred = dimred_model.transform(X_test)
        else:
            print("you need to give Y_train, now it's None")
            X_train_dimred = X_test_dimred = dimred_model = 0
    else:
        print("choose one algo: 'PCA', 'KernelPCA', 'SVD', 'LDA'")
        X_train_dimred = X_test_dimred = dimred_model = 0
        
    return {"X":(X_train_dimred, X_test_dimred), "model":dimred_model}
    
    

'''
Clusters data with k-means.
:paramater
    :param X: array
    :param X_names: list
    :param wcss_max_num: num or None- max iteration for wcss
    :param k: num or None - number of clusters
    :lst_features_2Dplot: list or None - two features to use for a 2D plot
:return
    dtf with X and clusters
'''
def clustering(X, X_names, wcss_max_num=10, k=3, lst_features_2Dplot=None):
    ## within-cluster sum of squares
    if wcss_max_num is not None:
        wcss = [] 
        for i in range(1, wcss_max_num + 1):
            kmeans = cluster.KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, wcss_max_num + 1), wcss)
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.show() 
    
    ## k-mean
    elif k is not None:
        model = cluster.KMeans(n_clusters=k, init='k-means++', random_state=0)
        Y_kmeans= model.fit_predict(X)
        dtf_clusters = pd.DataFrame(X, columns=X_names)
        dtf_clusters["cluster"] = Y_kmeans
        
        ## plot
        if lst_features_2Dplot is not None:
            x1_pos = X_names.index(lst_features_2Dplot[0])
            x2_pos = X_names.index(lst_features_2Dplot[1])
            sns.scatterplot(x=lst_features_2Dplot[0], y=lst_features_2Dplot[1], data=dtf_clusters, 
                            hue='cluster', style="cluster", legend="brief").set_title('K-means clustering')
            plt.scatter(kmeans.cluster_centers_[:,x1_pos], kmeans.cluster_centers_[:,x2_pos], s=200, c='red', label='Centroids')
            
        return dtf_clusters



'''
Fits a keras 3-layer artificial neural network.
:parameter
    :param X_train: array
    :param Y_train: array
    :param X_test: array
    :param Y_test: array
    :param batch_size: num - keras batch
    :param epochs: num - keras epochs
    :param Y_threshold: num - predictions > threshold are 1, otherwise 0
:return
    model fitted and predictions
'''
def ann(X_train, Y_train, X_test, Y_test, batch_size=32, epochs=100, Y_threshold=0.5):
    import keras
    ## build ann
    ### initialize
    model = keras.models.Sequential()
    n_features = X_train.shape[1]
    n_neurons = int(round((n_features + 1)/2))
    ### layer 1
    model.add(keras.layers.Dense(input_dim=n_features, units=n_neurons, kernel_initializer='uniform', activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    ### layer 2
    model.add(keras.layers.Dense(units=n_neurons, kernel_initializer='uniform', activation='relu'))
    model.add(keras.layers.Dropout(rate=0.2))
    ### layer output
    model.add(keras.layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    ### compile
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ## fit
    training = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    plt.plot(training.history['loss'], label='loss')
    plt.suptitle("Loss function during training", fontsize=20)
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.show()
    
    ## predict
    model = training.model
    predicted_prob = model.predict(X_test)
    predicted = (predicted_prob > Y_threshold)
    classes = ( str(np.unique(Y_train)[0]), str(np.unique(Y_train)[1]) )
    print( "accuracy =", metrics.accuracy_score(Y_test, predicted) )
    print( "auc =", metrics.roc_auc_score(Y_test, predicted_prob) )
    print( "log_loss =", metrics.log_loss(Y_test, predicted_prob) )
    print( " " )
    print( metrics.classification_report(Y_test, predicted, target_names=classes) )
    
    return {"model":model, "predicted_prob":predicted_prob, "predicted":predicted}
