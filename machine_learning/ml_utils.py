
## for data
import pandas as pd
import numpy as np

## for plotting
import matplotlib.pyplot as plt
import seaborn as sns

## for statistical tests
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm

## for machine learning
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, discriminant_analysis, cluster, ensemble
from tensorflow.keras import models, layers

## for explainer
from lime import lime_tabular



###############################################################################
#                       DATA ANALYSIS                                         #
###############################################################################
'''
Recognize whether a column is numerical or categorical.
'''
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"



'''
Get a general overview of a dataframe.
:parameter
    :param dtf: dataframe - input data
    :param max_cat: num - to recognize column type
'''
def dtf_overview(dtf, max_cat=20, figsize=(10,5)):
    ## recognize column type
    dic_cols = {col:utils_recognize_type(dtf, col, max_cat=max_cat) for col in dtf.columns}
        
    ## print info
    len_dtf = len(dtf)
    print("Shape:", dtf.shape)
    print("-----------------")
    for col in dtf.columns:
        info = col+" --> Type:"+dic_cols[col]
        info = info+" | Nas: "+str(dtf[col].isna().sum())+"("+str(int(dtf[col].isna().mean()*100))+"%)"
        if dic_cols[col] == "cat":
            info = info+" | Categories: "+str(dtf[col].nunique())
        else:
            info = info+" | Min-Max: "+str(int(dtf[col].min()))+"-"+str(int(dtf[col].max()))
        if dtf[col].nunique() == len_dtf:
            info = info+" | Possible PK"
        print(info)
                
    ## plot heatmap
    fig, ax = plt.subplots(figsize=figsize)
    heatmap = dtf.isnull()
    for k,v in dic_cols.items():
        if v == "num":
            heatmap[k] = heatmap[k].apply(lambda x: 0.5 if x is False else 1)
        else:
            heatmap[k] = heatmap[k].apply(lambda x: 0 if x is False else 1)
    sns.heatmap(heatmap, vmin=0, vmax=1, cbar=False, ax=ax).set_title('Dataset Overview')
    #plt.setp(plt.xticks()[1], rotation=0)
    plt.show()
    
    ## add legend
    print("\033[1;37;40m Categerocial ", "\033[1;30;41m Numerical ", "\033[1;30;47m NaN ")



'''
Plots the frequency distribution of a dtf column.
:parameter
    :param dtf: dataframe - input data
    :param x: str - column name
    :param max_cat: num - max number of uniques to consider a numerical variable as categorical
    :param top: num - plot setting
    :param show_perc: logic - plot setting
    :param bins: num - plot setting
    :param quantile_breaks: tuple - plot distribution between these quantiles (to exclude outilers)
    :param box_logscale: logic
    :param figsize: tuple - plot settings
'''
def freqdist_plot(dtf, x, max_cat=20, top=20, show_perc=True, bins=100, quantile_breaks=(0,10), box_logscale=False, figsize=(20,10)):
    try:
        ## cat --> freq
        if utils_recognize_type(dtf, x, max_cat) == "cat":   
            ax = dtf[x].value_counts().head(top).sort_values().plot(kind="barh", figsize=figsize)
            totals = []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=10, color='black')
            else:
                total = sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
            ax.grid(axis="x")
            plt.suptitle(x, fontsize=20)
            plt.show()
            
        ## num --> density
        else:
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x, fontsize=20)
            ### distribution
            ax[0].title.set_text('distribution')
            variable = dtf[x].fillna(dtf[x].mean())
            breaks = np.quantile(variable, q=np.linspace(0, 1, 11))
            variable = variable[ (variable > breaks[quantile_breaks[0]]) & (variable < breaks[quantile_breaks[1]]) ]
            sns.distplot(variable, hist=True, kde=True, kde_kws={"shade": True}, ax=ax[0])
            des = dtf[x].describe()
            ax[0].axvline(des["25%"], ls='--')
            ax[0].axvline(des["mean"], ls='--')
            ax[0].axvline(des["75%"], ls='--')
            ax[0].grid(True)
            des = round(des, 2).apply(lambda x: str(x))
            box = '\n'.join(("min: "+des["min"], "25%: "+des["25%"], "mean: "+des["mean"], "75%: "+des["75%"], "max: "+des["max"]))
            ax[0].text(0.95, 0.95, box, transform=ax[0].transAxes, fontsize=10, va='top', ha="right", 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=1))
            ### boxplot 
            if box_logscale == True:
                ax[1].title.set_text('outliers (log scale)')
                tmp_dtf = pd.DataFrame(dtf[x])
                tmp_dtf[x] = np.log(tmp_dtf[x])
                tmp_dtf.boxplot(column=x, ax=ax[1])
            else:
                ax[1].title.set_text('outliers')
                dtf.boxplot(column=x, ax=ax[1])
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
    :param max_cat: num - max number of uniques to consider a numerical variable as categorical
'''
def bivariate_plot(dtf, x, y, max_cat=20, figsize=(20,10)):
    try:
        ## num vs num --> scatter with density + stacked
        if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
            ### stacked
            dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
            breaks = np.quantile(dtf_noNan[x], q=np.linspace(0, 1, 11))
            groups = dtf_noNan.groupby([pd.cut(dtf_noNan[x], bins=breaks, duplicates='drop')])[y].agg(['mean','median','size'])
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            groups[["mean", "median"]].plot(kind="line", ax=ax)
            groups["size"].plot(kind="bar", ax=ax, rot=45, secondary_y=True, color="grey", alpha=0.3, grid=True)
            ax.set(ylabel=y)
            ax.right_ax.set_ylabel("Observazions in each bin")
            plt.show()
            ### joint plot
            sns.jointplot(x=x, y=y, data=dtf, dropna=True, kind='reg', height=int((figsize[0]+figsize[1])/2) )
            plt.show()

        ## cat vs cat --> hist count + hist %
        elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):  
            fig, ax = plt.subplots(nrows=1, ncols=2,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### count
            ax[0].title.set_text('count')
            order = dtf.groupby(x)[y].count().index.tolist()
            sns.catplot(x=x, hue=y, data=dtf, kind='count', order=order, ax=ax[0])
            ax[0].grid(True)
            ### percentage
            ax[1].title.set_text('percentage')
            a = dtf.groupby(x)[y].count().reset_index()
            a = a.rename(columns={y:"tot"})
            b = dtf.groupby([x,y])[y].count()
            b = b.rename(columns={y:0}).reset_index()
            b = b.merge(a, how="left")
            b["%"] = b[0] / b["tot"] *100
            sns.barplot(x=x, y="%", hue=y, data=b, ax=ax[1]).get_legend().remove()
            ax[1].grid(True)
            ### fix figure
            plt.close(2)
            plt.close(3)
            plt.show()
    
        ## num vs cat --> density + stacked + boxplot 
        else:
            if (utils_recognize_type(dtf, x, max_cat) == "cat"):
                cat,num = x,y
            else:
                cat,num = y,x
            fig, ax = plt.subplots(nrows=1, ncols=3,  sharex=False, sharey=False, figsize=figsize)
            fig.suptitle(x+"   vs   "+y, fontsize=20)
            ### distribution
            ax[0].title.set_text('density')
            for i in sorted(dtf[cat].unique()):
                sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)
            ### stacked
            dtf_noNan = dtf[dtf[num].notnull()]  #can't have nan
            ax[1].title.set_text('bins')
            breaks = np.quantile(dtf_noNan[num], q=np.linspace(0,1,11))
            tmp = dtf_noNan.groupby([cat, pd.cut(dtf_noNan[num], breaks, duplicates='drop')]).size().unstack().T
            tmp = tmp[dtf_noNan[cat].unique()]
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1)[sorted(dtf[cat].unique())].plot(kind='bar', stacked=True, ax=ax[1], legend=False, grid=True)
            ### boxplot   
            ax[2].title.set_text('outliers')
            sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2], order=sorted(dtf[cat].unique()))
            ax[2].grid(True)
            ### fix figure
            plt.close(2)
            plt.close(3)
            plt.show()
        
    except Exception as e:
        print("--- got error ---")
        print(e)
        


'''
Plots a bivariate analysis using Nan and not-Nan as categories.
'''
def nan_analysis(dtf, na_x, y, max_cat=20, figsize=(20,10)):
    dtf_NA = dtf[[na_x, y]]
    dtf_NA[na_x] = dtf[na_x].apply(lambda x: "Value" if not pd.isna(x) else "NA")
    bivariate_plot(dtf_NA, x=na_x, y=y, max_cat=max_cat, figsize=figsize)



'''
Plots a bivariate analysis with time variable.
'''
def ts_analysis(dtf, x, y, max_cat=20, figsize=(20,10)):
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        dtf_tmp = dtf.groupby(x)[y].sum()       
    else:
        dtf_tmp = dtf.groupby(x)[y].median()
    ax = dtf_tmp.plot(title=y+" by "+x)
    ax.grid(True)
      

  
'''
plots multivariate analysis.
'''
def cross_distributions(dtf, x1, x2, y, max_cat=20, figsize=(20,10)):
    ## Y cat
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        
        ### cat vs cat --> contingency table
        if (utils_recognize_type(dtf, x1, max_cat) == "cat") & (utils_recognize_type(dtf, x2, max_cat) == "cat"):
            cont_table = pd.crosstab(index=dtf[x1], columns=dtf[x2], values=dtf[y], aggfunc="sum")
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cont_table, annot=True, fmt='.0f', cmap="YlGnBu", ax=ax, linewidths=.5).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
    
        ### num vs num --> scatter with hue
        elif (utils_recognize_type(dtf, x1, max_cat) == "num") & (utils_recognize_type(dtf, x2, max_cat) == "num"):
            sns.lmplot(x=x1, y=x2, data=dtf, hue=y, height=int((figsize[0]+figsize[1])/2) )
        
        ### num vs cat --> boxplot with hue
        else:
            if (utils_recognize_type(dtf, x1, max_cat) == "cat"):
                cat,num = x1,x2
            else:
                cat,num = x2,x1
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(x=cat, y=num, hue=y, data=dtf, ax=ax).set_title(x1+'  vs  '+x2+'  (filter: '+y+')')
            ax.grid(True)
    
    ## Y num
    else:
        ### all num --> 3D scatter plot
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=figsize)
        ax = fig.gca(projection='3d')
        plot3d = ax.scatter(xs=dtf[x1], ys=dtf[x2], zs=dtf[y], c=dtf[y], cmap='inferno', linewidth=0.5)
        fig.colorbar(plot3d, shrink=0.5, aspect=5, label=y)
        ax.set(xlabel=x1, ylabel=x2, zlabel=y)
        plt.show()


    
'''
Computes correlation/dependancy and p-value (prob of happening something different than what observed in the sample)
'''
def test_corr(dtf, x, y, max_cat=20):
    ## num vs num --> pearson
    if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
        dtf_noNan = dtf[dtf[x].notnull()]  #can't have nan
        coeff, p = scipy.stats.pearsonr(dtf_noNan[x], dtf_noNan[y])
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Pearson Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## cat vs cat --> cramer (chiquadro)
    elif (utils_recognize_type(dtf, x, max_cat) == "cat") & (utils_recognize_type(dtf, y, max_cat) == "cat"):
        cont_table = pd.crosstab(index=dtf[x], columns=dtf[y])
        chi2_test = scipy.stats.chi2_contingency(cont_table)
        chi2, p = chi2_test[0], chi2_test[1]
        n = cont_table.sum().sum()
        phi2 = chi2/n
        r,k = cont_table.shape
        phi2corr = max(0, phi2-((k-1)*(r-1))/(n-1))
        rcorr = r-((r-1)**2)/(n-1)
        kcorr = k-((k-1)**2)/(n-1)
        coeff = np.sqrt(phi2corr/min((kcorr-1), (rcorr-1)))
        coeff, p = round(coeff, 3), round(p, 3)
        conclusion = "Significant" if p < 0.05 else "Non-Significant"
        print("Cramer Correlation:", coeff, conclusion, "(p-value: "+str(p)+")")
    
    ## num vs cat --> 1way anova (f: the means of the groups are different)
    else:
        if (utils_recognize_type(dtf, x, max_cat) == "cat"):
            cat,num = x,y
        else:
            cat,num = y,x
        model = smf.ols(num+' ~ '+cat, data=dtf).fit()
        table = sm.stats.anova_lm(model)
        p = table["PR(>F)"][0]
        coeff, p = None, round(p, 3)
        conclusion = "Correlated" if p < 0.05 else "Non-Correlated"
        print("Anova F: the variables are", conclusion, "(p-value: "+str(p)+")")
        
    return coeff, p


     
'''
Moves columns into a dtf.
:parameter
    :param dtf: dataframe - input data
    :param lst_cols: list - names of the columns that must be moved
    :param where: str - "front" or "end"
:return
    dtf with moved columns
'''
def pop_columns(dtf, lst_cols, where="end"):
    current_cols = dtf.columns.tolist()
    for col in lst_cols:    
        current_cols.pop( current_cols.index(col) )
    if where == "front":
        dtf = dtf[lst_cols + current_cols]
    else:
        dtf = dtf[current_cols + lst_cols]
    return dtf



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
            print("Example: ", pk+"==", ERROR.iloc[0,0])
            return dtf[ dtf[pk]==ERROR.iloc[0,0] ]
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        


###############################################################################
#                       PREPROCESSING                                         #
###############################################################################
'''
Split the dataframe into train / test
'''
def dtf_partitioning(dtf, y, test_size=0.3, shuffle=False):
    dtf_train, dtf_test = model_selection.train_test_split(dtf, test_size=test_size, shuffle=shuffle) 
    print("X_train shape:", dtf_train.drop(y, axis=1).shape, "| X_test shape:", dtf_test.drop(y, axis=1).shape)
    print("y_train mean:", round(np.mean(dtf_train[y]),2), "| y_test mean:", round(np.mean(dtf_test[y]),2))
    print(dtf_train.shape[1], "features:", dtf_train.drop(y, axis=1).columns.to_list())
    return dtf_train, dtf_test



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
Replace Na with a specific value or mean for numerical and mode for categorical. 
'''
def fill_na(dtf, x, value=None):
    if value is None:
        value = dtf[x].mean() if utils_recognize_type(dtf, x) == "num" else dtf[x].mode().iloc[0]
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf, value
    else:
        print("--- Replacing Nas with:", value, "---")
        dtf[x] = dtf[x].fillna(value)
        return dtf



'''
Transforms a categorical column into dummy columns
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param x: str - column name
    :param dropx: logic - whether the x column should be dropped
:return
    dtf with dummy columns added
'''
def add_dummies(dtf, x, dropx=False):
    dtf_dummy = pd.get_dummies(dtf[x], prefix=x, drop_first=True, dummy_na=False)
    dtf = pd.concat([dtf, dtf_dummy], axis=1)
    print( dtf.filter(like=x, axis=1).head() )
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf
    


'''
Reduces the classes a categorical column.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param x: str - column name
    :param dic_clusters_mapping: dict - ex: {"min":[30,45,180], "max":[60,120], "mean":[]}  where the residual class must have an empty list
    :param dropx: logic - whether the x column should be dropped
'''
def add_feature_clusters(dtf, x, dic_clusters_mapping, dropx=False):
    dic_flat = {v:k for k,lst in dic_clusters_mapping.items() for v in lst}
    for k,v in dic_clusters_mapping.items():
        if len(v)==0:
            residual_class = k 
    dtf[x+"_cluster"] = dtf[x].apply(lambda x: dic_flat[x] if x in dic_flat.keys() else residual_class)
    if dropx == True:
        dtf = dtf.drop(x, axis=1)
    return dtf



'''
Scales features.
'''
def scaling(dtf, y, scalerX=None, scalerY=None, fitted=False, task="classification"):
    scalerX = preprocessing.MinMaxScaler(feature_range=(0,1)) if scalerX is None else scalerX
    if fitted is False:
        scalerX.fit(dtf.drop(y, axis=1))
    X = scalerX.transform(dtf.drop(y, axis=1))
    dtf_scaled = pd.DataFrame(X, columns=dtf.drop(y, axis=1).columns, index=dtf.index)
    if task == "regression":
        scalerY = preprocessing.MinMaxScaler(feature_range=(0,1)) if scalerY is None else scalerY
        dtf_scaled[y] = scalerY.fit_transform(dtf[y].values.reshape(-1,1)) if fitted is False else dtf[y]
        return dtf_scaled, scalerX, scalerY
    else:
        dtf_scaled[y] = dtf[y]
        return dtf_scaled, scalerX



'''
Computes all the required data preprocessing.
:parameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable 
    :param processNas: str or None - "mean", "median", "most_frequent"
    :param processCategorical: str or None - "dummies"
    :param split: num or None - test_size (example 0.2)
    :param scale: str or None - "standard", "minmax"
    :param task: str - "classification" or "regression"
:return
    dictionary with dtf, X_names lsit, (X_train, X_test), (Y_train, Y_test), scaler
'''
def data_preprocessing(dtf, y, processNas=None, processCategorical=None, split=None, scale=None, task="classification"):
    try:
        dtf = pop_columns(dtf, [y], "front")
        
        ## missing
        ### check
        print("--- check missing ---")
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
                
        ## categorical data
        ### check
        print("--- check categorical data ---")
        cols_with_categorical = []
        for col in dtf.columns.to_list():
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
        print("--- split train/test ---")
        X = dtf.drop(y, axis=1).values
        Y = dtf[y].values
        if split is not None:
            X_train, X_test, y_train, y_test = model_selection.train_test_split(X, Y, test_size=split, shuffle=False)
            print("X_train shape:", X_train.shape, " | X_test shape:", X_test.shape)
            print("y_train mean:", round(np.mean(y_train),2), " | y_test mean:", round(np.mean(y_test),2))
            print(X_train.shape[1], "features:", dtf.drop(y, axis=1).columns.to_list())
        else:
            print("   OK: step skipped")
            X_train, y_train, X_test, y_test = X, Y, None, None
        
        ## 4.scaling
        print("--- scaling ---")
        if scale is not None:
            scalerX = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
            X_train = scalerX.fit_transform(X_train)
            scalerY = 0
            if X_test is not None:
                X_test = scalerX.transform(X_test)
            if task == "regression":
                scalerY = preprocessing.StandardScaler() if scale == "standard" else preprocessing.MinMaxScaler()
                y_train = scalerY.fit_transform(y_train.reshape(-1,1))
            print("   OK: scaled all features")
        else:
            print("   OK: step skipped")
            scalerX, scalerY = 0, 0
        
        return {"dtf":dtf, "X_names":dtf.drop(y, axis=1).columns.to_list(), 
                "X":(X_train, X_test), "y":(y_train, y_test), "scaler":(scalerX, scalerY)}
    
    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                  FEATURES SELECTION                                         #
###############################################################################    
'''
Computes the correlation matrix with seaborn.
:parameter
    :param dtf: dataframe - input data
    :param method: str - "pearson", "spearman" ...
    :param annotation: logic - plot setting
'''
def corrmatrix_plot(dtf, method="pearson", annotation=True, figsize=(10,10)):    
    ## factorize
    corr_matrix = dtf.copy()
    for col in corr_matrix.columns:
        if corr_matrix[col].dtype == "O":
            print("--- WARNING: Factorizing labels of", col, "---")
            corr_matrix[col] = corr_matrix[col].factorize(sort=True)[0]
    ## corr matrix
    corr_matrix = corr_matrix.corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr_matrix, vmin=-1., vmax=1., annot=annotation, fmt='.2f', cmap="YlGnBu", ax=ax, cbar=True, linewidths=0.5)
    plt.title(method + " correlation")
    return corr_matrix



'''
Performs features selections: by correlation (keeping the lowest p-value) and by lasso.
:prameter
    :param dtf: dataframe - feature matrix dtf
    :param y: str - name of the dependent variable
    :param top: num - number of top features
    :param task: str - "classification" or "regression"
:return
    dic with lists of features to keep.
'''     
def features_selection(dtf, y, top=10, task="classification", figsize=(20,10)):
    try:
        dtf_X = dtf.drop(y, axis=1)
        feature_names = dtf_X.columns
        
        ## Anova
        model = feature_selection.f_classif if task=="classification" else feature_selection.f_regression
        selector = feature_selection.SelectKBest(score_func=model, k=top).fit(dtf_X.values, dtf[y].values)
        anova_selected_features = feature_names[selector.get_support()]
        
        ## Lasso regularization
        model = linear_model.LogisticRegression(C=1, penalty="l1", solver='liblinear') if task=="classification" else linear_model.Lasso(alpha=1.0, fit_intercept=True)
        selector = feature_selection.SelectFromModel(estimator=model, max_features=top).fit(dtf_X.values, dtf[y].values)
        lasso_selected_features = feature_names[selector.get_support()]
        
        ## plot
        dtf_features = pd.DataFrame({"features":feature_names})
        dtf_features["anova"] = dtf_features["features"].apply(lambda x: "anova" if x in anova_selected_features else "")
        dtf_features["num1"] = dtf_features["features"].apply(lambda x: 1 if x in anova_selected_features else 0)
        dtf_features["lasso"] = dtf_features["features"].apply(lambda x: "lasso" if x in lasso_selected_features else "")
        dtf_features["num2"] = dtf_features["features"].apply(lambda x: 1 if x in lasso_selected_features else 0)
        dtf_features["method"] = dtf_features[["anova","lasso"]].apply(lambda x: (x[0]+" "+x[1]).strip(), axis=1)
        dtf_features["selection"] = dtf_features["num1"] + dtf_features["num2"]
        fig, ax = plt.subplots(figsize=figsize)
        sns.barplot(y="features", x="selection", hue="method", data=dtf_features.sort_values("selection", ascending=False), ax=ax, dodge=False)
               
        join_selected_features = list(set(anova_selected_features).intersection(lasso_selected_features))
        return {"anova":anova_selected_features, "lasso":lasso_selected_features, "join":join_selected_features}
    
    except Exception as e:
        print("--- got error ---")
        print(e)



'''
Computes features importance.
:parameter
    :param X: array
    :param X_names: list
    :param model: model istance (after fitting)
    :param figsize: tuple - plot setting
:return
    dtf with features importance
'''
def features_importance(X, y, X_names, model=None, task="classification", figsize=(10,10)):
    ## model
    if model is None:
        if task == "classification":
            model = ensemble.RandomForestClassifier(n_estimators=100, criterion="entropy", random_state=0)  
        elif task == "regression":
            model = ensemble.RandomForestRegressor(n_estimators=100, criterion="mse", random_state=0)
    model.fit(X,y)
    print("--- model used ---")
    print(model)
    
    ## importance dtf
    importances = model.feature_importances_
    dtf_importances = pd.DataFrame({"IMPORTANCE":importances, "VARIABLE":X_names}).sort_values("IMPORTANCE", ascending=False)
    dtf_importances['cumsum'] = dtf_importances['IMPORTANCE'].cumsum(axis=0)
    dtf_importances = dtf_importances.set_index("VARIABLE")
    
    ## plot
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=False, figsize=figsize)
    fig.suptitle("Features Importance", fontsize=20)
    ax[0].title.set_text('variables')
    dtf_importances[["IMPORTANCE"]].sort_values(by="IMPORTANCE").plot(kind="barh", legend=False, ax=ax[0]).grid(axis="x")
    ax[0].set(ylabel="")
    ax[1].title.set_text('cumulative')
    dtf_importances[["cumsum"]].plot(kind="line", linewidth=4, legend=False, ax=ax[1])
    ax[1].set(xlabel="", xticks=np.arange(len(dtf_importances)), xticklabels=dtf_importances.index)
    plt.xticks(rotation=70)
    plt.grid(axis='both')
    plt.show()
    return dtf_importances.reset_index()



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
            print("chosen x aint numerical")
    
    except Exception as e:
        print("--- got error ---")
        print(e)



###############################################################################
#                   MODEL DESIGN & TESTING - CLASSIFICATION                   #
###############################################################################
'''
Fits a sklearn classification model.
:parameter
    :param model: model object - model to fit (before fitting)
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param threshold: num - predictions > threshold are 1, otherwise 0 (only for classification)
:return
    model fitted and predictions
'''
def fit_classif_model(model, X_train, y_train, X_test, threshold=0.5):
    ## model
    model = ensemble.GradientBoostingClassifier() if model is None else model
    
    ## train/test
    model.fit(X_train, y_train)
    predicted_prob = model.predict_proba(X_test)[:,1]
    predicted = (predicted_prob > threshold)
    return model, predicted_prob, predicted



'''
Perform k-fold validation.
'''
def utils_kfold_validation(model, X_train, y_train, cv=10, figsize=(10,10)):
    cv = model_selection.StratifiedKFold(n_splits=cv, shuffle=True)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0,1,100)
    fig = plt.figure(figsize=figsize)
    
    i = 1
    for train, test in cv.split(X_train, y_train):
        prediction = model.fit(X_train[train], y_train[train]).predict_proba(X_train[test])
        fpr, tpr, t = metrics.roc_curve(y_train[test], prediction[:, 1])
        tprs.append(scipy.interp(mean_fpr, fpr, tpr))
        roc_auc = metrics.auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i = i+1
        
    plt.plot([0,1], [0,1], linestyle='--', lw=2, color='black')
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = metrics.auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='blue', label=r'Mean ROC (AUC = %0.2f )' % (mean_auc), lw=2, alpha=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('K-FOLD VALIDATION')
    plt.legend(loc="lower right")
    plt.show()
    
    

'''
Tunes the hyperparameters of a sklearn model.
:parameter
    :param model_base: model object - model istance to tune (before fitting)
    :param param_dic: dict - dictionary of parameters to tune
    :param X_train: array - feature matrix
    :param y_train: array - y vector
    :param scoring: string - "roc_auc" or "accuracy"
    :param searchtype: string - "RandomSearch" or "GridSearch"
:return
    model with hyperparams tuned
'''
def tune_classif_model(X_train, y_train, model_base=None, param_dic=None, scoring="accuracy", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5)):
    ## params
    model_base = ensemble.GradientBoostingClassifier() if model_base is None else model_base
    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750], 'max_depth':[2,3,4,5,6,7]} if param_dic is None else param_dic                        
            
    ## Search
    print("---", searchtype, "---")
    if searchtype == "RandomSearch":
        random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", random_search.best_params_)
        print("Best Model mean "+scoring+":", random_search.best_score_)
        model = random_search.best_estimator_
        
    elif searchtype == "GridSearch":
        grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", grid_search.best_params_)
        print("Best Model mean "+scoring+":", grid_search.best_score_)
        model = grid_search.best_estimator_
    
    ## K fold validation
    print("")
    Kfold_base = model_selection.cross_val_score(estimator=model_base, X=X_train, y=y_train, cv=cv, scoring=scoring)
    Kfold_model = model_selection.cross_val_score(estimator=model, X=X_train, y=y_train, cv=cv, scoring=scoring)
    print("--- Kfold Validation ---")
    print("mean", scoring, "- base:", Kfold_base.mean(), " --> best:", Kfold_model.mean())
    utils_kfold_validation(model, X_train, y_train, cv=cv, figsize=figsize)
    return model



'''
Fits a keras artificial neural network.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param batch_size: num - keras batch
    :param epochs: num - keras epochs
    :param threshold: num - predictions > threshold are 1, otherwise 0
:return
    model fitted and predictions
'''
def fit_ann_classif(X_train, y_train, X_test, model=None, batch_size=32, epochs=100, threshold=0.5):
    ## model
    if model is None:
        ### initialize
        model = models.Sequential()
        n_features = X_train.shape[1]
        n_neurons = int(round((n_features + 1)/2))
        ### layer 1
        model.add(layers.Dense(input_dim=n_features, units=n_neurons, kernel_initializer='uniform', activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        ### layer 2
        model.add(layers.Dense(units=n_neurons, kernel_initializer='uniform', activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        ### layer output
        model.add(layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        ### compile
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    ## train
    print(model.summary())
    training = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0, validation_split=0.3)
    utils_plot_keras_training(training)
    
    ## test
    predicted_prob = training.model.predict(X_test)
    predicted = (predicted_prob > threshold)
    return training.model, predicted_prob, predicted



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
    :param predicted_prob: array
'''
def evaluate_classif_model(y_test, predicted, predicted_prob, figsize=(20,10)):
    classes = np.unique(y_test)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)
    
    ## Accuray e AUC
    accuracy = metrics.accuracy_score(y_test, predicted)
    auc = metrics.roc_auc_score(y_test, predicted_prob)
    print("Accuracy (overall correct predictions):",  round(accuracy,2))
    print("Auc:", round(auc,2))
    
    ## Precision e Recall
    recall = metrics.recall_score(y_test, predicted)  #capacità del modello di beccare tutti gli 1 nel dataset (quindi anche a costo di avere falsi pos)
    precision = metrics.precision_score(y_test, predicted)  #capacità del modello di azzeccare quando dice 1 (quindi non dare falsi pos)
    print("Recall (all 1s predicted right):", round(recall,2))  #in pratica quanti 1 ho beccato
    print("Precision (confidence when predicting a 1):", round(precision,2))  #in pratica quanti 1 erano veramente 1
    print("Detail:")
    print(metrics.classification_report(y_test, predicted, target_names=[str(i) for i in classes]))
       
    ## Plot confusion matrix
    cm = metrics.confusion_matrix(y_test, predicted, labels=classes)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax[0], cmap=plt.cm.Blues, cbar=False)
    ax[0].set(xlabel="Pred", ylabel="True", title="Confusion matrix")
    ax[0].set_yticklabels(labels=classes, rotation=0)
 
    ## Plot roc
    fpr, tpr, thresholds = metrics.roc_curve(y_test, predicted_prob)
    roc_auc = metrics.auc(fpr, tpr)     
    ax[1].plot(fpr, tpr, color='darkorange', lw=3, label='ROC curve (area = %0.2f)' % roc_auc)
    ax[1].plot([0,1], [0,1], color='navy', lw=3, linestyle='--')
    ax[1].set(xlim=[0.0,1.0], ylim=[0.0,1.05], xlabel='False Positive Rate', ylabel="True Positive Rate (Recall)", title="Receiver operating characteristic")     
    ax[1].legend(loc="lower right")
    ax[1].grid(True)

    ## Plot precision-recall curve
    precision, recall, thresholds = metrics.precision_recall_curve(y_test, predicted_prob)
    ax[2].plot(recall, precision, lw=3)
    ax[2].set(xlabel='Recall', ylabel="Precision", title="Precision-Recall curve")
    ax[2].grid(True)
    plt.show()
     


###############################################################################
#                   MODEL DESIGN & TESTING - REGRESSION                       #
###############################################################################
'''
Fits a sklearn regression model.
:parameter
    :param model: model object - model to fit (before fitting)
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param scalerY: scaler object (only for regression)
:return
    model fitted and predictions
'''
def fit_regr_model(model, X_train, y_train, X_test, scalerY=None):  
    ## model
    model = linear_model.LinearRegression() if model is None else model
    
    ## train/test
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)
    if scalerY is not None:
        predicted = scalerY.inverse_transform(predicted.reshape(-1,1))
    return model, predicted
        


'''
Tunes the hyperparameters of a sklearn model.
:parameter
    :param model_base: model object - model istance to tune (before fitting)
    :param param_dic: dict - dictionary of parameters to tune
    :param X_train: array - feature matrix
    :param y_train: array - y vector
    :param scoring: string - "roc_auc" or "accuracy"
    :param searchtype: string - "RandomSearch" or "GridSearch"
:return
    model with hyperparams tuned
'''
def tune_regr_model(X_train, y_train, model_base=None, param_dic=None, scoring="accuracy", searchtype="RandomSearch", n_iter=1000, cv=10, figsize=(10,5)):
    ## params
    model_base = ensemble.GradientBoostingClassifier() if model_base is None else model_base
    param_dic = {'learning_rate':[0.15,0.1,0.05,0.01,0.005,0.001], 'n_estimators':[100,250,500,750,1000,1250,1500,1750], 'max_depth':[2,3,4,5,6,7]} if param_dic is None else param_dic                        
            
    ## Search
    print("---", searchtype, "---")
    if searchtype == "RandomSearch":
        random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", random_search.best_params_)
        print("Best Model mean "+scoring+":", random_search.best_score_)
        model = random_search.best_estimator_
        
    elif searchtype == "GridSearch":
        grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=scoring).fit(X_train, y_train)
        print("Best Model parameters:", grid_search.best_params_)
        print("Best Model mean "+scoring+":", grid_search.best_score_)
        model = grid_search.best_estimator_
    
    ## K fold validation
    print("")
    Kfold_base = model_selection.cross_val_score(estimator=model_base, X=X_train, y=y_train, cv=cv, scoring=scoring)
    Kfold_model = model_selection.cross_val_score(estimator=model, X=X_train, y=y_train, cv=cv, scoring=scoring)
    print("--- Kfold Validation ---")
    print("mean", scoring, "- base:", Kfold_base.mean(), " --> best:", Kfold_model.mean())
    utils_kfold_validation(model, X_train, y_train, cv=cv, figsize=figsize)
    return model



'''
Fits a keras artificial neural network.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param batch_size: num - keras batch
    :param epochs: num - keras epochs
    :param scalerY: scaler object (only for regression)
:return
    model fitted and predictions
'''
def fit_ann_regr(X_train, y_train, X_test, scalerY, model=None, batch_size=32, epochs=100):
    ## model
    if model is None:
        ### define R2 metric for Keras
        from tensorflow.keras import backend as K
        def R2(y, y_hat):
            ss_res =  K.sum(K.square(y - y_hat)) 
            ss_tot = K.sum(K.square(y - K.mean(y))) 
            return ( 1 - ss_res/(ss_tot + K.epsilon()) )

        ### initialize
        model = models.Sequential()
        n_features = X_train.shape[1]
        n_neurons = int(round((n_features + 1)/2))
        ### layer 1
        model.add(layers.Dense(input_dim=n_features, units=n_neurons, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        ### layer 2
        model.add(layers.Dense(units=n_neurons, kernel_initializer='normal', activation='relu'))
        model.add(layers.Dropout(rate=0.2))
        ### layer output
        model.add(layers.Dense(units=1, kernel_initializer='normal', activation='linear'))
        ### compile
        model.compile(optimizer='adam', loss='mean_absolute_error', metrics=[R2])

    ## train
    print(model.summary())
    training = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=0, validation_split=0.3)
    utils_plot_keras_training(training)
    
    ## test
    predicted = training.model.predict(X_test)
    if scalerY is not None:
        predicted = scalerY.inverse_transform(predicted)
    return training.model, predicted



'''
Evaluates a model performance.
:parameter
    :param y_test: array
    :param predicted: array
'''
def evaluate_regr_model(y_test, predicted, figsize=(15,5)):
    ## Kpi
    print("R2 (explained variance):", round(metrics.r2_score(y_test, predicted), 2))
    print("Mean Absolute Perc Error (|y-pred|/y):", round(np.mean(np.abs((y_test-predicted)/predicted)), 2))
    print("Mean Absolute Error (|y-pred|):", "{:,.0f}".format(metrics.mean_absolute_error(y_test, predicted)))
    print("Root Mean Squared Error (sqrt((y-pred)^2)):", "{:,.0f}".format(np.sqrt(metrics.mean_squared_error(y_test, predicted))))
    print("Max Error:", "{:,.0f}".format(metrics.max_error(y_test, predicted)))

    ## Plot predicted vs true
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    from statsmodels.graphics.api import abline_plot
    ax[0].scatter(predicted, y_test, color="black")
    abline_plot(intercept=0, slope=1, color="red", ax=ax[0])
    ax[0].grid(True)
    ax[0].set(xlabel="Predicted", ylabel="True", title="Predicted vs True")
    
    ### Plot predicted vs residuals
    residuals = y_test.reshape(-1,1) - predicted
    ax[1].scatter(predicted, residuals, color="red")
    ax[1].grid(True)
    ax[1].set(xlabel="Predicted", ylabel="Residuals", title="Residuals")
    plt.hlines(y=0, xmin=np.min(predicted), xmax=np.max(predicted))
    plt.show()
    


###############################################################################
#                         UNSUPERVISED                                        #
###############################################################################
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
Decomposes the feture matrix of train and test.
:parameter
    :param X_train: array
    :param X_test: array
    :param y_train: array or None - only for algo="LDA"
    :param n_features: num - how many dimensions you want
:return
    dict with new train and test, and the model 
'''
def utils_dimensionality_reduction(X_train, X_test, y_train=None, n_features=2):
    if (y_train is None) or len(np.unique(y_train)<=2):
        print("--- unsupervised: pca ---")
        model = decomposition.PCA(n_components=n_features)
        X_train = model.fit_transform(X_train)
    else:
        print("--- supervised: lda ---")
        model = discriminant_analysis.LinearDiscriminantAnalysis(n_components=n_features)
        X_train = model.fit_transform(X_train, y_train)
    X_test = model.transform(X_test)
    return {"X":(X_train, X_test), "model":model}



'''
Plots a 2-features classification model result.
:parameter
    :param X_train: array
    :param y_train: array
    :param X_test: array
    :param y_test: array
    :param model: model istance (before fitting)
'''
def plot2d_classif_model(X_train, y_train, X_test, y_test, model=None, annotate=False, figsize=(10,10)):
    ## n features > 2d
    if X_train.shape[1] > 2:
        pca = utils_dimensionality_reduction(X_train, X_test, y_train, n_features=2)
        X_train, X_test = pca["X"]
     
    ## fit 2d model
    print("--- fitting 2d model ---")
    model_2d = ensemble.GradientBoostingClassifier() if model is None else model
    model_2d.fit(X_train, y_train)
    
    ## plot predictions
    print("--- plotting test set ---")
    from matplotlib.colors import ListedColormap
    colors = {np.unique(y_test)[0]:"black", np.unique(y_test)[1]:"green"}
    X1, X2 = np.meshgrid(np.arange(start=X_test[:,0].min()-1, stop=X_test[:,0].max()+1, step=0.01),
                         np.arange(start=X_test[:,1].min()-1, stop=X_test[:,1].max()+1, step=0.01))
    fig, ax = plt.subplots(figsize=figsize)
    Y = model_2d.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape)
    ax.contourf(X1, X2, Y, alpha=0.5, cmap=ListedColormap(list(colors.values())))
    ax.set(xlim=[X1.min(),X1.max()], ylim=[X2.min(),X2.max()], title="Classification regions")
    for i in np.unique(y_test):
        ax.scatter(X_test[y_test==i, 0], X_test[y_test==i, 1], c=colors[i], label="true "+str(i))  
    if annotate is True:
        for n,i in enumerate(y_test):
            ax.annotate(n, xy=(X_test[n,0], X_test[n,1]), textcoords='offset points', ha='left', va='bottom')
    plt.legend()
    plt.show()
    
  
    
'''
Plot regression plane.
'''
def plot3d_regr_model(X_train, y_train, X_test, y_test, scalerY=None, model=None, rotate=(0,0), figsize=(10,10)):
    ## n features > 2d
    if X_train.shape[1] > 2:
        pca = utils_dimensionality_reduction(X_train, X_test, y_train, n_features=2)
        X_train, X_test = pca["X"]
    
    ## fit 2d model
    print("--- fitting 2d model ---")
    model_2d = linear_model.LinearRegression() if model is None else model
    model_2d.fit(X_train, y_train)
    
    ## plot predictions
    print("--- plotting test set ---")
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(plt.figure(figsize=figsize), elev=rotate[0], azim=rotate[1])
    ax.scatter(X_test[:,0], X_test[:,1], y_test, color="black")
    X1 = np.array([[X_test.min(), X_test.min()], [X_test.max(), X_test.max()]])
    X2 = np.array([[X_test.min(), X_test.max()], [X_test.min(), X_test.max()]])
    Y = model_2d.predict(np.array([[X_test.min(), X_test.min(), X_test.max(), X_test.max()], 
                                   [X_test.min(), X_test.max(), X_test.min(), X_test.max()]]).T).reshape((2,2))
    Y = scalerY.inverse_transform(Y) if scalerY is not None else Y
    ax.plot_surface(X1, X2, Y, alpha=0.5)
    ax.set(zlabel="Y", title="Regression plane", xticklabels=[], yticklabels=[])
    plt.show()
    
    

###############################################################################
#                       EXPLAINABILITY                                        #
###############################################################################
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
def explainer(X_train, X_names, model, y_train, X_test_instance, task="classification", top=10):
    if task == "classification":
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(y_train), mode=task)
        explained = explainer.explain_instance(X_test_instance, model.predict_proba, num_features=top)
        dtf_explainer = pd.DataFrame(explained.as_list(), columns=['reason','effect'])
        explained.as_pyplot_figure()
        
    elif task == "regression":
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names="Y", mode=task)
        explained = explainer.explain_instance(X_test_instance, model.predict, num_features=top)
        dtf_explainer = pd.DataFrame(explained.as_list(), columns=['reason','effect'])
        explained.as_pyplot_figure()
    
    return dtf_explainer



'''
Plot loss and metrics of keras training.
'''
def utils_plot_keras_training(training):
    metric = [k for k in training.history.keys() if ("loss" not in k) and ("val" not in k)][0]
    # fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,3))  
    # ax[0].set(title="Loss")
    # ax[0].plot(training.history['loss'], label='training')
    # ax[0].plot(training.history['val_loss'], label='validation')
    # ax[0].set_xlabel('Epochs')
    # ax[0].legend()
    # ax[0].grid(True)
    # ax[1].set(title="Accuracy")
    # ax[1].plot(training.history[metric], label='training')
    # ax[1].plot(training.history['val_'+metric], label='validation')
    # ax[1].set_xlabel('Epochs')
    # ax[1].grid(True)
    # plt.show()
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
Extract info for each layer in a keras model.
'''
def utils_ann_config(model):
    lst_layers = []
    for layer in model.layers:
        if "drop" in layer.name:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":0, 
                         "out":int(layer.output.shape[-1]), "activation":None,
                         "params":0, "bias":0} 
        else:
            dic_layer = {"name":layer.name, "in":int(layer.input.shape[-1]), "neurons":layer.units, 
                         "out":int(layer.output.shape[-1]), "activation":layer.get_config()["activation"],
                         "params":layer.get_weights()[0], "bias":layer.get_weights()[1]} 
        lst_layers.append(dic_layer)
    return lst_layers



'''
Plot the structure of a keras neural network.
'''
def visualize_ann(model, titles=False, figsize=(10,8)):
    ## get layers info
    lst_layers = utils_ann_config(model)
    layer_sizes, layer_infos = [], []
    for i,layer in enumerate(lst_layers):
        if "drop" not in layer["name"]:
            layer_sizes.append(layer["in"] if i==0 else layer["out"])
            layer_infos.append( (layer["activation"], layer["params"], layer["bias"]) )
    
    ## fig setup
    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    ax.set(title="Neural Network structure")
    ax.axis('off')
    left, right, bottom, top = 0.1, 0.9, 0.1, 0.9
    x_space = (right-left) / float(len(layer_sizes)-1)
    y_space = (top-bottom) / float(max(layer_sizes))
    p = 0.025
    
    ## nodes
    for i,n in enumerate(layer_sizes):
        top_on_layer = y_space*(n-1)/2.0 + (top+bottom)/2.0
        
        ### add titles
        if titles is True:
            if i == 0:
                plt.text(x=left, y=top, fontsize=10, s="Input layer")
                plt.text(x=left, y=top-p, fontsize=8, s="activation: "+str(layer_infos[i][0]))
                plt.text(x=left, y=top-2*p, fontsize=8,
                         s="params:"+str(layer_infos[i][1].shape)+" + bias:"+str(layer_infos[i][2].shape))
            elif i == len(layer_sizes)-1:
                plt.text(x=right, y=top, fontsize=10, s="Output layer")
                plt.text(x=right, y=top-p, fontsize=8, s="activation: "+str(layer_infos[i][0]))
                plt.text(x=right, y=top-2*p, fontsize=8,
                         s="params:"+str(layer_infos[i][1].shape)+" + bias:"+str(layer_infos[i][2].shape))
            else:
                plt.text(x=left+i*x_space, y=top, fontsize=10, s="Hidden layer "+str(i))
                plt.text(x=left+i*x_space, y=top-p, fontsize=8, s="activation: "+str(layer_infos[i][0]))
                plt.text(x=left+i*x_space, y=top-2*p, fontsize=8,
                         s="params:"+str(layer_infos[i][1].shape)+" + bias:"+str(layer_infos[i][2].shape))
        
        ### circles
        for m in range(n):
            color = "limegreen" if (i == 0) or (i == len(layer_sizes)-1) else "red" 
            circle = plt.Circle(xy=(left+i*x_space, top_on_layer-m*y_space-4*p), radius=y_space/4.0, color=color, ec='k', zorder=4)
            ax.add_artist(circle)
            
            ### add text
            if i == 0:
                plt.text(x=left-4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$X_{'+str(m+1)+'}$')
            elif i == len(layer_sizes)-1:
                plt.text(x=right+4*p, y=top_on_layer-m*y_space-4*p, fontsize=10, s=r'$y_{'+str(m+1)+'}$')
            else:
                plt.text(x=left+i*x_space+p, y=top_on_layer-m*y_space+(y_space/8.+0.01*y_space)-4*p, fontsize=10, s=r'$H_{'+str(m+1)+'}$')
    
    ## links
    for i, (n_a, n_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = y_space*(n_a-1)/2. + (top+bottom)/2. -4*p
        layer_top_b = y_space*(n_b-1)/2. + (top+bottom)/2. -4*p
        for m in range(n_a):
            for o in range(n_b):
                line = plt.Line2D([i*x_space+left, (i+1)*x_space+left], [layer_top_a-m*y_space, layer_top_b-o*y_space], c='k', alpha=0.5)
                ax.add_artist(line)
    plt.show()