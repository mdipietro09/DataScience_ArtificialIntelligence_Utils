
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn import preprocessing, impute, utils, linear_model, feature_selection, model_selection, metrics, decomposition, discriminant_analysis, cluster
import itertools
from lime import lime_tabular



###############################################################################
#                       DATA ANALYSIS                                         #
###############################################################################
'''
'''
def utils_recognize_type(dtf, col, max_cat=20):
    if (dtf[col].dtype == "O") | (dtf[col].nunique() < max_cat):
        return "cat"
    else:
        return "num"



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
            totals= []
            for i in ax.patches:
                totals.append(i.get_width())
            if show_perc == False:
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(i.get_width()), fontsize=10, color='black')
            else:
                total= sum(totals)
                for i in ax.patches:
                    ax.text(i.get_width()+.3, i.get_y()+.20, str(round((i.get_width()/total)*100, 2))+'%', fontsize=10, color='black')
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
            ax[0].grid(True)
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
    :param max_cat: num - max number of uniques to consider a numeric variable categorical
'''
def bivariate_plot(dtf, x, y, max_cat=20, figsize=(20,10)):
    try:
        ## num vs num --> scatter with density + stacked
        if (utils_recognize_type(dtf, x, max_cat) == "num") & (utils_recognize_type(dtf, y, max_cat) == "num"):
            ### stacked
            breaks = np.quantile(dtf[x], q=np.linspace(0, 1, 11))
            groups = dtf.groupby([pd.cut(dtf[x], bins=breaks)])[y].agg(['mean','median','size'])
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
            sns.barplot(x=x, y="%", hue=y, data=b, ax=ax[1])
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
            for i in dtf[cat].unique():
                sns.distplot(dtf[dtf[cat]==i][num], hist=False, label=i, ax=ax[0])
            ax[0].grid(True)
            ### stacked
            ax[1].title.set_text('bins')
            breaks = np.quantile(dtf[num], q=np.linspace(0, 1, 11))
            tmp = dtf.groupby([cat, pd.cut(dtf[num], breaks, duplicates='drop')]).size().unstack().T
            tmp["tot"] = tmp.sum(axis=1)
            for col in tmp.drop("tot", axis=1).columns:
                tmp[col] = tmp[col] / tmp["tot"]
            tmp.drop("tot", axis=1).plot(kind='bar', stacked=True, ax=ax[1])
            ax[1].grid(True)
            ### boxplot   
            ax[2].title.set_text('outliers')
            sns.catplot(x=cat, y=num, data=dtf, kind="box", ax=ax[2])
            ax[2].grid(True)
            ### fix figure
            plt.close(2)
            plt.close(3)
            plt.show()
        
    except Exception as e:
        print("--- got error...check Nas ---")
        print(e)
        


'''
'''
def nan_analysis(dtf, na_x, y, max_cat=20, figsize=(20,10)):
    dtf_NA = dtf[[na_x, y]]
    dtf_NA[na_x] = dtf[na_x].apply(lambda x: "Value" if not pd.isna(x) else "NA")
    bivariate_plot(dtf_NA, x=na_x, y=y, max_cat=max_cat, figsize=figsize)



'''
'''
def ts_analysis(dtf, x, y, max_cat=20, figsize=(20,10)):
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        dtf_tmp = dtf.groupby(x)[y].sum()       
    else:
        dtf_tmp = dtf.groupby(x)[y].median()
    ax = dtf_tmp.plot(title=y+" by "+x)
    ax.grid(True)
      

  
'''
'''
def cross_distributions(dtf, x1, x2, y, max_cat=20, figsize=(20,10)):
    if utils_recognize_type(dtf, y, max_cat) == "cat":
        
        ## cat vs cat --> contingency table
        if (utils_recognize_type(dtf, x1, max_cat) == "cat") & (utils_recognize_type(dtf, x2, max_cat) == "cat"):
            cont_table = pd.crosstab(index=dtf[x1], columns=dtf[x2], values=dtf[y], aggfunc="sum")
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cont_table, annot=True, cmap="YlGnBu", ax=ax, linewidths=.5).set_title(x1+'  vs  '+x2)
            #return cont_table
    
        ## num vs num --> scatter with hue
        elif (utils_recognize_type(dtf, x1, max_cat) == "num") & (utils_recognize_type(dtf, x2, max_cat) == "num"):
            sns.lmplot(x=x1, y=x2, data=dtf, hue=y, height=int((figsize[0]+figsize[1])/2) )
        
        ## num vs cat --> boxplot with hue
        else:
            if (utils_recognize_type(dtf, x1, max_cat) == "cat"):
                cat,num = x1,x2
            else:
                cat,num = x2,x1
            fig, ax = plt.subplots(figsize=figsize)
            sns.boxplot(x=cat, y=num, hue=y, data=dtf, ax=ax).set_title(x1+'  vs  '+x2)
            ax.grid(True)
    
    else:
        ## all num --> 3D scatter plot
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
        coeff, p = scipy.stats.pearsonr(dtf[x], dtf[y])
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
Computes the correlation matrix with seaborn.
:parameter
    :param dtf: dataframe - input data
    :param method: str - "pearson", "spearman" ...
    :param annotation: logic - plot setting
    :param figsize: tuple - plot setting
'''
def corrmatrix_plot(dtf, method="pearson", annotation=True, figsize=(10,10)):    
    ## divide cols
    num_cols, cat_cols = [], []
    for col in dtf.columns:
        if utils_recognize_type(dtf, col) == "cat":
            cat_cols.append(col)
        else:
            num_cols.append(col)
    ## factorize
    if len(cat_cols) > 0:
        dtf_cat = dtf[cat_cols].apply(lambda x: pd.factorize(x)[0])
        dtf_num = dtf[num_cols]
        dtf = pd.concat([dtf_num, dtf_cat], axis=1)
    ## corr matrix
    corr_matrix = dtf.corr(method=method)
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
            X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=split, random_state=123, shuffle=True)
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



###############################################################################
#                   MODEL DESIGN & TESTING                                    #
###############################################################################
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
def tuning_model(model_base, param_dic, X_train, Y_train, scoring="roc_auc", searchtype="RandomSearch", n_iter=1000, cv=10):
    try:
        ## Search
        print(searchtype.upper())
        if searchtype == "RandomSearch":
            random_search = model_selection.RandomizedSearchCV(model_base, param_distributions=param_dic, n_iter=n_iter, scoring=scoring).fit(X_train, Y_train)
            print("Best Model parameters:", random_search.best_params_)
            print("Best Model mean "+scoring+":", random_search.best_score_)
            model = random_search.best_estimator_
            
        elif searchtype == "GridSearch":
            grid_search = model_selection.GridSearchCV(model_base, param_dic, scoring=scoring).fit(X_train, Y_train)
            print("Best Model parameters:", grid_search.best_params_)
            print("Best Model mean "+scoring+":", grid_search.best_score_)
            model = grid_search.best_estimator_
        
        ## K fold validation
        print("")
        Kfold_base = model_selection.cross_val_score(estimator=model_base, X=X_train, y=Y_train, cv=cv, scoring=scoring)
        Kfold_model = model_selection.cross_val_score(estimator=model, X=X_train, y=Y_train, cv=cv, scoring=scoring)
        print("K-FOLD VALIDATION")
        print(scoring+" mean: from", Kfold_base.mean(), " ----> ", Kfold_model.mean() )
        return model
    
    except Exception as e:
        print("--- got error ---")
        print(e)
        
        
        
'''
'''
def kfold_validation(model, X_train, Y_train, cv=10, figsize=(10,10)):
    cv = model_selection.StratifiedKFold(n_splits=cv, shuffle=True)
    tprs, aucs = [], []
    mean_fpr = np.linspace(0,1,100)
    fig = plt.figure(figsize=figsize)
    
    i = 1
    for train, test in cv.split(X_train, Y_train):
        prediction = model.fit(X_train[train], Y_train[train]).predict_proba(X_train[test])
        fpr, tpr, t = metrics.roc_curve(Y_train[test], prediction[:, 1])
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
    plt.title('ROC')
    plt.legend(loc="lower right")
    plt.show()


        
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
        ## Accuray e AUC
        accuracy = metrics.accuracy_score(Y_test, predicted)
        auc = metrics.roc_auc_score(Y_test, predicted_prob)
        print("Accuracy (overall correct predictions):",  round(accuracy,3))
        print("Auc:", round(auc,3))
        ## Precision e Recall
        recall = metrics.recall_score(Y_test, predicted)  #capacità del modello di beccare tutti gli 1 nel dataset (quindi anche a costo di avere falsi pos)
        precision = metrics.precision_score(Y_test, predicted)  #capacità del modello di azzeccare quando dice 1 (quindi non dare falsi pos)
        print("Recall (ability to get all 1s):", round(recall,3))  #in pratica quanti 1 ho beccato
        print("Precision (success rate when predicting a 1):", round(precision,3))  #in pratica quanti 1 erano veramente 1
        print("Detail:")
        print(metrics.classification_report(Y_test, predicted, target_names=classes))
        return {"model":model, "predicted_prob":predicted_prob, "predicted":predicted}
    
    elif task == "regression":
        model.fit(X_train, Y_train)
        predicted = model.predict(X_test)
        if scalerY is not None:
            predicted = scalerY.inverse_transform(predicted)
        print("R2:", metrics.r2_score(Y_test, predicted))
        print("Explained variance:", metrics.explained_variance_score(Y_test, predicted))
        print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, predicted))
        return {"model":model, "predicted":predicted}
        


'''
Evaluates a model performance.
:parameter
    :param Y_test: array
    :param predicted: array
    :param predicted_prob: array
    :param task: str - "classification" or "regression"
    :param figsize: tuple - plot setting
'''
def evaluate_model(Y_test, predicted, predicted_prob, task="classification", figsize=(20,10)):
    try:
        if task == "classification":
            classes = (np.unique(Y_test)[0], np.unique(Y_test)[1])
            
            ## confusion matrix
            cm = metrics.confusion_matrix(Y_test, predicted, labels=classes)
            cm = pd.DataFrame(cm)
            fig, ax = plt.subplots(figsize=figsize)
            sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap=plt.cm.Blues, cbar=False)
            ax.set(xlabel="Pred", ylabel="True")
            plt.yticks(rotation=0)
            plt.title("Confusion matrix")
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
            plt.ylabel('True Positive Rate (Recall)')
            plt.title('Receiver operating characteristic')
            plt.legend(loc="lower right")
            plt.grid(True)
            plt.show()
            
            ## precision-recall curve
            precision, recall, thresholds = metrics.precision_recall_curve(Y_test, predicted_prob)
            plt.figure(figsize=figsize)
            plt.plot(recall, precision)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall curve')
            plt.grid(True)
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
        explainer = lime_tabular.LimeTabularExplainer(training_data=X_train, feature_names=X_names, class_names=np.unique(Y_train), mode=task)
        explained = explainer.explain_instance(X_test_instance, model.predict_proba, num_features=top)
        dtf_explainer = pd.DataFrame(explained.as_list(), columns=['reason','effect'])
        explained.as_pyplot_figure()
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
def ann_classif(X_train, Y_train, X_test, Y_test, batch_size=32, epochs=100, Y_threshold=0.5):
    from tensorflow.keras import models, layers
    ## build ann
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
    
    ## fit
    training = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    plt.plot(training.history['loss'], label='loss')
    plt.suptitle("Loss function during training", fontsize=20)
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.show()
    print(training.model.summary())
    model = training.model
    
    ## predict
    predicted_prob = model.predict(X_test)
    predicted = (predicted_prob > Y_threshold)
    classes = ( str(np.unique(Y_train)[0]), str(np.unique(Y_train)[1]) )
    print( "accuracy =", metrics.accuracy_score(Y_test, predicted) )
    print( "auc =", metrics.roc_auc_score(Y_test, predicted_prob) )
    print( "log_loss =", metrics.log_loss(Y_test, predicted_prob) )
    print( " " )
    print( metrics.classification_report(Y_test, predicted, target_names=classes) )
    
    return {"model":model, "predicted_prob":predicted_prob, "predicted":predicted}



'''
'''
def ann_regr(X_train, Y_train, X_test, Y_test, batch_size=32, epochs=100, scalerY=None):
    from tensorflow.keras import models, layers
    ## build ann
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
    model.compile(optimizer='adam', loss='mean_absolute_error', metrics=['mean_absolute_error'])
    
    ## fit
    training = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs)
    plt.plot(training.history['loss'], label='loss')
    plt.suptitle("Loss function during training", fontsize=20)
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.show()
    print(training.model.summary())
    model = training.model
    
    ## predict
    predicted = model.predict(X_test)
    if scalerY is not None:
        predicted = scalerY.inverse_transform(predicted)
    print("R2:", metrics.r2_score(Y_test, predicted))
    print("Explained variance:", metrics.explained_variance_score(Y_test, predicted))
    print("Mean Absolute Error:", metrics.mean_absolute_error(Y_test, predicted))
    
    return {"model":model, "predicted":predicted}
