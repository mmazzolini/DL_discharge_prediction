from sf_runoff import create_it_matrix
from climatology_ensemble import  daily_climatology_p_et_ensemble
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import numpy as np

from sklearn.svm import SVR, LinearSVR
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from sf_runoff import  smape


import pdb
import seaborn as sns

def nested_CV_SVR_predict(daily_input, C, eps, t_length, t_unit, n_splits, test_size, radius_for_ensemble, linear=False):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,t_unit,radius_for_ensemble)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = TimeSeriesSplit(gap=t_unit ,n_splits=n_splits, test_size=test_size)
    sets = tscv.split(it_matrix.index)
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        #set up training features
        X = it_matrix.drop(columns='Q').iloc[train_index]
        y = it_matrix['Q'].iloc[train_index]
        
        #set up the model according to the parameters
        if linear:
            svr_estimator = LinearSVR(tol=0.0001, C=C, epsilon=eps,random_state=0)
        else:
            svr_estimator = SVR(kernel='rbf', gamma='scale', C=C, epsilon=eps, cache_size=7000)
        svr_model_tuned = make_pipeline(StandardScaler(),
                                        TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))

        #fit the model
        svr_model_tuned.fit(X, y)

        # get the test dates (end of the month)
        test_dates = it_matrix.index[test_index]#[daily_input.index.is_month_end]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]

        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]

        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['trueTP'] = svr_model_tuned.predict(X_trueTP)


        # Predict using temperature and precipitation climatology
        # predict also for 25th and 75th quantile situations.
        
        X_climTP = X_trueTP.copy()
        X_climTP_Q25=X_trueTP.copy()
        X_climTP_Q75=X_trueTP.copy()

        #predict till 6 months of advance
        lead_times = range(1,6)
        for lt in lead_times:
            
            
            # modify the X matrix by substituting the climatology to the real meteo vars for lt months.
            change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
            change_source = [c.split('_')[0] for c in change_dest]
            #pdb.set_trace()
            X_climTP.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source].values
            
            #predict
            target[f'climTP_lt{lt}'] = svr_model_tuned.predict(X_climTP)

            # modify the X matrix by substituting the climatology to the extreme (25th and 75th quantiles) meteo vars for lt months.
            change_source_25 = []
            change_source_75 = []
            #modify the source, by taking the daily climathological data referred to the quantiles situations
            for i in change_source:
                    change_source_25.append(i+'_Q25')
                    change_source_75.append((i+'_Q75'))

            X_climTP_Q25.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_25].values
            target[f'climTP_lt{lt}_Q25'] = svr_model_tuned.predict(X_climTP_Q25)

            X_climTP_Q75.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_75].values
            target[f'climTP_lt{lt}_Q75'] = svr_model_tuned.predict(X_climTP_Q75)
            #pdb.set_trace()

        target['split']= np.repeat(j,test_size)
            
        #add this split prediction to the 
        prediction=prediction.append(pd.DataFrame(data=target, index=test_dates))
        
        j=j+1

    return prediction



def nested_CV_PCA_SVR_predict(daily_input, C, eps, n, t_length, t_unit,  n_splits, test_size, radius_for_ensemble, linear=False):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,t_unit,radius_for_ensemble)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = TimeSeriesSplit(gap=t_unit ,n_splits=n_splits, test_size=test_size)
    sets = tscv.split(it_matrix.index)
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        #set up training features
        X = it_matrix.drop(columns='Q').iloc[train_index]
        y = it_matrix['Q'].iloc[train_index]
        
        #set up the model according to the parameters
        if linear:
            svr_estimator = LinearSVR(tol=0.0001, C=C, epsilon=eps,random_state=0)
        else:
            svr_estimator = SVR(kernel='rbf', gamma='scale', C=C, epsilon=eps, cache_size=7000)
        svr_model_tuned = make_pipeline(StandardScaler(),
                                        PCA(n_components=n),
                                      TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))

        #fit the model
        svr_model_tuned.fit(X, y)

        # get the test dates (end of the month)
        test_dates = it_matrix.index[test_index]#[daily_input.index.is_month_end]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]

        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]

        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['trueTP'] = svr_model_tuned.predict(X_trueTP)


        # Predict using temperature and precipitation climatology
        # predict also for 25th and 75th quantile situations.
        
        X_climTP = X_trueTP.copy()
        X_climTP_Q25=X_trueTP.copy()
        X_climTP_Q75=X_trueTP.copy()

        #predict till 6 periods of advance
        lead_times = range(1,6)
        for lt in lead_times:
            
            # modify the X matrix by substituting the climatology to the real meteo vars for lt months.
            change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
            change_source = [c.split('_')[0] for c in change_dest]
            #pdb.set_trace()
            X_climTP.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source].values
            
            #predict
            target[f'climTP_lt{lt}'] = svr_model_tuned.predict(X_climTP)

            # modify the X matrix by substituting the climatology to the extreme (25th and 75th quantiles) meteo vars for lt months.
            change_source_25 = []
            change_source_75 = []
            #modify the source, by taking the daily climathological data referred to the quantiles situations
            for i in change_source:
                    change_source_25.append(i+'_Q25')
                    change_source_75.append((i+'_Q75'))

            X_climTP_Q25.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_25].values
            target[f'climTP_lt{lt}_Q25'] = svr_model_tuned.predict(X_climTP_Q25)

            X_climTP_Q75.loc[:, change_dest]=daily_clim.loc[(test_dates-np.timedelta64(t_unit*(lt-1),'D')).dayofyear][change_source_75].values
            target[f'climTP_lt{lt}_Q75'] = svr_model_tuned.predict(X_climTP_Q75)
            
            #pdb.set_trace()
        target['split']= np.repeat(j,test_size)

            
        #add this split prediction to the 
        prediction = prediction.append(pd.DataFrame(data=target, index=test_dates))
        #pdb.set_trace()
        j=j+1

    return prediction



def plot_prediction(prediction):

    splits=prediction['split'].max()
    for i in range(splits+1):
        query=f'split=={i}'
        #query='split==' + str(i)
        pred=prediction.query(query)
        pred.loc[:,'date']= pred.index

        ax,fig=plt.subplots(figsize=(20,10))
        #plot the real
        sns.lineplot(y=("true_runoff"),x="date",data=pred,color='red',linewidth=1.3,legend='auto')
        sns.lineplot(y=("runoff_clim"),x="date",data=pred,color='yellow',linewidth=1.3,legend='auto')
        sns.lineplot(y=("trueTP"),x="date",data=pred,color='green',linewidth=1.3,legend='auto')


        #plot the lead_time_
        lt1=pred[["climTP_lt1","climTP_lt1_Q25","climTP_lt1_Q75"]]
        #lt1.columns=np.repeat('climatologia_lt1_ensemple_prec',3)
        sns.lineplot(data=lt1["climTP_lt1"],legend='auto')
        plt.fill_between(x=lt1.index, y1=lt1['climTP_lt1_Q25'], y2=lt1['climTP_lt1_Q75'], alpha=0.2)

        """
        #plot the lead_time_
        lt4=pred[["climTP_lt4","climTP_lt4_Q25","climTP_lt4_Q75"]]
        #lt4.columns=np.repeat('climatologia_lt4_ensemple_prec',3)
        sns.lineplot(data=lt4["climTP_lt4"], palette=['green'],legend='auto')
        plt.fill_between(x=lt4.index, y1=lt4['climTP_lt4_Q25'], y2=lt4['climTP_lt4_Q75'], alpha=0.2)
        """

        plt.ylabel('30 days discharge average [m^3/sec]')

        plt.legend(['TRUE DISCHARGE', 'DISCHARGE CLIMATOLOGY', 'LEAD TIME = 0', 'LEAD TIME = 1'])    
        plt.title("Precipitation variability(Q= 25 and 75) mapped on the prediction discharge")
    return;



def plot_anomalies(prediction):

    splits=prediction['split'].max()
    for i in range(splits+1):
        query=f'split=={i}'
        #query='split==' + str(i)
        pred=prediction.query(query)
        pred.loc[:,'date']= pred.index

        ax,fig=plt.subplots(figsize=(20,10))
        #plot the real
        sns.lineplot(y=("true_runoff"),x="date",data=pred,color='red',linewidth=1.3,legend='auto')
        sns.lineplot(y=("trueTP"),x="date",data=pred,color='green',linewidth=1.3,legend='auto')


        #plot the lead_time_
        lt1=pred[["climTP_lt1","climTP_lt1_Q25","climTP_lt1_Q75"]]
        #lt1.columns=np.repeat('climatologia_lt1_ensemple_prec',3)
        sns.lineplot(data=lt1["climTP_lt1"],legend='auto')
        plt.fill_between(x=lt1.index, y1=lt1['climTP_lt1_Q25'], y2=lt1['climTP_lt1_Q75'], alpha=0.2)

        #plot the lead_time_
        lt4=pred[["climTP_lt4","climTP_lt4_Q25","climTP_lt4_Q75"]]
        #lt4.columns=np.repeat('climatologia_lt4_ensemple_prec',3)
        sns.lineplot(data=lt4["climTP_lt4"], palette=['green'],legend='auto')
        plt.fill_between(x=lt4.index, y1=lt4['climTP_lt4_Q25'], y2=lt4['climTP_lt4_Q75'], alpha=0.2)

        plt.axhline(0,ls='--')
        plt.ylabel('30 days averaged discharge anomaly [m^3/sec]')

        plt.legend(['TRUE DISCHARGE','LEAD TIME = 0','LEAD TIME = 1','LEAD TIME = 4'])    
        plt.title("Anomalies plotting with precipitation variability(Q= 25 and 75) mapped on the prediction discharge")
    return;


def evaluate_prediction(prediction):
    to_drop=[]
    for c in prediction.columns:
        if c[-3] == 'Q':
            to_drop.append(c)
    to_drop.append('split')
    runoff=prediction.drop(columns=to_drop)
    
    dictio={"true_runoff" : "measured runoff",
        "runoff_clim" : "runoff climatology",
        "trueTP"      : "model output",
        "climTP_lt1"  : "output 1 month lead time",
        "climTP_lt2"  : "output 2 month lead time",
        "climTP_lt3"  : "output 3 month lead time",
        "climTP_lt4"  : "output 4 month lead time",
        "climTP_lt5"  : "output 5 month lead time",
       }
       
    runoff = runoff.rename(columns=dictio)
    runoff_error_r2=runoff.apply(lambda y_pred: r2_score(runoff['measured runoff'], y_pred), axis=0)
    #runoff_error_smape=runoff.apply(lambda y_pred: smape(runoff['true_runoff'], y_pred), axis=0)
    
    #runoff_error_smape.plot.bar(ylabel="SMAPE score [/]")
    

    
    #pdb.set_trace()
   
    runoff_error_r2.plot.bar(ylabel="R^2 score [/]", rot=90)
    #plt.legend(['MEASURED DISCHARGE','RUNOFF CLIMATOLOGY','MODEL PREDICTION','PREDICTION WITH 1 MONTH LEAD TIME','','']
    
    return runoff_error_r2