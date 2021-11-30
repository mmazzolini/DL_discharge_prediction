from sf_runoff import create_it_matrix, create_gap
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
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error, r2_score
from sf_runoff import  smape


import pdb
import seaborn as sns

def classic_CV_SVR_predict(daily_input, C, eps,t_length,t_unit, n_splits, linear=False):#, radius_for_ensemble):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,0,t_unit)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = KFold(n_splits=n_splits)
    sets = tscv.split(it_matrix.index)
    
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        
        #reduce train_set to have a gap with test index and ensure independence
        train_index=create_gap(train_index, test_index,t_unit)

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
        target['prediction'] = svr_model_tuned.predict(X_trueTP)

        """
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
        """
        
        target['split']= np.repeat(j,test_index.shape[0]) 
    
        #add this split prediction to the 
        #pdb.set_trace()

        prediction=prediction.append(pd.DataFrame(data=target, index=test_dates))
        j=j+1

    return prediction.drop(columns='split')



def classic_CV_PCA_SVR_predict(daily_input, C, eps, n, t_length,t_unit ,n_splits,linear=False): #radius_for_ensemble):
    
    #compute the daily climatology and the quantile analysis
    daily_clim = daily_climatology_p_et_ensemble(daily_input,0,t_unit)
    #get the input-target matrix
    it_matrix=create_it_matrix(daily_input,t_length,t_unit)
    #split in train-test sets
    tscv = KFold(n_splits=n_splits)
    sets = tscv.split(it_matrix.index)
    
    j=0;
    prediction=pd.DataFrame(data=None)
    
    for train_index, test_index in sets:
        
        #reduce train_set to have a gap with test index and ensure independence
        train_index=create_gap(train_index, test_index,t_unit)

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
        test_dates = it_matrix.index[test_index]
        # and their day of the year
        doy_test_dates = test_dates.dayofyear

        # Save the true runoff values (with t_unit days rolling average)
        target = {}
        target['true_runoff'] = daily_input.Q.rolling(30, min_periods=30).mean().loc[test_dates]

        # Compute runoff monthly climatology using the whole dataset
        runoff_daily_clim = daily_input.Q.rolling(30, min_periods=30).mean()
        target['runoff_clim'] = [runoff_daily_clim.loc[runoff_daily_clim.index.dayofyear == d].mean() for d in doy_test_dates]

        X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
        target['prediction'] = svr_model_tuned.predict(X_trueTP)

        """
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
        """
        target['split']= np.repeat(j,test_index.shape[0])
            
        #pdb.set_trace()

        #add this split prediction to the 
        prediction=prediction.append(pd.DataFrame(data=target, index=test_dates))
        
        j=j+1

    return prediction.drop(columns='split')