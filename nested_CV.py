from sf_runoff import create_it_matrix
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
from sklearn.metrics import mean_squared_error

import pdb
import seaborn as sns



def SVR_nested_CV_gridsearch(daily_input, C_range, epsilon_range, t_range,t_unit,n_splits,test_siz, linear=False):


    for t_length in t_range:
        it_matrix=create_it_matrix(daily_input,t_length,t_unit).astype('float32')
        tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_siz,gap=t_unit)
        sets = tscv.split(it_matrix.index)
        pdb.set_trace()
        all_models= []
        #pdb.set_trace()
        for train_index, test_index in sets:
            #validation set is the last 2 years of the "old_training"
            val_index   = train_index[-365:]

            #training set is reduced by 2y and 1month"
            train_index_update = train_index[:-395]
            trainCvSplit = [(list(train_index_update),list(val_index))]
            
            X = it_matrix.drop(columns='Q')
            y = it_matrix['Q']
            
            
            if linear:
                svr_estimator = LinearSVR(tol=0.0001,random_state=0)
            else:
                svr_estimator = SVR(kernel='rbf', gamma='scale', cache_size=6000)


            svr_estimator = make_pipeline(StandardScaler(),
                                          TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
            parameters = {'transformedtargetregressor__regressor__C': C_range,
                          'transformedtargetregressor__regressor__epsilon': epsilon_range}


            svr_model = GridSearchCV(svr_estimator, cv=trainCvSplit, param_grid=parameters, n_jobs=-1, verbose=1, refit=True, return_train_score=True)

            # execute the grid search
            svr_model.fit(X, y)
            all_models.append(pd.DataFrame(svr_model.cv_results_))
        
        #PUT ALL THE TRAINED MODELS AND RESULTS IN A DATAFRAME
        all_m = pd.DataFrame(data=None,columns=all_models[0].columns)
        for i in all_models :
            all_m=all_m.append(i);
        
        #GROUP BY PARAMETERS AND AVERAGE OVER THE DIFFERENT VALIDATION SETS
        par=(['param_transformedtargetregressor__regressor__C','param_transformedtargetregressor__regressor__epsilon'])
        avg_models = all_m.groupby(par).mean()
        #avg_models['train_test_diff']= avg_models.mean_train_score - avg_models.mean_test_score
        
        # SELECT THE SINGLE BEST MODEL OVERALL
        best_model_overall = avg_models.loc[[avg_models.mean_test_score.idxmax()]]
        
        best_C=best_model_overall.reset_index().param_transformedtargetregressor__regressor__C[0]
        best_epsilon = best_model_overall.reset_index().param_transformedtargetregressor__regressor__epsilon[0]

        #INVESTIGATE WITH HEATMAPS THE FACT THAT WE'RE NOT OVERFITTING
        # get the coordinates of the "best model"
        y=np.where(epsilon_range==best_epsilon)[0]+0.5
        x=np.where(C_range==best_C)[0]+0.5

        hm_test = avg_models.reset_index().pivot( columns='param_transformedtargetregressor__regressor__C', index='param_transformedtargetregressor__regressor__epsilon', values='mean_test_score')
        hm_train= avg_models.reset_index().pivot(columns='param_transformedtargetregressor__regressor__C',index='param_transformedtargetregressor__regressor__epsilon',values='mean_train_score')
        plt.figure(figsize=(15,7))
        plt.subplot(1,2,1)
        sns.heatmap(hm_test,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('VALIDATION')
        plt.plot(x,y,marker='o')

        plt.subplot(1,2,2)
        sns.heatmap(hm_train,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('TRAIN')
        plt.plot(x,y,marker='o')

        plt.tight_layout()
            

        # Check if the best C (or epsion) is in the border of the grid
        if best_C == max(C_range) or best_C == min(C_range):
            print(f'Warning: best C found on the grid limit: C = {best_C}')
        if best_epsilon == max(epsilon_range) or best_epsilon == min(epsilon_range):
            print(f'Warning: best epsilon found on the grid limit: epsilon = {best_epsilon}')
        print()
    
    return best_C, best_epsilon


def SVR_PCA_nested_CV_gridsearch(daily_input, C_range, epsilon_range, components_range, t_range,t_unit,n_splits,test_size, linear=False):
                         
    for t_length in t_range:
        it_matrix=create_it_matrix(daily_input,t_length,t_unit).astype('float32')
        tscv = TimeSeriesSplit(gap=t_unit ,n_splits=n_splits, test_size=test_size)
        sets = tscv.split(it_matrix.index)
        
        all_models= []
    
        for train_index, test_index in sets:
            #validation set is the last 2 years of the "old_training"
            val_index   = train_index[-365:]

            #training set is reduced by 2y and 1month"
            train_index_update = train_index[:-395]
            trainCvSplit = [(list(train_index_update),list(val_index))]
            
            X = it_matrix.drop(columns='Q')
            y = it_matrix['Q']

            if linear:
                svr_estimator = LinearSVR(tol=0.0001, random_state=0)
            else:
                svr_estimator = SVR(kernel='rbf', gamma='scale', cache_size=6000)
                
            svr_estimator = make_pipeline(StandardScaler(),
                                          PCA(),
                                          TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
            parameters = {'pca__n_components': components_range,
                          'transformedtargetregressor__regressor__C': C_range,
                          'transformedtargetregressor__regressor__epsilon': epsilon_range
                         }

            svr_model = GridSearchCV(svr_estimator, cv=trainCvSplit, param_grid=parameters, n_jobs=-1, verbose=1, refit=True, return_train_score=True)

            # execute the grid search
            svr_model.fit(X, y)
            all_models.append(pd.DataFrame(svr_model.cv_results_))
        
        #PUT ALL THE TRAINED MODELS AND RESULTS IN A DATAFRAME
        all_m = pd.concat(all_models)
        
        #GROUP BY PARAMETERS AND AVERAGE OVER THE DIFFERENT VALIDATION SETS
        par=(['param_pca__n_components', 'param_transformedtargetregressor__regressor__C', 'param_transformedtargetregressor__regressor__epsilon'])
        
        
        #pdb.set_trace()
        avg_models = all_m.groupby(par).mean()
        #avg_models['train_test_diff']= avg_models.mean_train_score - avg_models.mean_test_score

        # SELECT THE SINGLE BEST MODEL OVERALL
        best_model_overall = avg_models.loc[[avg_models.mean_test_score.idxmax()]]
        best_C=best_model_overall.reset_index().param_transformedtargetregressor__regressor__C[0]
        best_epsilon = best_model_overall.reset_index().param_transformedtargetregressor__regressor__epsilon[0]
        best_n = best_model_overall.reset_index().param_pca__n_components[0]
        
        #INVESTIGATE WITH HEATMAPS THE FACT THAT WE'RE NOT OVERFITTING
        # get the coordinates of the "best model"
        y=np.where(epsilon_range==best_epsilon)[0]+0.5
        x=np.where(C_range==best_C)[0]+0.5

        #get the models with a certain number of components
        query=f'param_pca__n_components=={best_n}'
        nc=avg_models.query(query)

        hm_test  = nc.reset_index().pivot( columns='param_transformedtargetregressor__regressor__C', index='param_transformedtargetregressor__regressor__epsilon', values='mean_test_score')
        hm_train = nc.reset_index().pivot( columns='param_transformedtargetregressor__regressor__C', index='param_transformedtargetregressor__regressor__epsilon', values='mean_train_score')

        plt.figure(figsize=(15,7))
        plt.subplot(1,2,1)
        sns.heatmap(hm_test,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('TEST')
        plt.plot(x,y,marker='o')

        plt.subplot(1,2,2)
        sns.heatmap(hm_train,vmin=hm_test.min().min(),vmax=hm_train.max().max())
        plt.title('TRAIN')
        plt.plot(x,y,marker='o')

        plt.tight_layout()
        
               
        # Check if the best C, epsion or gamma is in the border of the grid
        if best_C == max(C_range) or best_C == min(C_range):
            print(f'Warning: best C found on the grid limit: C = {best_C}')
        if best_epsilon == max(epsilon_range) or best_epsilon == min(epsilon_range):
            print(f'Warning: best epsilon found on the grid limit: epsilon = {best_epsilon}')
        print()
        
    return best_C, best_epsilon, best_n