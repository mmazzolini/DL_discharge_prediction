import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV,TimeSeriesSplit
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt

import os

import pdb


def create_gap(train_index,test_index,gap):
    right=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==0)
    centre=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==1)
    left = ((train_index+1 == test_index[0]).sum()==0) and ((train_index-1 == test_index[-1]).sum()==1)
    if right:
        train_index=train_index[0:-gap]

    if left:
        train_index=train_index[gap:]

    if centre:
        pos = np.where(train_index+1 == test_index[0])[0][0]
        train_index=np.concatenate((train_index[:pos-gap],train_index[pos+gap:]),axis=0)
    return train_index
    
def create_in_m(daily_input, t_length,t_unit):

    # Compute the t_unit days average temperature
    if not daily_input.empty:
        daily_input_t_unit = daily_input.rolling(t_unit, min_periods=t_unit).mean()
        output = pd.concat([shift_series_(daily_input_t_unit[col], (-t_length + 1, 1),t_unit) for col in daily_input_t_unit], axis=1)

    return output;


def shift_series_(s, shift_range,t_unit):

    s_shifts = [s.shift(-t_unit * shift, freq='D').rename(s.name + (str(shift),)) for shift in range(*shift_range)]
    return pd.concat(s_shifts, axis=1)




"""
def sf_input_matrix(input_matrix, seasonal_forecast, lead_time, member):

    X = input_matrix.copy()
    X = X.loc[X.index.is_month_end, :]

    for offset in range(0, -lead_time, -1):
        sf_dates = X.index + pd.offsets.MonthBegin(offset - 1)
        X_columns = [c for c in X.columns if int(c.split('_')[1]) == offset]
        sf_columns = [f'SF_{c.split("_")[0]}_lt{lead_time + offset}_m{member}' for c in X_columns]
        X.loc[:, X_columns] = seasonal_forecast.loc[sf_dates, sf_columns].values

    return X
"""

"""
def svr_gridSearch(daily_input, t_length,t_unit, C_range=np.logspace(-3, 1, 5), epsilon_range=np.logspace(-5, 0, 5),
                   plot=False, n_splits=8):

    # svr_gridSearch run the grid search on C and epsilon svr parameters.
    
    # Read the input-target matrix 
    it_matrix = create_it_matrix(daily_input, t_length)
    X = it_matrix.drop(columns='Q')
    y = it_matrix['Q']
    
    # Set up the splits respecting of the time-series nature of the dataset

    tscv = TimeSeriesSplit(gap=t_unit,n_splits=n_splits, test_size=365)
    tscv.split(X)
    
    #pdb.set_trace()
     
    # Set up the grid search parameters
    svr_estimator = SVR(kernel='rbf', gamma='scale', cache_size=1000)
    svr_estimator = make_pipeline(StandardScaler(),
                                  TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
    parameters = {'transformedtargetregressor__regressor__C': C_range,
                  'transformedtargetregressor__regressor__epsilon': epsilon_range}
    svr_model = GridSearchCV(svr_estimator, cv=tscv, param_grid=parameters, n_jobs=-1, verbose=1, refit=True)


    # execute the grid search
    svr_model.fit(X, y)
    
    
    # Select the best C and epsilon
    best_C = svr_model.best_params_['transformedtargetregressor__regressor__C']
    best_epsilon = svr_model.best_params_['transformedtargetregressor__regressor__epsilon']
    print(f'For {t_length} months of data input: Best estimator: C={best_C}, epsilon={best_epsilon}')

    # Check if the best C (or epsion) is in the border of the grid
    if best_C == max(C_range) or best_C == min(C_range):
        print(f'Warning: best C found on the grid limit: C = {best_C}')
    if best_epsilon == max(epsilon_range) or best_epsilon == min(epsilon_range):
        print(f'Warning: best epsilon found on the grid limit: epsilon = {best_epsilon}')
    print()
    
    if plot:

        # Gridsearch plot
        plt.figure()
        scores = svr_model.cv_results_['mean_test_score'].reshape(
            len(svr_model.param_grid['transformedtargetregressor__regressor__C']),
            len(svr_model.param_grid['transformedtargetregressor__regressor__epsilon']))
        plt.imshow(scores)
        plt.colorbar()
        plt.xticks(np.arange(len(svr_model.param_grid['transformedtargetregressor__regressor__epsilon'])),
                   ['%.3f' % a for a in svr_model.param_grid['transformedtargetregressor__regressor__epsilon']], rotation=45)
        plt.yticks(np.arange(len(svr_model.param_grid['transformedtargetregressor__regressor__C'])),
                   ['%.3f' % a for a in svr_model.param_grid['transformedtargetregressor__regressor__C']])
        plt.xlabel('epsilon')
        plt.ylabel('C')
        plt.title(f'Heatmap for {t_length} months of input data')
        plt.tight_layout()

        # Scatterplot true vs. estimated training
        plt.figure()
        true_values = y
        est_values = svr_model.predict(X)
        xy = np.vstack([true_values, est_values])
        z = gaussian_kde(xy)(xy)
        idx = z.argsort()
        x, y, z = true_values[idx], est_values[idx], z[idx]
        plt.scatter(x, y, c=z)
        plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='black')
        plt.xlabel('True values')
        plt.ylabel('Estimated values')
        plt.title(f'Heatmap for {t_length} months of input data')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.draw()
    return {'C': best_C, 'epsilon': best_epsilon, 'score': svr_model.best_score_,
            'best_estimator': svr_model.best_estimator_}


def feature_sel(daily_input):

    # Find the best t_length and temperature/precipitation stations. Return score for each combination

    gridSearch_results = []
    for t_len in range(1, 12):
        gridSearch_results.append(svr_gridSearch(daily_input, 
                                                 t_len,
                                                 C_range=np.logspace(-2, 2, 6), 
                                                 epsilon_range=np.logspace(-3, 0, 6),
                                                 plot=True))

    return pd.DataFrame(data=gridSearch_results, index=range(1, 12))




def training(daily_input, t_length=None, svr_C=None, svr_epsilon=None):

    # This function takes as input the daily temperature, precipitation and runoff and train a model to predict the
    # monthly mean runoff using a time series of temperature and precipitation of length t_length as input features

    # Execute the feature selection if t_length is None
    if t_length is None:

        print('Finding the best input time series length...')
        fs = feature_sel(daily_input)
        fs_best = fs.loc[fs['score'].idxmax()]
        svr_C, svr_epsilon = fs_best['C'], fs_best['epsilon']
        t_length = fs_best.name
        print(f'Best time series length: {t_length}. '
              f'Best C={svr_C}, epsilon={svr_epsilon} (score={fs_best["score"]})')
        svr_model = fs_best['best_estimator']

    else:

        it_matrix = create_it_matrix(daily_input, t_length)
        X = it_matrix.drop(columns='Q')
        y = it_matrix['Q']

        # Fit the SVR with standardized input and target
        svr_estimator = SVR(kernel='rbf', C=svr_C, epsilon=svr_epsilon, gamma='scale', cache_size=1000)
        svr_model = make_pipeline(StandardScaler(),
                                  TransformedTargetRegressor(regressor=svr_estimator, transformer=StandardScaler()))
        svr_model.fit(X, y)

    return svr_model
"""

"""
def monthly_climatology(daily_input,t_unit):

    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)

    monthly_mean_columns = [c for c in daily_input.columns if c[0] in ['Q', 'T']]
    monthly_mean = daily_input.loc[:, monthly_mean_columns].groupby(by=daily_input.index.month).mean()
    #remember to add ['E', 'P']
    monthly_sum_columns = [c for c in daily_input.columns if c[0] in ['P','E']]
    monthly_sum = daily_input.loc[:, monthly_sum_columns].groupby(by=daily_input.index.month).sum()
    #pdb.set_trace()
    return pd.concat([monthly_mean, monthly_sum], axis=1)#[monthly_mean_columns,monthly_sum_columns]
"""


def daily_climatology(daily_input,t_unit):
    
    runoff = daily_input[['Q']]
    temp = daily_input[[c for c in daily_input.columns if c[0] == 'T']]
    prec = daily_input[[c for c in daily_input.columns if c[0] == 'P']]
    evap = daily_input[[c for c in daily_input.columns if c[0] == 'E']]
    snow = daily_input[[c for c in daily_input.columns if c[0] == 'S']]


    # Compute the t_unit days average runoff
    runoff_t_unit = runoff.rolling(30, min_periods=30).mean()

    
    # Compute the t_unit days average temperature
    if not temp.empty:
        temp_t_unit = temp.rolling(t_unit, min_periods=t_unit).mean()
        #temp_t_unit = pd.concat([shift_series_t_unitdays(temp_t_unit.loc[:, col], (-t_length + 1, 1)) for col in temp_t_unit], axis=1)


    # Compute the t_unit days average snow water equivalent
    if not snow.empty:
        snow_t_unit = snow.rolling(t_unit, min_periods=t_unit).mean()
        #temp_t_unit = pd.concat([shift_series_t_unitdays(temp_t_unit.loc[:, col], (-t_length + 1, 1)) for col in temp_t_unit], axis=1)


    # Compute the t_unit days sum precipitation
    if not prec.empty:
        prec_t_unit = prec.rolling(t_unit, min_periods=t_unit).sum()
        #prec_t_unit = pd.concat([shift_series_t_unitdays(prec_t_unit.loc[:, col], (-t_length + 1, 1)) for col in prec_t_unit], axis=1)
    
    # Compute the t_unit days sum evapotranspiration
    if not evap.empty:
        evap_t_unit = evap.rolling(t_unit, min_periods=t_unit).sum()
        #evap_t_unit = pd.concat([shift_series_t_unitdays(evap_t_unit.loc[:, col], (-t_length + 1, 1)) for col in evap_t_unit], axis=1)
        

    daily_t_unit = pd.concat([runoff_t_unit, temp_t_unit, prec_t_unit, evap_t_unit, snow_t_unit], axis=1)
    daily_mean = daily_t_unit.groupby(by=daily_t_unit.index.dayofyear).mean()

    #pdb.set_trace()
    return daily_mean



"""
def loyo_cv_lc_nofor(daily_input,output_folder, t_length=None, svr_C=None, svr_epsilon=None,
               lead_time=range(1, 8)):

    # Compute the climatology for all the inputs
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    monthly_clim = monthly_climatology(daily_input)


    # Create the total input-target matrix and select the start and end year for the loo-cv
    it_matrix = create_it_matrix(daily_input, t_length)
    year_start = min(it_matrix.index[(it_matrix.index.month == 1) & (it_matrix.index.day == 31)].year)
    year_end = max(it_matrix.index[(it_matrix.index.month == 12) & (it_matrix.index.day == 31)].year)

    # For each training length and each year train a different model and test with different configurations
    for Nyears_training in range(1, year_end-year_start+1):
        prediction = []
        for year in range(year_start, year_end+1):
            print(f'Testing on year {year}; Training years: {Nyears_training} ...')

            # Drop the years that should not be considered in the training and train the model
            daily_input_loo = daily_input.copy()
            training_years = (list(range(year+1, year_end+1)) + list(range(year_start, year)))[-Nyears_training:]
            daily_input_loo.loc[[i for i in daily_input.index if i.year not in training_years], 'Q'] = np.nan
            # daily_input_loo.loc[daily_input.index.year == year, 'Q'] = np.nan
            svr_model = training(daily_input_loo, t_length, svr_C, svr_epsilon)

            # Select the dates on which to execute the prediction
            test_dates = daily_input.index[daily_input.index.is_month_end & (daily_input.index.year == year)]

            # Save the true runoff values
            y = {}
            y['true_runoff'] = daily_input.loc[daily_input.index.year == year, 'Q'].resample("M").mean().values

            # Compute runoff monthly climatology considering the subset used for training the svr
            y['runoff_clim'] = [daily_input_loo.loc[daily_input_loo.index.month == m, 'Q'].mean() for m in range(1, 13)]

            # Predict using true temperature and precipitation (no forecast)
            X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
            y['trueTP'] = svr_model.predict(X_trueTP)

             # Predict using temperature and precipitation climatology (no forecast) for all lead times
            X_climTP = X_trueTP.copy()
            for lt in lead_time:
                change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
                change_source = [c.split('_')[0] for c in change_dest]
                X_climTP.loc[:, change_dest] = monthly_clim.loc[(test_dates - pd.offsets.MonthBegin(lt)).month, change_source].values
                y[f'climTP_lt{lt}'] = svr_model.predict(X_climTP)
                
            '''
            # Predict using seasonal forecast with different lead times
            for lt in lead_time:
                for m in range(1, 26):
                    X_sf = sf_input_matrix(X_trueTP, seasonal_forecast, lt, m)
                    y[f'sfTP_m{m}_lt{lt}'] = svr_model.predict(X_sf)
            '''
            # Store the results in a dataframe
            prediction.append(pd.DataFrame(data=y, index=test_dates-pd.offsets.MonthBegin()))
            
        pd.concat(prediction, axis=0).to_csv(
            os.path.join(output_folder, f'Runoff_forecast_{Nyears_training}_trainingyears.csv'))

def loyo_cv_lc(daily_input, seasonal_forecast, output_folder, t_length=None, svr_C=None, svr_epsilon=None,
               lead_time=range(1, 8)):

    # Compute the climatology for all the inputs
    if isinstance(daily_input, str):
        daily_input = pd.read_csv(daily_input, index_col=0, parse_dates=True)
    monthly_clim = monthly_climatology(daily_input)

    # Read the seasonal forecast of temperature and precipitation
    if isinstance(seasonal_forecast, str):
        seasonal_forecast = pd.read_csv(seasonal_forecast, index_col=0, parse_dates=True)

    # Create the total input-target matrix and select the start and end year for the loo-cv
    it_matrix = create_it_matrix(daily_input, t_length)
    year_start = min(it_matrix.index[(it_matrix.index.month == 1) & (it_matrix.index.day == 31)].year)
    year_end = max(it_matrix.index[(it_matrix.index.month == 12) & (it_matrix.index.day == 31)].year)

    # For each training length and each year train a different model and test with different configurations
    for Nyears_training in range(1, year_end-year_start+1):
        prediction = []
        for year in range(year_start, year_end+1):
            print(f'Testing on year {year}; Training years: {Nyears_training} ...')

            # Drop the years that should not be considered in the training and train the model
            daily_input_loo = daily_input.copy()
            training_years = (list(range(year+1, year_end+1)) + list(range(year_start, year)))[-Nyears_training:]
            daily_input_loo.loc[[i for i in daily_input.index if i.year not in training_years], 'Q'] = np.nan
            # daily_input_loo.loc[daily_input.index.year == year, 'Q'] = np.nan
            svr_model = training(daily_input_loo, t_length, svr_C, svr_epsilon)

            # Select the dates on which to execute the prediction
            test_dates = daily_input.index[daily_input.index.is_month_end & (daily_input.index.year == year)]

            # Save the true runoff values
            y = {}
            y['true_runoff'] = daily_input.loc[daily_input.index.year == year, 'Q'].resample("M").mean().values

            # Compute runoff monthly climatology considering the subset used for training the svr
            y['runoff_clim'] = [daily_input_loo.loc[daily_input_loo.index.month == m, 'Q'].mean() for m in range(1, 13)]

            # Predict using true temperature and precipitation (no forecast)
            X_trueTP = it_matrix.loc[test_dates, :].drop(columns='Q')
            y['trueTP'] = svr_model.predict(X_trueTP)

            # Predict using temperature and precipitation climatology (no forecast) for all lead times
            X_climTP = X_trueTP.copy()
            for lt in lead_time:
                change_dest = [c for c in X_climTP.columns if c.split('_')[1] == str(-lt + 1)]
                change_source = [c.split('_')[0] for c in change_dest]
                X_climTP.loc[:, change_dest] = monthly_clim.loc[(test_dates - pd.offsets.MonthBegin(lt)).month, change_source].values
                y[f'climTP_lt{lt}'] = svr_model.predict(X_climTP)
                

            # Predict using seasonal forecast with different lead times
            for lt in lead_time:
                for m in range(1, 26):
                    X_sf = sf_input_matrix(X_trueTP, seasonal_forecast, lt, m)
                    y[f'sfTP_m{m}_lt{lt}'] = svr_model.predict(X_sf)

            # Store the results in a dataframe
            prediction.append(pd.DataFrame(data=y, index=test_dates-pd.offsets.MonthBegin()))

        pd.concat(prediction, axis=0).to_csv(
            os.path.join(output_folder, f'Runoff_forecast_{Nyears_training}_trainingyears.csv'))
"""

# -----------------------------------------------------------------------

# Read and plot results


def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))


def monthly_rmse(fileName, plot=True):

    # Open the result file
    runoff = pd.read_csv(fileName, index_col=0, parse_dates=True)

    # Create the ensamble mean
    for lt in range(1, 8):
        columns = [c for c in runoff.columns if 'sfTP_m' in c and f'lt{lt}' in c]
        runoff[f'sfTP_em_lt{lt}'] = runoff.loc[:, columns].mean(axis=1)

    # Compute the RMSE for each month
    runoff_error = []
    for m in range(1, 12):
        runoff_m = runoff.loc[runoff.index.month == m, :]
        runoff_error.append(
            runoff_m.apply(lambda y_pred: root_mean_squared_error(runoff_m['true_runoff'], y_pred), axis=0)
        )

    runoff_error = pd.DataFrame(data=runoff_error, index=range(1, 12))

    if plot:
        plt.figure()
        for lt in range(1, 8):
            runoff_error.loc[:, f'sfTP_em_lt{lt}'].plot(marker='o', label=f'lead_time={lt}')
        runoff_error.loc[:, 'runoff_clim'].plot(marker='o', color='black', label='climatology', linewidth=3)
        runoff_error.loc[:, 'trueTP'].plot(marker='o', color='red', label='era5', linewidth=3)
        plt.legend()
        plt.ylabel('RMSE ($m^3/s$)')
        plt.xlabel('Month')


def plot_it_matrix(daily_input, var, common_ylim=True):

    # ## Plot each variable of the input-target matrix

    # Create the input-target matrix from the daily_input
    it_matrix = create_it_matrix(daily_input, 1).rename(columns=lambda c: c[:2])

    # Create the ylabel dictionary
    plt_ylabel = {'P': 'Total precipitation (m)', 'T': 'Mean temperature (K)', 'Q': 'Runoff ($m^3/s$)'}

    # Set the ylim
    if common_ylim:
        selected_vars = it_matrix.loc[:, [c for c in it_matrix.columns if c[0] == var[0]]].values
        b = (selected_vars.max() - selected_vars.min()) * 0.03
        plt_ylim = (selected_vars.min()-b, selected_vars.max()+b)

    # Plot each year with a different color using the day of the year as x axis
    plt.figure()
    for y in range(it_matrix.index.year.min(), it_matrix.index.year.max() + 1):
        curr = it_matrix[it_matrix.index.year == y]
        curr.set_index(curr.index.dayofyear, inplace=True)
        curr[var].plot(label='_nolegend_')

    # Plot the daily climatology
    clim = it_matrix[var].groupby(it_matrix.index.dayofyear).mean().loc[1:365]
    clim.plot(label='Mean', color='black', linewidth=5)

    # Set the figure properties
    if common_ylim:
        plt.ylim(plt_ylim)
    plt.xlabel('Day')
    plt.ylabel(plt_ylabel[var[0]])
    plt.legend()


def lead_time_rmse(fileName):

    # fileName = '/home/mcallegari@eurac.edu/SECLI-FIRM/Mattia/SF_runoff/Zoccolo/Results/Learning_curve/Runoff_forecast_26_trainingyears.csv'

    # Open the result file
    runoff = pd.read_csv(fileName, index_col=0, parse_dates=True)

    # Create the ensamble mean
    for lt in range(1, 8):
        columns = [c for c in runoff.columns if 'sfTP_m' in c and f'lt{lt}' in c]
        runoff[f'sfTP_em_lt{lt}'] = runoff.loc[:, columns].mean(axis=1)

    # Compute the RMSE
    runoff_error = runoff.apply(lambda y_pred: root_mean_squared_error(runoff['true_runoff'], y_pred), axis=0)

    plt.figure()
    plt.plot([1, 7], [runoff_error['runoff_clim']]*2, color='black', label='runoff climatology')
    plt.plot([1, 7], [runoff_error['trueTP']] * 2, color='red', label='era5')
    plt.plot(range(1, 8), runoff_error[[f'climTP_lt{lt}' for lt in range(1, 8)]], marker='o', label='era5 climatology')
    plt.plot(range(1, 8), runoff_error[[f'sfTP_em_lt{lt}' for lt in range(1, 8)]], marker='o', label='SEAS5 ensamble mean')
    plt.plot(range(1, 8), runoff_error[[f'sfTP_m1_lt{lt}' for lt in range(1, 8)]], color='C1', alpha=0.5, label='SEAS5 members')
    for m in range(2, 26):
        plt.plot(range(1, 8), runoff_error[[f'sfTP_m{m}_lt{lt}' for lt in range(1, 8)]], color='C1', alpha=0.5)
    plt.legend()
    plt.xlabel('Lead time (months)')
    plt.ylabel('RMSE ($m^3/s$)')

    
def spatial_avg_daily_input(daily_input):
    t_columns = [c for c in daily_input.columns if c[0] =='T']
    daily_input['T'] = daily_input[t_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = t_columns)

    e_columns = [c for c in daily_input.columns if c[0] =='E']
    daily_input['E'] = daily_input[e_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = e_columns)

    p_columns = [c for c in daily_input.columns if c[0] =='P']
    daily_input['P'] = daily_input[p_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = p_columns)
    
    s_columns = [c for c in daily_input.columns if c[0] =='S']
    daily_input['S'] = daily_input[s_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = s_columns)
    
    r_columns = [c for c in daily_input.columns if c[0] =='R']
    daily_input['R'] = daily_input[r_columns].mean(axis=1)
    daily_input=daily_input.drop(columns = r_columns)
    
    return daily_input;


def spatial_stats_daily_input(daily_input):
    
    new_daily_input=pd.DataFrame(daily_input.Q)
    
    t_columns = [c for c in daily_input.columns if c[0] =='T']
    t_vars=daily_input[t_columns]
    new_daily_input.loc[:,'T']  = t_vars.mean(axis=1)
    new_daily_input.loc[:,'T5'] =t_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'T25']=t_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'T75']=t_vars.quantile(q=0.75,axis=1)
    new_daily_input.insert(loc=5,column='T95',value=t_vars.quantile(q=0.95,axis=1))
    
    e_columns = [c for c in daily_input.columns if c[0] =='E']
    e_vars=daily_input[e_columns]
    new_daily_input.loc[:,'E']  = e_vars.mean(axis=1)
    new_daily_input.loc[:,'E5'] =e_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'E25']=e_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'E75']=e_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'E95']=e_vars.quantile(q=0.95,axis=1)
    
    p_columns = [c for c in daily_input.columns if c[0] =='P']
    p_vars=daily_input[p_columns]
    new_daily_input.loc[:,'P']  = p_vars.mean(axis=1)
    new_daily_input.loc[:,'P5'] =p_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'P25']=p_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'P75']=p_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'P95']=p_vars.quantile(q=0.95,axis=1)
    
    s_columns = [c for c in daily_input.columns if c[0] =='S']
    s_vars=daily_input[s_columns]
    new_daily_input.loc[:,'S']  = s_vars.mean(axis=1)
    new_daily_input.loc[:,'S5'] =s_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'S25']=s_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'S75']=s_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'S95']=s_vars.quantile(q=0.95,axis=1)
    
    r_columns = [c for c in daily_input.columns if c[0] =='R']
    r_vars=daily_input[r_columns]
    new_daily_input.loc[:,'R'] = r_vars.mean(axis=1)
    new_daily_input.loc[:,'R5']= r_vars.quantile(q=0.05,axis=1)
    new_daily_input.loc[:,'R25']=r_vars.quantile(q=0.25,axis=1)
    new_daily_input.loc[:,'R75']=r_vars.quantile(q=0.75,axis=1)
    new_daily_input.loc[:,'R95']=r_vars.quantile(q=0.95,axis=1)
    
    return new_daily_input

def create_gap(train_index,test_index,gap):
    right=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==0)
    centre=((train_index+1 == test_index[0]).sum()==1) and ((train_index-1 == test_index[-1]).sum()==1)
    left = ((train_index+1 == test_index[0]).sum()==0) and ((train_index-1 == test_index[-1]).sum()==1)
    if right:
        train_index=train_index[0:-gap]

    if left:
        train_index=train_index[gap:]

    if centre:
        pos = np.where(train_index+1 == test_index[0])[0][0]
        train_index=np.concatenate((train_index[:pos-gap],train_index[pos+gap:]),axis=0)
        
    return train_index;


def compute_anomalies(climatologies,pred):
    
    #compute real climatology
    anomalies=pd.DataFrame(pred.true_runoff-pred.runoff_clim,columns=['true_runoff'])
    
    #get the climatology of prediction on the wanted days
    clim_on_test_dates = pd.DataFrame(climatologies.loc[anomalies.index.dayofyear])
    
    #create an array with the proper shape
    repeated_clim=np.repeat((np.array(clim_on_test_dates.prediction)[...,np.newaxis]),
                    pred.shape[1]-2,
                    axis=1)

    #subtract clim to predictions
    anomalies_pred= pred.iloc[:,2:]- repeated_clim

    #put together real and predicted anomalies
    anomalies=pd.concat([anomalies,anomalies_pred],axis=1)

    return anomalies
