
"""Data imputer (regression | mode,median,constant,... | timeseries)

This script test three diffrent machine learning regressors in order
to predict the four most relevant wireline logs,DTC, NPHI, DTS, and RHOB.
Afterwards, the models' results are compared between each other in each
prediction stage in order to select the best performing model. Later, 
at each stage the missing instances encountered in the training, open test,
and hidden test datasets are imputed by the predictions otained by the best
ML model at each stage. Finally, 6 additional features are included in each
data subset.

It requires xgboost, lightgbm, catboost to be installed before running, as
well as pandas and numpy functionalities.
"""

def regression_imputer(traindata, testdata, hiddendata):
  
  """Receives the pre-processed data and returns the data with additional
  columns named as cleaned data. These additional columns include the 
  predicted, augmented, and 6 additional features icluding impedances 
  (S_I, P_I), bulk and shear modulus (K, G), slowness ratio (DT_R), and 
  true and measure depths ratio (MD_TVD).

  Parameters
  ----------
  training: Dataframe
    Pre-processed training dataframe.
  testdata: Dataframe
    Pre-processed open test dataframe.
  hiddendata: Dataframe
    Pre-processed hidden test dataframe.

  Returns
  ----------
  cleaned_traindata: Dataframe
    Cleaned trainig dataframe.
  cleaned_testdata: Dataframe
    Cleaned test dataframe.
  cleaned_hiddendata: Dataframe
    Cleaned hidden dataframe.
  """
  import numpy as np
  import pandas as pd
  from xgboost import XGBRegressor
  from lightgbm import LGBMRegressor
  from catboost import CatBoostRegressor
  from sklearn.model_selection import train_test_split
  from sklearn.metrics import max_error, mean_absolute_error, mean_squared_error
  from sklearn.metrics import explained_variance_score, r2_score
  print('Number of Nan data is >>> tarin {} | test{} | hidden{}'. format(traindata.isna().sum().sum(), testdata.isna().sum().sum(), hiddendata.isna().sum().sum()))
  # list of regressors to be tests set to be supported by GPUs
  estimators = [XGBRegressor(n_estimators=250,
                             tree_method='gpu_hist',
                             learning_rate=0.05,
                             objective ='reg:squarederror',), 
                
                CatBoostRegressor(task_type='GPU'),
                
                #LGBMRegressor(device='gpu',
                #            gpu_platform_id=1,
                #            gpu_device_id=0), 
                ]
                
  estimator_names = ['EXTREME BOOST REGRESSOR',
                     'CATBOOST REGRESSOR',
                     #'LIGHTBOOST REGRESSOR'
                    ] # regressors names

  # predicting DTS
  print('-----------------------------------------PREDINCTING DTS------------------------------------------')
  
  """Getting data containing DTS readings and 
  spliting into a new training and validation subsets.
  """
  traindata_dts = traindata[traindata.DTS.notna()]
  X_dts = traindata_dts.drop(['LITHO', 'DTS'], axis=1)
  Y_dts = traindata_dts['DTS']

  """Inputing remanent features by their median.
  """
  X_dts_inp = X_dts.apply(lambda x: x.fillna(x.median()), axis=0) #Imputation
  X_dts_train, X_dts_val, Y_dts_train, Y_dts_val = train_test_split(X_dts_inp,
                                                                    Y_dts, 
                                                                    test_size=0.3,
                                                                    random_state=42) #train-validation split
  """Getting DTS testing set from open test set
  """
  testdata_dts = testdata[testdata.DTS.notna()]
  X_dts = testdata_dts.drop(['LITHO', 'DTS'], axis=1)
  X_dts_test = X_dts.apply(lambda x: x.fillna(x.median()), axis=0)
  Y_dts_test = testdata_dts['DTS']
  
  """Testing regressors
  """
  i = 0
  rme_errors = []
  for estimator in estimators:
    
    print('-----------------------{} MODEL-----------------------'.format(estimator_names[i]))

    estimator.fit(X_dts_train,
                  Y_dts_train.values.ravel(),
                  early_stopping_rounds=100,
                  eval_set=[(X_dts_test, Y_dts_test)],
                  verbose=100
                  ) # fitting while validationg on open test set

    """Predicting DTS on training, validation, and test sets and computing metrics
    """
    train_pred = estimator.predict(X_dts_train)
    val_pred = estimator.predict(X_dts_val)
    test_pred = estimator.predict(X_dts_test)
    
    varinace_train = explained_variance_score(Y_dts_train, train_pred)
    max_eror_train = max_error(Y_dts_train, train_pred)
    ms_error_train = np.sqrt(mean_squared_error(Y_dts_train, train_pred))
    mabs_error_train = mean_absolute_error(Y_dts_train, train_pred)
    r2_train = r2_score(Y_dts_train, train_pred)

    varinace_val = explained_variance_score(Y_dts_val, val_pred)
    max_eror_val = max_error(Y_dts_val, val_pred)
    ms_error_val = np.sqrt(mean_squared_error(Y_dts_val, val_pred))
    mabs_error_val = mean_absolute_error(Y_dts_val, val_pred)
    r2_val = r2_score(Y_dts_val, val_pred)

    varinace_test = explained_variance_score(Y_dts_test, test_pred)
    max_eror_test = max_error(Y_dts_test, test_pred)
    ms_error_test = np.sqrt(mean_squared_error(Y_dts_test, test_pred))
    mabs_error_test = mean_absolute_error(Y_dts_test, test_pred)
    r2_test = r2_score(Y_dts_test, test_pred)

    rme_errors.append(ms_error_test) #storing root mean squared error for model comparition
    
    print('''--------TRAINING SET METRICS--------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_train, max_eror_train, ms_error_train, mabs_error_train, r2_train))
    
    print('''-------VALIDATION SET METRICS-------
    explianed varianve {},
    maximum error {}, 
    root mean squared error{}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_val, max_eror_val, ms_error_val, mabs_error_val, r2_val))

    print('''-----------TEST SET METRICS----------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_test, max_eror_test, ms_error_test, mabs_error_test, r2_test))

    i += 1

  # selecting bet model to perform imputation
  selected_model_index = rme_errors.index(min(rme_errors)) 
  print('TRAINING BEST MODEL TO PERFORM DTS IMPUTATION {}'.format(estimator_names[selected_model_index]))
  model1000 = estimators[selected_model_index]
  model1000.fit(X_dts_train,
                Y_dts_train.values.ravel(),
                early_stopping_rounds=100,
                eval_set=[(X_dts_test, Y_dts_test)],
                verbose=100) # fitting best ML regresor

  # filling nan values before predicting DTS
  X_train_DTS = traindata.drop(['LITHO', 'DTS'], axis=1)
  X_train_DTS2 = X_train_DTS.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_DTS = testdata.drop(['LITHO', 'DTS'], axis=1)
  X_test_DTS2 = X_test_DTS.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_DTS = hiddendata.drop(['LITHO', 'DTS'], axis=1)
  X_hidden_DTS2 = X_hidden_DTS.apply(lambda x: x.fillna(x.median()), axis=0)

  # predicting DTS on complete datasets
  traindata['DTS_pred'] = model1000.predict(X_train_DTS2)
  testdata['DTS_pred'] = model1000.predict(X_test_DTS2)
  hiddendata['DTS_pred'] = model1000.predict(X_hidden_DTS2)

  # imputing nan values in DTS by ML predictions
  traindata['DTS_COMB'] = traindata['DTS']
  traindata['DTS_COMB'].fillna(traindata['DTS_pred'], inplace=True)

  testdata['DTS_COMB'] = testdata['DTS']
  testdata['DTS_COMB'].fillna(testdata['DTS_pred'], inplace=True)

  hiddendata['DTS_COMB'] = hiddendata['DTS']
  hiddendata['DTS_COMB'].fillna(hiddendata['DTS_pred'], inplace=True)

  # predicting NPHI
  print('-----------------------------------------PREDINCTING NPHI------------------------------------------')
  """Getting data containing DTS readings and 
  spliting into a new training and validation subsets.
  """

  traindata_nphi = traindata[traindata.NPHI.notna()]
  X_nphi = traindata_nphi.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  Y_nphi = traindata_nphi['NPHI']

  """Inputing remanent features by their median.
  """

  X_nphi_inp = X_nphi.apply(lambda x: x.fillna(x.median()), axis=0) # imputation
  X_nphi_train, X_nphi_val, Y_nphi_train, Y_nphi_val = train_test_split(X_nphi_inp,
                                                                        Y_nphi,
                                                                        test_size=0.3,
                                                                        random_state=42) # train-validation split
  """Getting NPHI testing set from open test set
  """
  testdata_nphi = testdata[testdata.NPHI.notna()]
  X_nphi = testdata_nphi.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_nphi_test = X_nphi.apply(lambda x: x.fillna(x.median()), axis=0)
  Y_nphi_test = testdata_nphi['NPHI']

  print('Training set sahpe {} and validation set shape {} and test set shape'.format(X_nphi_train.shape,
                                                                                      X_nphi_val.shape,
                                                                                      X_nphi_test.shape))
  """Teting regressors
  """
  i = 0
  rme_errors = []
  for estimator in estimators:
    
    print('-----------------------{} MODEL-----------------------'.format(estimator_names[i]))
  
    estimator.fit(X_nphi_train,
                  Y_nphi_train.values.ravel(),
                  early_stopping_rounds=100,
                  eval_set=[(X_nphi_test, Y_nphi_test)],
                  verbose=100)
    """Predicting DTS on training, validation, and test sets and computing metrics
    """
    train_pred = estimator.predict(X_nphi_train)
    val_pred = estimator.predict(X_nphi_val)
    test_pred = estimator.predict(X_nphi_test)

    varinace_train = explained_variance_score(Y_nphi_train, train_pred)
    max_eror_train = max_error(Y_nphi_train, train_pred)
    ms_error_train = np.sqrt(mean_squared_error(Y_nphi_train, train_pred))
    mabs_error_train = mean_absolute_error(Y_nphi_train, train_pred)
    r2_train = r2_score(Y_nphi_train, train_pred)

    varinace_val = explained_variance_score(Y_nphi_val, val_pred)
    max_eror_val = max_error(Y_nphi_val, val_pred)
    ms_error_val = np.sqrt(mean_squared_error(Y_nphi_val, val_pred))
    mabs_error_val = mean_absolute_error(Y_nphi_val, val_pred)
    r2_val = r2_score(Y_nphi_val, val_pred)

    varinace_test = explained_variance_score(Y_nphi_test, test_pred)
    max_eror_test = max_error(Y_nphi_test, test_pred)
    ms_error_test = np.sqrt(mean_squared_error(Y_nphi_test, test_pred))
    mabs_error_test = mean_absolute_error(Y_nphi_test, test_pred)
    r2_test = r2_score(Y_nphi_test, test_pred)

    rme_errors.append(ms_error_test) # storing root mean squared error for model comparition

    print('''--------TRAINING SET METRICS--------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_train, max_eror_train, ms_error_train, mabs_error_train, r2_train))
    
    print('''-------VALIDATION SET METRICS-------
    explianed varianve {},
    maximum error {}, 
    root mean squared error{}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_val, max_eror_val, ms_error_val, mabs_error_val, r2_val))

    print('''-----------TEST SET METRICS----------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_test, max_eror_test, ms_error_test, mabs_error_test, r2_test))

    i += 1
    
  # selecting bet model to perform imputation
  selected_model_index = rme_errors.index(min(rme_errors))
  print('TRAINING BEST MODEL TO PERFORM NPHI IMPUTATION {}'.format(estimator_names[selected_model_index]))
  model2000 = estimators[selected_model_index]
  model2000.fit(X_nphi_train,
                Y_nphi_train.values.ravel(),
                early_stopping_rounds=100,
                eval_set=[(X_nphi_test, Y_nphi_test)],
                verbose=100) # fitting bets ML regressor

  # filling nan values before predicting NPHI
  X_train_NPHI = traindata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_train_NPHI2 = X_train_NPHI.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_NPHI = testdata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_test_NPHI2 = X_test_NPHI.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_NPHI = hiddendata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI'], axis=1)
  X_hidden_NPHI2 = X_hidden_NPHI.apply(lambda x: x.fillna(x.median()), axis=0)

  # predicting NPHI on complete datasets
  traindata['NPHI_pred'] = model2000.predict(X_train_NPHI2)
  testdata['NPHI_pred'] = model2000.predict(X_test_NPHI2)
  hiddendata['NPHI_pred'] = model2000.predict(X_hidden_NPHI2)

  # inputing nan values in NPHI by ML predictions
  traindata['NPHI_COMB'] = traindata['NPHI']
  traindata['NPHI_COMB'].fillna(traindata['NPHI_pred'], inplace=True)

  testdata['NPHI_COMB'] = testdata['NPHI']
  testdata['NPHI_COMB'].fillna(testdata['NPHI_pred'], inplace=True)

  hiddendata['NPHI_COMB'] = hiddendata['NPHI']
  hiddendata['NPHI_COMB'].fillna(hiddendata['NPHI_pred'], inplace=True)

  # predicting RHOB 
  print('----------------------------------------PREDINCTING RHOB-----------------------------------------')
  
  """Getting data containing RHOB readings and 
  spliting into a new training and validation subsets.
  """

  traindata_rhob = traindata[traindata.RHOB.notna()]
  X_rhob = traindata_rhob.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  Y_rhob = traindata_rhob['RHOB']

  """Inputing remanent features by their median.
  """

  X_rhob_inp = X_rhob.apply(lambda x: x.fillna(x.median()), axis=0) #imputation
  X_rhob_train, X_rhob_val, Y_rhob_train, Y_rhob_val = train_test_split(X_rhob_inp,
                                                                        Y_rhob,
                                                                        test_size=0.3,
                                                                        random_state=42) # train-validation split
  """Getting RHOB testing set from open test set
  """
  testdata_rhob = testdata[testdata.RHOB.notna()]
  X_rhob = testdata_rhob.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_rhob_test = X_rhob.apply(lambda x: x.fillna(x.median()), axis=0)
  Y_rhob_test = testdata_rhob['RHOB']

  print('Training set sahpe {}, validation set shape {}, and test shape {}'.format(X_rhob_train.shape,
                                                                                   X_rhob_val.shape,
                                                                                   X_rhob_test.shape))
  """Testing regressors
  """
  i = 0
  rme_errors = []
  for estimator in estimators:
    
    print('-----------------------{} MODEL-----------------------'.format(estimator_names[i]))
    estimator.fit(X_rhob_train,
                  Y_rhob_train.values.ravel(),
                  early_stopping_rounds=100,
                  eval_set=[(X_rhob_test, Y_rhob_test)],
                  verbose=100)

    """Predicitng RHOB on training, validation, and test sets and computing metrics
    """
    train_pred = estimator.predict(X_rhob_train)
    val_pred = estimator.predict(X_rhob_val)
    test_pred = estimator.predict(X_rhob_test)
    
    varinace_train = explained_variance_score(Y_rhob_train, train_pred)
    max_eror_train = max_error(Y_rhob_train, train_pred)
    ms_error_train = np.sqrt(mean_squared_error(Y_rhob_train, train_pred))
    mabs_error_train = mean_absolute_error(Y_rhob_train, train_pred)
    r2_train = r2_score(Y_rhob_train, train_pred)

    varinace_val = explained_variance_score(Y_rhob_val, val_pred)
    max_eror_val = max_error(Y_rhob_val, val_pred)
    ms_error_val = np.sqrt(mean_squared_error(Y_rhob_val, val_pred))
    mabs_error_val = mean_absolute_error(Y_rhob_val, val_pred)
    r2_val = r2_score(Y_rhob_val, val_pred)

    varinace_test = explained_variance_score(Y_rhob_test, test_pred)
    max_eror_test = max_error(Y_rhob_test, test_pred)
    ms_error_test = np.sqrt(mean_squared_error(Y_rhob_test, test_pred))
    mabs_error_test = mean_absolute_error(Y_rhob_test, test_pred)
    r2_test = r2_score(Y_rhob_test, test_pred)

    rme_errors.append(ms_error_test) # storing root mean squared error for model comparition

    print('''--------TRAINING SET METRICS--------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_train, max_eror_train, ms_error_train, mabs_error_train, r2_train))
    
    print('''-------VALIDATION SET METRICS-------
    explianed varianve {},
    maximum error {}, 
    root mean squared error{}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_val, max_eror_val, ms_error_val, mabs_error_val, r2_val))

    print('''-----------TEST SET METRICS----------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_test, max_eror_test, ms_error_test, mabs_error_test, r2_test))

    i += 1

  #selecting best model to perform imputation
  selected_model_index = rme_errors.index(min(rme_errors))
  print('TRAINING BEST MODEL TO PERFORM RHOB IMPUTATION {}'.format(estimator_names[selected_model_index]))
  model4000 = estimators[selected_model_index]
  
  model4000.fit(X_rhob_train,
                Y_rhob_train.values.ravel(),
                early_stopping_rounds=100,
                eval_set=[(X_rhob_test, Y_rhob_test)],
                verbose=100) #fitting best ML regressor

  # filling nan values before predicting RHOB
  X_train_RHOB = traindata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_train_RHOB2 = X_train_RHOB.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_RHOB = testdata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_test_RHOB2 = X_test_RHOB.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_RHOB = hiddendata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB'], axis=1)
  X_hidden_RHOB2 = X_hidden_RHOB.apply(lambda x: x.fillna(x.median()), axis=0)

  # predicting RHOB on complete datasets
  traindata['RHOB_pred'] = model4000.predict(X_train_RHOB2)
  testdata['RHOB_pred'] = model4000.predict(X_test_RHOB2)
  hiddendata['RHOB_pred'] = model4000.predict(X_hidden_RHOB2)

  # imputing nan values in RHOB by ML predictions
  traindata['RHOB_COMB'] = traindata['RHOB']
  traindata['RHOB_COMB'].fillna(traindata['RHOB_pred'], inplace=True)

  testdata['RHOB_COMB'] = testdata['RHOB']
  testdata['RHOB_COMB'].fillna(testdata['RHOB_pred'], inplace=True)

  hiddendata['RHOB_COMB'] = hiddendata['RHOB']
  hiddendata['RHOB_COMB'].fillna(hiddendata['RHOB_pred'], inplace=True)

  # predicting DTC
  print('----------------------------------------PREDINCTING DTC-----------------------------------------')

  """Getting data containing DTC readings and 
  spliting into a new training and validation subsets.
  """
  traindata_dtc = traindata[traindata.DTC.notna()]
  X_dtc = traindata_dtc.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  Y_dtc = traindata_dtc['DTC']

  """Inputing remanent features by their median.
  """

  X_dtc_inp = X_dtc.apply(lambda x: x.fillna(x.median()), axis=0) #imputation
  X_dtc_train, X_dtc_val, Y_dtc_train, Y_dtc_val = train_test_split(X_dtc_inp,
                                                                    Y_dtc,
                                                                    test_size=0.3,
                                                                    random_state=42) #train_validation split
  """Getting DTC testing set from open test set
  """
  testdata_dtc = testdata[testdata.DTC.notna()]
  X_dtc = testdata_dtc.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_dtc_test = X_dtc.apply(lambda x: x.fillna(x.median()), axis=0)
  Y_dtc_test = testdata_dtc['DTC']

  print('Training set sahpe {}, validation set shape {}, and test shape {}'.format(X_dtc_train.shape,
                                                                                   X_dtc_val.shape,
                                                                                   X_dtc_test.shape))

  """Testing regressors
  """
  i = 0
  rme_errors = []
  for estimator in estimators:
    
    print('-----------------------{} MODEL-----------------------'.format(estimator_names[i]))
    estimator.fit(X_dtc_train,
                  Y_dtc_train.values.ravel(),
                  early_stopping_rounds=100,
                  eval_set=[(X_dtc_test, Y_dtc_test)],
                  verbose=100)
    
    train_pred = estimator.predict(X_dtc_train)
    val_pred = estimator.predict(X_dtc_val)
    test_pred = estimator.predict(X_dtc_test)
    
    varinace_train = explained_variance_score(Y_dtc_train, train_pred)
    max_eror_train = max_error(Y_dtc_train, train_pred)
    ms_error_train = np.sqrt(mean_squared_error(Y_dtc_train, train_pred))
    mabs_error_train = mean_absolute_error(Y_dtc_train, train_pred)
    r2_train = r2_score(Y_dtc_train, train_pred)

    varinace_val = explained_variance_score(Y_dtc_val, val_pred)
    max_eror_val = max_error(Y_dtc_val, val_pred)
    ms_error_val = np.sqrt(mean_squared_error(Y_dtc_val, val_pred))
    mabs_error_val = mean_absolute_error(Y_dtc_val, val_pred)
    r2_val = r2_score(Y_dtc_val, val_pred)

    varinace_test = explained_variance_score(Y_dtc_test, test_pred)
    max_eror_test = max_error(Y_dtc_test, test_pred)
    ms_error_test = np.sqrt(mean_squared_error(Y_dtc_test, test_pred))
    mabs_error_test = mean_absolute_error(Y_dtc_test, test_pred)
    r2_test = r2_score(Y_dtc_test, test_pred)

    rme_errors.append(ms_error_test) # storing root mean squared error for model comparition

    print('''--------TRAINING SET METRICS--------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_train, max_eror_train, ms_error_train, mabs_error_train, r2_train))
    
    print('''-------VALIDATION SET METRICS-------
    explianed varianve {},
    maximum error {}, 
    root mean squared error{}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_val, max_eror_val, ms_error_val, mabs_error_val, r2_val))

    print('''-----------TEST SET METRICS----------
    explianed varianve {},
    maximum error {}, 
    root mean squared error {}, 
    maximum absolute error {},
    R2 {}'''.format(varinace_test, max_eror_test, ms_error_test, mabs_error_test, r2_test))

    i += 1
  # selecting bet model to perform imputation
  selected_model_index = rme_errors.index(min(rme_errors))
  print('TRAINING BEST MODEL TO PERFORM DTC IMPUTATION {}'.format(estimator_names[selected_model_index]))
  model3000 = estimators[selected_model_index]
  model3000.fit(X_dtc_train,
                Y_dtc_train.values.ravel(),
                early_stopping_rounds=100,
                eval_set=[(X_dtc_test, Y_dtc_test)],
                verbose=100) # fitting best ML regressor

  # filling nan values before predicting DTC
  X_train_DTC = traindata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_train_DTC2 = X_train_DTC.apply(lambda x: x.fillna(x.median()), axis=0)

  X_test_DTC = testdata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_test_DTC2 = X_test_DTC.apply(lambda x: x.fillna(x.median()), axis=0)

  X_hidden_DTC = hiddendata.drop(['LITHO', 'DTS', 'DTS_pred', 'NPHI', 'NPHI_pred', 'RHOB', 'RHOB_pred', 'DTC'], axis=1)
  X_hidden_DTC2 = X_hidden_DTC.apply(lambda x: x.fillna(x.median()), axis=0)

  # predict DTC on complete datasets
  traindata['DTC_pred'] = model3000.predict(X_train_DTC2)
  testdata['DTC_pred'] = model3000.predict(X_test_DTC2)
  hiddendata['DTC_pred'] = model3000.predict(X_hidden_DTC2)

  # imputing nan values in DTC by ML predictions
  traindata['DTC_COMB'] = traindata['DTC']
  traindata['DTC_COMB'].fillna(traindata['DTC_pred'], inplace=True)

  testdata['DTC_COMB'] = testdata['DTC']
  testdata['DTC_COMB'].fillna(testdata['DTC_pred'], inplace=True)

  hiddendata['DTC_COMB'] = hiddendata['DTC']
  hiddendata['DTC_COMB'].fillna(hiddendata['DTC_pred'], inplace=True)
  
  return traindata, testdata, hiddendata
