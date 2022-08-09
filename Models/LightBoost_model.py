
"""Light gradient boosting tree-based machine-learning model

This script receives the clean datasets and trains a light gradient boosting
tree-based machine learning model and test it on the clean open test and hidden
test datasets. The function returns the lithofacies predictions obtained for 
the training, open test, and hidden test sets.
"""

def run_LightBoost(train_scaled, test_scaled, hidden_scaled):
        
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a light gradient boosting
  tree-based model, LGBM.

  Parameters
  ----------
  cleaned_traindata: Dataframe
    Starndardized training dataframe.
  cleaned_testdata: Dataframe
    Starndardized open test dataframe.
  cleaned_hiddendata: Dataframe
    Starndardized hidden test dataframe.

  Returns
  ----------
  train_pred_light: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  open_pred_light: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_light: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from lightgbm import LGBMClassifier

  selected_features_lightboost = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'DTC', 'DTS_COMB', 'RSHA', 'DT_R',
                                'RHOB', 'K', 'DCAL', 'Y_LOC', 'GROUP_encoded', 'WELL_encoded',
                                'DEPTH_MD', 'Z_LOC', 'CALI', 'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD',
                                'ROP', 'DRHO']
                 
  x_train = train_scaled[selected_features_lightboost]
  y_train = train_scaled['LITHO']

  x_test = test_scaled[selected_features_lightboost]
  y_test = test_scaled['LITHO']

  x_hidden = hidden_scaled[selected_features_lightboost]
  y_hidden = hidden_scaled['LITHO']

  """The model is trained on 10 stratified k-folds, also uses 
  the open set as validation set to avoid overfitting
  and a 100-round early stopping callback.

  The model uses a multi-soft_probability objective function
  which returns the probabilities predicted for each class.
  This probabilities are computed and stacked by using each k-fold
  to give the final prediction.
  """
  # defining LGBM model with optimal hyper-parameters
  lightboost_model1 = LGBMClassifier(n_estimators=1000,
                                     learning_rate=0.015,
                                     random_state=42,max_depth=8,
                                     reg_lambda=250, verbose=-1,
                                     objective='multi:softprob',
                                     device='gpu', gpu_platform_id=1,
                                     gpu_device_id=0, silent=True
                                      )
  # fitting the LBGM model
  lightboost_model1.fit(x_train,
                        y_train.values.ravel(),
                        early_stopping_rounds=100,
                        eval_set=[(x_test, y_test)],
                        verbose=-100
                        )
  # predicting
  train_pred_light = lightboost_model1.predict(x_train)
  open_pred_light = lightboost_model1.predict(x_test)
  hidden_pred_light = lightboost_model1.predict(x_hidden)

  return train_pred_light, open_pred_light, hidden_pred_light
  