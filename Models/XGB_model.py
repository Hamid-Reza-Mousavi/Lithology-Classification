
"""eXtreme gradient boosting tree-based machine-learning model

This script receives the clean datasets and trains an extreme gradient boosting
tree-based machine learning model and test it on the clean open test and hidden
test datasets. The function returns the lithofacies predictions obtained for 
the training, open test, and hidden test sets.
"""

def run_XGB(train_norm, test_norm, hidden_norm):
          
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a extreme gradient boosting
  tree-based model, XGB.

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
  train_pred_xgb1: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  open_pred_xgb1: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_xgb1: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from xgboost import XGBClassifier
  from sklearn.model_selection import StratifiedKFold
  import numpy as np
  import pandas as pd
  from sklearn.metrics import accuracy_score
 
  # selected features to be used while training
  selected_fetures_xgb = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'P_I', 'DTC', 'DTS_COMB', 'RSHA',
                          'DT_R', 'RHOB', 'K', 'DCAL', 'Y_LOC', 'Cluster', 'GROUP_encoded',
                          'WELL_encoded', 'FORMATION_encoded', 'DEPTH_MD', 'Z_LOC', 'CALI', 'BS',
                          'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD', 'RMIC', 'DRHO']

  x_train = train_norm[selected_fetures_xgb]
  y_train = train_norm['LITHO']

  x_test = test_norm[selected_fetures_xgb]
  y_test = test_norm['LITHO']

  x_hidden = hidden_norm[selected_fetures_xgb]
  y_hidden = hidden_norm['LITHO']

  """The model is trained on 10 stratified k-folds, also uses the open set as 
  validation set to avoid overfitting and a 100-round early stopping callback.

  The model uses a multi-soft_probability objective function which returns the
  probabilities predicted for each class. This probabilities are computed and
  stacked by using each k-fold to give the final prediction.

  """

  split = 10
  kf = StratifiedKFold(n_splits=split, shuffle=True)

  train_prob_xgb1 = np.zeros((len(x_train), 12))
  open_prob_xgb1 = np.zeros((len(x_test), 12))
  hidden_prob_xgb1 = np.zeros((len(x_hidden), 12))

  xgbmodel_noarg = XGBClassifier(n_estimators=1000, max_depth=4,
                                 booster='gbtree', objective='multi:softprob',
                                 learning_rate=0.075, random_state=42,
                                 subsample=1, colsample_bytree=1,
                                 tree_method='gpu_hist', predictor='gpu_predictor',
                                 verbose=2020, reg_lambda=1500
                                 )
  i = 1
  for (train_index, test_index) in kf.split(x_train, y_train):
    X_train, X_test = x_train.iloc[train_index], x_train.iloc[test_index]
    Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]

    xgbmodel_noarg.fit(X_train,
                       Y_train.values.ravel(),
                       early_stopping_rounds=100,
                       eval_set=[(X_test, Y_test)],
                       verbose=100
                       )
    
    prediction = xgbmodel_noarg.predict(X_test)
    print('Fold accuracy:', accuracy_score(Y_test, prediction))

    print(f'-----------------------FOLD {i}---------------------')
    
    # stacking probabilities
    train_prob_xgb1 += xgbmodel_noarg.predict_proba(x_train)
    open_prob_xgb1 += xgbmodel_noarg.predict_proba(x_test)
    hidden_prob_xgb1 += xgbmodel_noarg.predict_proba(x_hidden)

    i += 1

  # final lithology class prediction
  train_prob_xgb1 = pd.DataFrame(train_prob_xgb1/split)
  train_pred_xgb1 = np.array(pd.DataFrame(train_prob_xgb1).idxmax(axis=1))

  open_prob_xgb1 = pd.DataFrame(open_prob_xgb1/split)
  open_pred_xgb1 = np.array(pd.DataFrame(open_prob_xgb1).idxmax(axis=1))

  hidden_prob_xgb1 = pd.DataFrame(hidden_prob_xgb1/split)
  hidden_pred_xgb1 = np.array(pd.DataFrame(hidden_prob_xgb1).idxmax(axis=1))

  return train_pred_xgb1, open_pred_xgb1, hidden_pred_xgb1