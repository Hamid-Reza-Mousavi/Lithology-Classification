
"""Categorical gradient boosting tree-based machine-learning model

This script receives the clean datasets and trains a categorical tree gradient
boosting machine learning model and test it on the clean open test and hidden
test datasets. The function returns the lithofacies predictions obtained for 
the training, open test, and hidden test sets.
"""

def run_CatBoost(train_scaled, test_scaled, hidden_scaled):
        
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a categorical tree-based
  gradient boosting model, CAT.

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
  train_pred_cat1: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  open_pred_cat1: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_cat1: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from sklearn.model_selection import StratifiedKFold
  from catboost import CatBoostClassifier
  from sklearn.metrics import accuracy_score
  import pandas as pd
  import numpy as np

  # selected features to be used while training
  selected_features_catboost = ['GR', 'NPHI_COMB', 'DTC', 'DTS_COMB','RHOB',
                                'Y_LOC', 'GROUP_encoded', 'WELL_encoded', 
                                'FORMATION_encoded', 'DEPTH_MD', 'Z_LOC', 'CALI',
                                'X_LOC', 'RMED', 'SP', 'MD_TVD']
                 
  x_train = train_scaled[selected_features_catboost]
  y_train = train_scaled['LITHO']

  x_test = test_scaled[selected_features_catboost]
  y_test = test_scaled['LITHO']

  x_hidden = hidden_scaled[selected_features_catboost]
  y_hidden = hidden_scaled['LITHO']

  """ The model is trained on 10 stratified k-folds, also uses the open set as
  validation set to avoid overfitting and a 100-round early stopping callback.

  The model uses a multi-soft_probability objective function which returns the
  probabilities predicted for each class. This probabilities are computed and
  stacked by using each k-fold to give the final prediction.
  """

  split = 10
  kf = StratifiedKFold(n_splits=split, shuffle=True)

  train_prob_cat1 = np.zeros((len(x_train), 12))
  open_prob_cat1 = np.zeros((len(x_test), 12))
  hidden_prob_cat1 = np.zeros((len(x_hidden), 12))

  catboost_model1 = CatBoostClassifier(iterations=1000, 
                               learning_rate=0.1,
                               depth = 6,
                               l2_leaf_reg = 300,
                               #border_count = 128,
                               #bagging_temperature = 10,
                               grow_policy = 'SymmetricTree',
                               task_type='GPU',
                               verbose=100)
  i = 1
  for (train_index, test_index) in kf.split(x_train, y_train):
    X_train, X_test = x_train.iloc[train_index], x_train.iloc[test_index]
    Y_train, Y_test = y_train.iloc[train_index], y_train.iloc[test_index]

    catboost_model1.fit(X_train,
                        Y_train.values.ravel(),
                        early_stopping_rounds=100,
                        eval_set=[(X_test, Y_test)],
                        verbose=100
                        )
    
    prediction = catboost_model1.predict(X_test)
    print('Fold accuracy:', accuracy_score(Y_test, prediction))

    print(f'-----------------------FOLD {i}---------------------')
    
    # staking predicted probabilities 
    train_prob_cat1 += catboost_model1.predict_proba(x_train)
    open_prob_cat1 += catboost_model1.predict_proba(x_test)
    hidden_prob_cat1 += catboost_model1.predict_proba(x_hidden)

    i += 1

  # getting final predicted classes
  train_prob_cat1 = pd.DataFrame(train_prob_cat1/split)
  train_pred_cat1 = np.array(pd.DataFrame(train_prob_cat1).idxmax(axis=1))

  open_prob_cat1 = pd.DataFrame(open_prob_cat1/split)
  open_pred_cat1 = np.array(pd.DataFrame(open_prob_cat1).idxmax(axis=1))

  hidden_prob_cat1 = pd.DataFrame(hidden_prob_cat1/split)
  hidden_pred_cat1 = np.array(pd.DataFrame(hidden_prob_cat1).idxmax(axis=1))

  return train_pred_cat1, open_pred_cat1, hidden_pred_cat1