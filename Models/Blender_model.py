def run_Blender(train_scaled, test_scaled, hidden_scaled):
          
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a votting model(xgb and catb).

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
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import StratifiedKFold
  from sklearn.ensemble import VotingClassifier
  from catboost import CatBoostClassifier
  from sklearn.utils.class_weight import compute_class_weight
  import numpy as np
  import pandas as pd
  from sklearn.metrics import accuracy_score, f1_score
 
  # selected features to be used while training
  #selected_fetures_xgb = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'P_I', 'DTC', 'DTS_COMB', 'RSHA',
  #                        'DT_R', 'RHOB', 'K', 'DCAL', 'Y_LOC', 'Cluster', 'GROUP_encoded',
  #                        'WELL_encoded', 'FORMATION_encoded', 'DEPTH_MD', 'Z_LOC', 'CALI', 'BS',
  #                        'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD', 'RMIC', 'DRHO']

  #x_train = train_scaled[selected_fetures_xgb]
  #y_train = train_scaled['LITHO']

  #x_test = test_scaled[selected_fetures_xgb]
  #y_test = test_scaled['LITHO']

  #x_hidden = hidden_scaled[selected_fetures_xgb]
  #y_hidden = hidden_scaled['LITHO']

  x_train = train_scaled.iloc[:, :-1]
  y_train = train_scaled.iloc[:, -1]

  x_test = test_scaled.iloc[:, :-1]
  y_test = test_scaled.iloc[:, -1]

  x_hidden = hidden_scaled.iloc[:, :-1]
  y_hidden = hidden_scaled.iloc[:, -1]
  class_weights = compute_class_weight(
                                          class_weight = "balanced",
                                          classes = np.unique(y_train),
                                          y = y_train                                                    
                                      )
  class_weights = dict(zip(np.unique(y_train), class_weights))
  """The model is trained on 10 stratified k-folds, also uses the open set as 
  validation set to avoid overfitting and a 100-round early stopping callback.

  The model uses a multi-soft_probability objective function which returns the
  probabilities predicted for each class. This probabilities are computed and
  stacked by using each k-fold to give the final prediction.

  """


  split = 10
  kf = StratifiedKFold(n_splits=split, shuffle=True, random_state=0)
  train_pred = np.zeros((len(x_train), 12))
  test_pred = np.zeros((len(x_test), 12))
  hidden_pred = np.zeros((len(x_hidden), 12))

  clf1 = XGBClassifier(n_estimators=1000, max_depth=4,
                                 booster='gbtree', objective='multi:softprob',
                                 learning_rate=0.075, random_state=42,
                                 subsample=1, colsample_bytree=1,
                                 tree_method='gpu_hist', predictor='gpu_predictor',
                                 verbose=2020, reg_lambda=1500
                                 )
  clf2 = CatBoostClassifier(iterations=1000, 
                               learning_rate=0.1,
                               depth = 6,
                               l2_leaf_reg = 300,
                               #border_count = 128,
                               #bagging_temperature = 10,
                               grow_policy = 'SymmetricTree',
                               task_type='GPU',
                               verbose=100, class_weights=class_weights)


  clf = VotingClassifier(estimators=[('xgb', clf1), ('catb', clf2)], voting='soft')

  i = 1
  for (train_index, test_index) in kf.split(pd.DataFrame(x_train), pd.DataFrame(y_train)):
    X_train, X_test = pd.DataFrame(x_train).iloc[train_index], pd.DataFrame(x_train).iloc[test_index]
    Y_train, Y_test = pd.DataFrame(y_train).iloc[train_index],pd.DataFrame(y_train).iloc[test_index]
    clf.fit(X_train, Y_train)
    prediction = clf.predict(X_test)
    print('Fold accuracy:', accuracy_score(Y_test, prediction))
    print(f'F1 is:', f1_score(prediction, Y_test, average="weighted"))
    print(f'-----------------------FOLD {i}---------------------')
    i+=1
    train_pred += clf.predict_proba(pd.DataFrame(x_train))
    test_pred += clf.predict_proba(pd.DataFrame(x_test))
    hidden_pred += clf.predict_proba(pd.DataFrame(x_hidden))

  train_pred= pd.DataFrame(train_pred/split)
  test_pred= pd.DataFrame(test_pred/split)
  hidden_pred= pd.DataFrame(hidden_pred/split)
  train_pred = np.array(pd.DataFrame(train_pred).idxmax(axis=1))
  test_pred = np.array(pd.DataFrame(test_pred).idxmax(axis=1))
  hidden_pred = np.array(pd.DataFrame(hidden_pred).idxmax(axis=1))
  print('---------------CROSS VALIDATION COMPLETE-----------------')

  return train_pred, test_pred, hidden_pred