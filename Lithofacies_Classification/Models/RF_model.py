
"""Random Forest machine-learning model

This script receives the clean datasets and trains a random forest machine
learning model and test it on the clean open test and hidden test datasets. 
The function returns the lithofacies predictions obtained for the training,
open test, and hidden test sets.
"""

def run_RF(train_scaled, test_scaled, hidden_scaled):
      
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a random forest.

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
  train_pred_rf: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  open_pred_rf: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_rf: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier

  # selected features to be used while training
  features_selected_rf = ['RDEP', 'GR', 'NPHI_COMB', 'G', 'P_I', 'S_I', 'DTC', 'DTS_COMB',
                        'RSHA', 'DT_R', 'RHOB', 'K', 'DCAL', 'Y_LOC', 'GROUP_encoded',
                        'WELL_encoded', 'FORMATION_encoded', 'DEPTH_MD', 'Z_LOC', 'CALI',
                        'X_LOC', 'RMED', 'PEF', 'SP', 'MD_TVD', 'ROP', 'DRHO']

  x_train = train_scaled[features_selected_rf]
  y_train = train_scaled['LITHO']

  x_test = test_scaled[features_selected_rf]
  y_test = test_scaled['LITHO']

  x_hidden = hidden_scaled[features_selected_rf]
  y_hidden = hidden_scaled['LITHO']

  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train,
                                                          y_train,
                                                          train_size=0.5,
                                                          shuffle=True,
                                                          stratify=y_train,
                                                          random_state=0)
  # defining a RF model with the optimal hyper-parameters
  model_rf = RandomForestClassifier(n_estimators=350,
                                    bootstrap=False,
                                    max_depth=45,
                                    max_features='sqrt'
                                    )

  # fitting the random forest model
  model_rf.fit(x_train_strat[features_selected_rf], y_train_strat.values.ravel())

  # predicting
  train_pred_rf = model_rf.predict(x_train[features_selected_rf])
  open_pred_rf = model_rf.predict(x_test[features_selected_rf])
  hidden_pred_rf = model_rf.predict(x_hidden[features_selected_rf])

  return train_pred_rf, open_pred_rf, hidden_pred_rf
