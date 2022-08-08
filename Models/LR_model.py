
"""Logistic regression machine-learning model

This script trains a logistic regression machine learning model and test it on the
open test and hidden test dataset. The function retuns the lithofacies predictions 
obtained for the training, open test, and hidden test sets.
"""

def run_LR(train_scaled, test_scaled, hidden_scaled):

  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by Logistic Regression.

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
  train_pred_lr: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  test_pred_lr: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_lr: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from sklearn.model_selection import train_test_split
  from sklearn.linear_model import LogisticRegression

  # selected features to be used while training
  features_selected_lr = ['DTS_COMB', 'G', 'P_I', 'GR','NPHI_COMB', 
                          'DTC', 'RHOB', 'DT_R', 'Z_LOC', 'S_I','K'
                          ]

  x_train = train_scaled[features_selected_lr]
  y_train = train_scaled['LITHO']

  x_test = test_scaled[features_selected_lr]
  y_test = test_scaled['LITHO']

  x_hidden = hidden_scaled[features_selected_lr]
  y_hidden = hidden_scaled['LITHO']
  
  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train,
                                                          y_train,
                                                          train_size=0.1,
                                                          shuffle=True,
                                                          stratify=y_train,
                                                          random_state=0)
  # difining model with optimal hyper-parameters
  model_lr = LogisticRegression(C=0.1,
                                solver='saga',
                                max_iter=4000,
                                verbose=1
                                )
  
  # fitting a logistic regression model
  model_lr.fit(x_train_strat[features_selected_lr], y_train_strat) 

  # predicting 
  train_pred_lr = model_lr.predict(x_train[features_selected_lr])
  test_pred_lr = model_lr.predict(x_test[features_selected_lr])
  hidden_pred_lr = model_lr.predict(x_hidden[features_selected_lr])

  return train_pred_lr, test_pred_lr, hidden_pred_lr