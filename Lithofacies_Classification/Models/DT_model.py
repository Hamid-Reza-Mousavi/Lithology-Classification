
"""Decision Tree machine-learning model

This script receives the clean datasets and trains a decision tree machine
learning model and test it on the clean open test and hidden test datasets. 
The function returns the lithofacies predictions obtained for the training,
open test, and hidden test sets.
"""

def run_DT(train_scaled, test_scaled, hidden_scaled):
    
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a decision tree.

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
  train_pred_dtp: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  open_pred_dtp: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_dtp: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from sklearn.model_selection import train_test_split
  from sklearn.tree import DecisionTreeClassifier
  x_train = train_scaled.drop(['LITHO'], axis=1)
  y_train = train_scaled['LITHO']

  x_test = test_scaled.drop(['LITHO'], axis=1)
  y_test = test_scaled['LITHO']

  x_hidden = hidden_scaled.drop(['LITHO'], axis=1)
  y_hidden = hidden_scaled['LITHO']
  
  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train,
                                                          y_train,
                                                          train_size=0.1,
                                                          shuffle=True,
                                                          stratify=y_train,
                                                          random_state=0
                                                          )
  # defining DT model after pruning
  tunned_dt = DecisionTreeClassifier(max_depth=15,
                                     ccp_alpha=0.002
                                     )

  # fitting the decision tree model
  tunned_dt.fit(x_train_strat, y_train_strat)

  # predicting
  train_pred_dtp = tunned_dt.predict(x_train)
  open_pred_dtp = tunned_dt.predict(x_test)
  hidden_pred_dtp = tunned_dt.predict(x_hidden)

  return train_pred_dtp, open_pred_dtp, hidden_pred_dtp
