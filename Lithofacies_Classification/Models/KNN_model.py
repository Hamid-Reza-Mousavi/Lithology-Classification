
"""K-Nearest Neighbors machine-learning model

This script receives the clean datasets and trains a K-nearest neighbor
machine-learning model and test it on the clean open test and hidden test
datasets. The function retuns the lithofacies predictions obtained for the
training, open test, and hidden test sets.
"""

def run_KNN(train_scaled, test_scaled, hidden_scaled):

  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by K-nearesr Neighbors.

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
  train_pred_knn: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  test_pred_knn: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_pred_knn: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  from sklearn.model_selection import train_test_split
  from sklearn import neighbors

  # selected features to be used while training
  selectedfeatures_knn = ['GR', 'FORMATION_encoded', 'GROUP_encoded', 'NPHI_COMB', 'RHOB', 
                          'X_LOC', 'BS', 'CALI', 'SP', 'WELL_encoded', 'Z_LOC', 'DT_R', 'DEPTH_MD',
                          'DTC', 'Cluster']

  x_train = train_scaled[selectedfeatures_knn]
  y_train = train_scaled['LITHO']

  x_test = test_scaled[selectedfeatures_knn]
  y_test = test_scaled['LITHO']

  x_hidden = hidden_scaled[selectedfeatures_knn]
  y_hidden = hidden_scaled['LITHO']

  x_train_strat, X2, y_train_strat, Y2 = train_test_split(x_train,
                                                          y_train,
                                                          train_size=0.1,
                                                          shuffle=True,
                                                          stratify=y_train,
                                                          random_state=0
                                                          )
  # defining KNN mdoel with optimal hyper-parameters
  model_knn = neighbors.KNeighborsClassifier(n_neighbors=80, 
                                              weights='distance', 
                                              metric='manhattan'
                                              )
  # fitting a logistic regression model
  model_knn.fit(x_train_strat[selectedfeatures_knn], y_train_strat)

  # predicting
  train_pred_knn = model_knn.predict(x_train[selectedfeatures_knn])
  test_pred_knn = model_knn.predict(x_test[selectedfeatures_knn])
  hidden_pred_knn = model_knn.predict(x_hidden[selectedfeatures_knn])

  return train_pred_knn, test_pred_knn, hidden_pred_knn