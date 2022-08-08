
"""Data scaler

This script standardize the training, open test, and hidden test sets
after the datasets have been pre-process and machine-learning augmented.
It provides the datastes ready to be used in any machine-learning model.

"""

def standardscaler(traindata, testdata, hiddendata):

  """Returns the starndardize training, open test, and hidden test
  dataframes once they have been pre-processed and augmented.

  Parameters
  ----------
  traindata: Dataframe
    Augmented trainig dataframe.
  testdata: Dataframe
    Augmented open test dataframe.
  hiddendata: Dataframe
    Augmented hidden test dataframe.

  Returns
  ----------
  cleaned_traindata: Dataframe
    Starndardized training dataframe.
  cleaned_testdata: Dataframe
    Starndardized open test dataframe.
  cleaned_hiddendata: Dataframe
    Starndardized hidden test dataframe.
  """

  import pandas as pd
  from sklearn.preprocessing import StandardScaler

  """The features that were not augmented
  by ML are inputed by the median and then standardized.
  """

  train_features = traindata.drop(['LITHO'], axis=1);   train_labels = traindata['LITHO']
  test_features = testdata.drop(['LITHO'], axis=1);     test_labels = testdata['LITHO']
  hidden_features = hiddendata.drop(['LITHO'], axis=1); hidden_labels = hiddendata['LITHO']

  #Imputng features by median
  train_features_inp = train_features.apply(lambda x: x.fillna(x.median()), axis=0)
  test_features_inp = test_features.apply(lambda x: x.fillna(x.median()), axis=0)
  hidden_features_inp = hidden_features.apply(lambda x: x.fillna(x.median()), axis=0)

  """Normalizing features on each dataset
  """

  n = train_features_inp.shape[1]
  std = StandardScaler()
  x_train_std = train_features_inp.copy()
  x_test_std = test_features_inp.copy()
  x_hidden_std = hidden_features_inp.copy()

  x_train_std.iloc[:,:n] = std.fit_transform(x_train_std.iloc[:,:n])
  x_test_std.iloc[:,:n] = std.transform(x_test_std.iloc[:,:n])
  x_hidden_std.iloc[:,:n] = std.transform(x_hidden_std.iloc[:,:n])

  """Concatenating features and targets
  """

  scaled_traindata = pd.concat([x_train_std, train_labels], axis=1)
  scaled_testdata = pd.concat([x_test_std, test_labels], axis=1)
  scaled_hiddendata = pd.concat([x_hidden_std, hidden_labels], axis=1)

  return scaled_traindata, scaled_testdata, scaled_hiddendata