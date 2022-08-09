
"""Neural Network model

This script receives the clean datasets and trains neural network and test it
on the clean open test and hidden test datasets. The function returns the
lithofacies predictions obtained for the training, open test, and hidden test sets.
"""

def run_NN(train_scaled, test_scaled, hidden_scaled):
            
  """Returns the predicted lithology classes for the training,
  open test, and hidden test obtained by a bayesian optimized
  two-hidden layer neural network.

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
  train_nn2: one-dimentional array
    Predicted lithology classes obtained from the training dataset.
  open_nn2: one-dimentional array
    Predicted lithology classes obtained from the open test dataset.
  hidden_nn2: one-dimentional array
    Predicted lithology classes obtained from the hidden test dataset.
  """

  # importing dependencies
  import numpy as np
  import pandas as pd
  import tensorflow as tf
  from tensorflow import keras
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.layers import Dense, Activation
  from tensorflow.keras.layers import InputLayer, Input
  from tensorflow.keras.layers import Dense, Dropout, Activation
  from tensorflow.keras.callbacks import EarlyStopping
  from tensorflow.keras.optimizers import SGD
  
  # selected features to be used while training
  features_selected_nn = ['GROUP_encoded', 'GR', 'NPHI_COMB', 'Y_LOC', 'RHOB',
                        'DEPTH_MD', 'FORMATION_encoded', 'Z_LOC', 'WELL_encoded', 'X_LOC',
                        'RMED', 'CALI', 'DTC', 'MD_TVD', 'DT_R',
                        'PEF', 'RDEP', 'DTS_COMB', 'G', 'SP',
                        'Cluster', 'K', 'P_I', 'DRHO', 'DCAL'
                          ]
  """hyper-parameters optimized through a Bayesian optimization process.
  """

  learning_rate = 0.1
  num_layers = 2
  num_nodes = 512
  activation = 'sigmoid'

  """Structuring a two-hidden layer feed fordward neural network
  """
  
  x_train_nn = tf.convert_to_tensor(train_scaled[features_selected_nn])
  x_test_nn = tf.convert_to_tensor(test_scaled[features_selected_nn])
  x_hidden_nn = tf.convert_to_tensor(hidden_scaled[features_selected_nn])

  opt_model = Sequential()
  opt_model.add(InputLayer(input_shape=(x_train_nn.shape[1])))
  opt_model.add(Dropout(0.1))

  opt_model.add(Dense(num_nodes,
                      activation=activation,
                      kernel_initializer='random_normal')) #input layer
  
  opt_model.add(Dropout(0.7)) # drop out layer for regularization

  opt_model.add(Dense(num_nodes,
                      activation=activation,
                      kernel_initializer='random_normal')) # hidden layer 1
  
  opt_model.add(Dense(12,
                      activation='softmax',
                      kernel_initializer='random_normal')) # hidden layer 2

  # setting stochastic gradint descent optimizer
  optimizer = SGD(learning_rate=learning_rate, momentum=0.1) 
      
  opt_model.compile(optimizer=optimizer, 
                    loss='sparse_categorical_crossentropy',  
                    metrics=['accuracy']
                    ) # compiling model

  monitor = EarlyStopping(monitor='val_loss',
                          min_delta=1e-3,
                          patience=50, 
                          verbose=1,
                          mode='auto',
                          restore_best_weights=True
                          ) # early stopping callback to avoid overfitting

  """Fitting neural network while validation on the open test set
  """

  histories = opt_model.fit(x_train_nn, 
                            train_norm['LITHO'],
                            batch_size = 256,
                            validation_data = (x_test_nn, test_norm['LITHO']),
                            callbacks = [monitor],
                            verbose=1,
                            epochs=100
                            ) 
  
  """ Predicting lithology classes
  """

  nn_train_prob = opt_model.predict(x_train_nn)
  train_nn2 = np.array(pd.DataFrame(nn_train_prob).idxmax(axis=1))

  nn_open_prob = opt_model.predict(x_test_nn)
  open_nn2 = np.array(pd.DataFrame(nn_open_prob).idxmax(axis=1))

  nn_hidden_prob = opt_model.predict(x_hidden_nn)
  hidden_nn2 = np.array(pd.DataFrame(nn_hidden_prob).idxmax(axis=1))

  return train_nn2, open_nn2, hidden_nn2