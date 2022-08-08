
"""Data preprocessing

This script preprocess the training, open test, and hidden test sets. 
The preprocess involves droping unnecessary columns bases un presence percentage,
clustering well logs by location as features, and splitting again the data into
3 subsets. Clustering performed by unsipervides K-Means algorithm.
"""

def base_well_name(row):
    well_name = row['WELL']
    return well_name.split()[0]

def preprocess_data(training_formated, test_formated, hidden_formated):

  """Returns the training, open test, and hidden test dataframes with without 
  'SGR', 'ROPA', 'RXO', 'MUDWEIGHT', and including 'Cluster' as a feature.

  Parameters
  ----------
  training_formated: Dataframe
    Formated training dataframe.
  test_formated: Dataframe
    Formated open test dataframe.
  hidden_formated: Dataframe
    Formated hidden test dataframe.

  Returns
  ----------
  traindata_prep: Dataframe
    Pre-processed training dataframe.
  testdata_prep: Dataframe
    Pre-processed open test dataframe.
  hiddendata_prep: Dataframe
    Pre-processed hidden test dataframe.
  """

  import pandas as pd
  import numpy as np
  from sklearn.cluster import KMeans

  train_len = training_formated.shape[0] # storing datasets lenghts
  test_len = test_formated.shape[0]
  hidden_len = hidden_formated.shape[0]
  
  # concatenating datasets and dropping indexes
  df_concat = pd.concat((training_formated, 
                         test_formated,
                         hidden_formated)).reset_index(drop=True) 
  
  drop_cols = ['SGR', 'ROPA', 'RXO', 'MUDWEIGHT']
  df_drop = df_concat.drop(drop_cols, axis=1) # dropping unnecesary columns

  # encoding GROUP, FORMATION, and WELL
  df_drop['GROUP_encoded'] = df_drop['GROUP'].astype('category')
  df_drop['GROUP_encoded'] = df_drop['GROUP_encoded'].cat.codes

  df_drop['FORMATION_encoded'] = df_drop['FORMATION'].astype('category')
  df_drop['FORMATION_encoded'] = df_drop['FORMATION_encoded'].cat.codes

  df_drop['WELL_encoded'] = df_drop['WELL'].astype('category')
  df_drop['WELL_encoded'] = df_drop['WELL_encoded'].cat.codes

  # creating a well names dataframe
  training_wells = training_formated['WELL'].unique()
  test_wells = test_formated['WELL'].unique()
  hidden_wells= hidden_formated['WELL'].unique()

  well_names = np.concatenate((training_wells, test_wells, hidden_wells))
  well_names_df = pd.DataFrame({'WELL':well_names})

  # importing wells metadata
  well_meta_df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/wellbore_exploration_all.csv')
  well_meta_df.rename(columns={'wlbWellboreName': 'WELL', 
                               'wlbWell': 'WELL_HEAD', 
                               'wlbNsDecDeg': 'lat', 
                               'wlbEwDesDeg': 'lon', 
                               'wlbDrillingOperator': 'Drilling_Operator', 
                               'wlbPurposePlanned': 'Purpose', 
                               'wlbCompletionYear': 'Completion_Year', 
                               'wlbFormationAtTd': 'Formation'
                               }, inplace=True)
  
  well_locations_df = well_meta_df[['WELL_HEAD', 'lat', 'lon']].drop_duplicates(subset=['WELL_HEAD'])
  well_meta_df = well_meta_df[['WELL', 'Drilling_Operator', 'Purpose', 'Completion_Year', 'Formation']]

  well_names_df['WELL_HEAD'] = well_names_df.apply(lambda row: base_well_name(row), axis=1)
  locations_df = well_names_df.merge(well_locations_df, how='inner', on='WELL_HEAD')
  locations_df = locations_df.merge(well_meta_df, how='left', on='WELL')

  # labeling train and test wells in a new column
  locations_df.loc[locations_df['WELL'].isin(training_wells), 'Dataset'] = 'Train'
  locations_df.loc[locations_df['WELL'].isin(test_wells), 'Dataset'] = 'Test'
  locations_df.loc[locations_df['WELL'].isin(hidden_wells), 'Dataset'] = 'Hidden'

  LonLat_df =  locations_df.drop(['WELL', 
                                  'WELL_HEAD', 
                                  'Drilling_Operator', 
                                  'Purpose', 
                                  'Completion_Year', 
                                  'Formation', 
                                  'Dataset'], 
                                  axis=1)
  # applying K-Means 
  location = LonLat_df[['lon', 'lat']].values
  kmeans = KMeans(n_clusters=3, init='k-means++', random_state=1)
  labels = kmeans.fit_predict(location)

  # including Cluster as a feature
  df_drop = df_drop.rename(columns={'WELL':'Cluster'})
  clust_map = dict(zip(locations_df.WELL.values, labels))
  df_drop['Cluster'] = df_drop['Cluster'].map(clust_map)

  #dropping categorial features replaces beforehan by encoded features
  df_drop2 = df_drop.drop(['GROUP', 'FORMATION'], axis=1)

  # splitting dataset into training, test, and hidden sets
  traindata_prep = df_drop2[:train_len].copy()
  testdata_prep = df_drop2[train_len:(train_len+test_len)].copy()
  hiddendata_prep = df_drop2[(train_len+test_len):].copy()

  return traindata_prep, testdata_prep, hiddendata_prep
