import pandas as pd
def formating(
  training_raw:pd.DataFrame,
  test_raw:pd.DataFrame, 
  hidden_raw:pd.DataFrame
):

  """This script simply rename the column names in a consitent manner for the training, 
  open test, and hidden test sets. It also maps the lithofacies labesl with numbers
  from 0 to 11 and drops the intepretation confidence column.
  Returns the training, open test, and hidden test dataframes with consistent formats.

  Parameters
  ----------
  training_raw: Dataframe
    Raw training dataframe.
  test_raw: Dataframe
    Raw open test dataframe.
  hidden_raw: Dataframe
    Raw hidden test dataframe.

  Returns
  ----------
  training_formated: Dataframe
    Formated training dataframe.
  test_formated: Dataframe
    Formated open test dataframe.
  hidden_formated: Dataframe
    Formated hidden test dataframe.
  """
  
  from sklearn.preprocessing import LabelEncoder
  le = LabelEncoder()

  # formating raw training set
  training_formated = training_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  training_formated['LITHO'].replace([30000, 65000, 65030, 70000, 70032, 74000, 80000, 86000, 88000, 90000, 93000, 99000],
                                               ['SS', 'Sh', 'SS-Sh', 'Lims', 'Chlk', 'Dol', 'Marl', 'Anhy', 'Hal', 'Coal', 'Bsmt', 'Tuf'], inplace=True)                                      
  training_formated = training_formated.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
  training_formated['LITHO'] = le.fit_transform(training_formated["LITHO"])

  #formating raw test set
  test_formated = test_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  test_formated['LITHO'].replace([30000, 65000, 65030, 70000, 70032, 74000, 80000, 86000, 88000, 90000, 93000, 99000],
                                             ['SS', 'Sh', 'SS-Sh', 'Lims', 'Chlk', 'Dol', 'Marl', 'Anhy', 'Hal', 'Coal', 'Bsmt', 'Tuf'], inplace=True)
  test_formated['LITHO'] = le.transform(test_formated["LITHO"])

  # formating raw hidden set
  hidden_formated = hidden_raw.rename(columns={'FORCE_2020_LITHOFACIES_LITHOLOGY':'LITHO'})
  hidden_formated['LITHO'].replace([30000, 65000, 65030, 70000, 70032, 74000, 80000, 86000, 88000, 90000, 93000, 99000],
                                           ['SS', 'Sh', 'SS-Sh', 'Lims', 'Chlk', 'Dol', 'Marl', 'Anhy', 'Hal', 'Coal', 'Bsmt', 'Tuf'], inplace=True)
  hidden_formated = hidden_formated.drop(['FORCE_2020_LITHOFACIES_CONFIDENCE'], axis=1)
  hidden_formated['LITHO'] = le.transform(hidden_formated["LITHO"])

  return(training_formated, test_formated, hidden_formated, le)