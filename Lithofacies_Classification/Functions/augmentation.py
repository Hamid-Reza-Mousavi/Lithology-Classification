
def features_augmentation(traindata, testdata, hiddendata):
  
  """Receives the pre-processed data and returns the data with additional
  columns named as cleaned data. 6 additional features icluding impedances 
  (S_I, P_I), bulk and shear modulus (K, G), slowness ratio (DT_R), and 
  true and measure depths ratio (MD_TVD).

  Parameters
  ----------
  training: Dataframe
    Pre-processed training dataframe.
  testdata: Dataframe
    Pre-processed open test dataframe.
  hiddendata: Dataframe
    Pre-processed hidden test dataframe.

  Returns
  ----------
  cleaned_traindata: Dataframe
    Cleaned trainig dataframe.
  cleaned_testdata: Dataframe
    Cleaned test dataframe.
  cleaned_hiddendata: Dataframe
    Cleaned hidden dataframe.
  """

  print('--------------------------------Creating additional features--------------------------------')
  # https://geoloil.com/computingGeomechanics.php
  # training Set
  traindata['S_I'] = traindata.RHOB * (1e6/traindata.DTS_COMB) # s-impedance
  traindata['P_I'] = traindata.RHOB * (1e6/traindata.DTC) #p-impedance
  traindata['DT_R'] = traindata.DTC / traindata.DTS_COMB # Shear wave slowness
  traindata['G'] = ((1e6/traindata.DTS_COMB)**2) * traindata.RHOB #Shear modulus
  traindata['K'] = (((1e6/traindata.DTC)**2) * traindata.RHOB) - (4 * traindata.G/3) # bulk modulus
  traindata['MD_TVD'] = -(traindata.DEPTH_MD/traindata.Z_LOC) #MD-TVD ratio

  # test Set
  testdata['S_I'] = testdata.RHOB * (1e6/testdata.DTS_COMB)
  testdata['P_I'] = testdata.RHOB * (1e6/testdata.DTC)
  testdata['DT_R'] = testdata.DTC / testdata.DTS_COMB
  testdata['G'] = ((1e6/testdata.DTS_COMB)**2) * testdata.RHOB
  testdata['K'] = (((1e6/testdata.DTC)**2) * testdata.RHOB) - (4 * testdata.G/3)
  testdata['MD_TVD'] = -(testdata.DEPTH_MD/testdata.Z_LOC)

  # hidden Set
  hiddendata['S_I'] = hiddendata.RHOB * (1e6/hiddendata.DTS_COMB)
  hiddendata['P_I'] = hiddendata.RHOB * (1e6/hiddendata.DTC)
  hiddendata['DT_R'] = hiddendata.DTC / hiddendata.DTS_COMB
  hiddendata['G'] = ((1e6/hiddendata.DTS_COMB)**2) * hiddendata.RHOB
  hiddendata['K'] = (((1e6/hiddendata.DTC)**2) * hiddendata.RHOB) - (4 * hiddendata.G/3)
  hiddendata['MD_TVD'] = -(hiddendata.DEPTH_MD/hiddendata.Z_LOC)
  print('Shape of datasets after agumation >>> train: {} | test: {} | hidden: {}'.format(traindata.shape, testdata.shape, hiddendata.shape))

  # print column names held on datasets
  print('Features included in the datasets: {}'.format(traindata.columns)) 
  return traindata, testdata, hiddendata
