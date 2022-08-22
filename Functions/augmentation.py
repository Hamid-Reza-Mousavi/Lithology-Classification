
"""features_augmentation

This script test three diffrent machine learning regressors in order
to predict the four most relevant wireline logs,DTC, NPHI, DTS, and RHOB.
Afterwards, the models' results are compared between each other in each
prediction stage in order to select the best performing model. Later, 
at each stage the missing instances encountered in the training, open test,
and hidden test datasets are imputed by the predictions otained by the best
ML model at each stage. Finally, 6 additional features are included in each
data subset.

It requires xgboost, lightgbm, catboost to be installed before running, as
well as pandas and numpy functionalities.
"""

def features_augmentation(traindata, testdata, hiddendata):
  
  
  print('--------------------------------Creating additional features--------------------------------')

  # training Set
  traindata['S_I'] = traindata.RHOB * (1e6/traindata.DTS_COMB) # s-impedance
  traindata['P_I'] = traindata.RHOB * (1e6/traindata.DTC) #p-impedance
  traindata['DT_R'] = traindata.DTC / traindata.DTS_COMB # slowness ratio
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
