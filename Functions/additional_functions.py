
"""Additional functions

This script holds the 
1- Penalty_score 
2- classification report (clf_rep)
3- plot confusion matrix (cm_plot)
for evaluation the classification performance.

They require some functionalities from libraries such as  pandas, numpy, matplotlib
"""

def matrix_score(
y_true:list,
y_pred:list
):
    """Returns the penalty matrix score obined by the predicted lithofacies a
    particular machine-learning model is able to provide. The matrix score was a metric 
    measure proposed by the FORCE commitee in order to provide the prediction performance
    measure from a petrophyicist perpective.

    Parameters
    ----------
    y_true: list
      The actual lithologies given by the datasets provider.
    y_pred: list
      The predicted lithofacies obtained by a particular machine learning model.

    Returns
    ----------
    matrix penaty score:
      Penalty matrix score obined by a particular machine-learning model.
    """
    
    import numpy as np
    matrix_path = '/content/gdrive/My Drive/Colab Notebooks/penalty_matrix_ordered.npy'
    A = np.load(matrix_path)
    S = 0.0
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)
    for i in range(0, y_true.shape[0]):
        S -= A[y_true[i], y_pred[i]]
    return S/y_true.shape[0]
    
# classification report function
def clf_rep(
y_true:list,
y_pred:list,
le): 
    import numpy as np
    l1 = le.inverse_transform(np.unique(y_true, return_counts=True)[0].astype(int))
    l2 = le.inverse_transform(np.unique(y_pred, return_counts=True)[0].astype(int))
    l3 = np.array([value for value in list(l2) if not value in list(l1)] + [value for value in list(l1) if not value in list(l2)], dtype='O')
    print('Real Facies (l1):', list(l1))
    print('Pred Facies (l2):', list(l2))
    print('Not exist in l1 or l2:', l3)
    
    c1 = l1
    c2 = l2 
    c3 = np.unique(np.concatenate([l1, l2, l3]))
    from sklearn.metrics import classification_report, confusion_matrix
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=c3))

def cm_plot(
y_true:list,
y_pred:list,
le
):
    import numpy as np 
    from sklearn.metrics import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix
    from matplotlib import pyplot as plt
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

    l1 = le.inverse_transform(np.unique(y_true, return_counts=True)[0].astype(int))
    l2 = le.inverse_transform(np.unique(y_pred, return_counts=True)[0].astype(int))
    l3 = np.array([value for value in list(l2) if not value in list(l1)] + [value for value in list(l1) if not value in list(l2)], dtype='O')
    c1 = l1
    c2 = l2 
    class_names = np.unique(np.concatenate([l1, l2, l3]))
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=False,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=class_names,
                                    figsize=(8, 8))
    plt.show()

def error_plot(y_true, y_pred, le):
  import matplotlib.pyplot as plt
  import numpy as np
  import pandas as pd
  from matplotlib.pyplot import figure
  df = pd.concat([pd.Series(y_true), pd.Series(y_pred)], axis=1)
  class_0 = [0 for i in range(12)]; class_1 = [0 for i in range(12)]; class_2 = [0 for i in range(12)]; class_3 = [0 for i in range(12)];
  class_4 = [0 for i in range(12)]; class_5 = [0 for i in range(12)]; class_6 = [0 for i in range(12)]; class_7 = [0 for i in range(12)];
  class_8 = [0 for i in range(12)]; class_9 = [0 for i in range(12)]; class_10 = [0 for i in range(12)]; class_11 = [0 for i in range(12)]


  for n in range(len([df[df[0]==i][1].value_counts() for i in sorted(df[0].unique())])):
    indexs = sorted([df[df[0]==i][1].value_counts() for i in sorted(df[0].unique())][n].index)
    for f in indexs:
      value = [df[df[0]==i][1].value_counts() for i in sorted(df[0].unique())][n].loc[f]
      if f == 0:
        class_0[n] = value / df.shape[0] * 100
      if f == 1:
        class_1[n] = value / df.shape[0] * 100
      if f == 2:
        class_2[n] = value / df.shape[0] * 100
      if f == 3:
        class_3[n] = value / df.shape[0] * 100
      if f == 4:
        class_4[n] = value / df.shape[0] * 100
      if f == 5:
        class_5[n] = value / df.shape[0] * 100
      if f == 6:
        class_6[n] = value / df.shape[0] * 100
      if f == 7:
        class_7[n] = value / df.shape[0] * 100
      if f == 8:
        class_8[n] = value / df.shape[0] * 100
      if f == 9:
        class_9[n] = value / df.shape[0] * 100
      if f == 10:
        class_10[n] = value / df.shape[0] * 100
      if f == 11:
        class_11[n] = value / df.shape[0] * 100
  class_num = [class_0, class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10, class_11]
  aa = [class_0[i]+class_1[i]+class_2[i]+class_3[i]+class_4[i]+class_5[i]+class_6[i]+class_7[i]+class_8[i]+class_9[i]+class_10[i]+class_11[i] for i in range(12)]
  for i, w in enumerate(aa):
    if w == 0:
      g=[]
      g.append(i)

  d = ['Anhy', 'Bsmt', 'Chlk', 'Coal', 'Dol', 'Hal', 'Lims', 'Marl', 'SS',
         'SS-Sh', 'Sh', 'Tuf']
  l1 = le.inverse_transform(np.unique(df[0], return_counts=True)[0].astype(int))
  l2 = le.inverse_transform(np.unique(df[1], return_counts=True)[0].astype(int))
  d_new = list(set(np.array([value for value in list(l1) if not value in g] + [value for value in list(l2) if not value in g])))
  c_index = le.transform(d_new)
  class_num_new = [v for v in class_num if sum(v)!=0]
  t1 = np.array(class_num_new).T.tolist()
  class_num_new_2 = [v for v in t1 if sum(v)!=0]
  t2 = np.array(class_num_new_2).T.tolist()



  # Set figure size
  colors = ['magenta','lawngreen','gold','lightblue','lightseagreen','cyan','darkorange','#228B22','grey','#FF4500','#000000']
  figure(figsize=(10, 10))

  # Plot stacked bar chart
  for c, v in enumerate(t2):
    plt.bar(sorted(d_new), t2[c], color=colors[c], label=sorted(d_new)[c])



  # Define labels

  plt.xlabel("actual class")
  plt.ylabel("Percent of predicted class")
  plt.yticks(np.arange(0, 61, 5))


  # Add legend

  plt.legend(loc=1)

  # Display

  plt.show()