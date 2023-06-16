
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
    """Returns the penalty matrix score obined by the predicted lithofacies.

    Parameters
    ----------
    y_true: list
      The actual lithologies.
    y_pred: list
      The predicted lithofacies.

    Returns
    ----------
    matrix penaty score:
      Penalty matrix score.
    """
    
    import numpy as np
    matrix_path = 'penalty_matrix_ordered.npy'
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
  from sklearn.metrics import confusion_matrix
  d = le.classes_
  l1 = le.inverse_transform(np.unique(y_true, return_counts=True)[0].astype(int))
  l2 = le.inverse_transform(np.unique(y_pred, return_counts=True)[0].astype(int))
  l3 = np.array([value for value in list(l2) if not value in list(l1)] + [value for value in list(l1) if not value in list(l2)], dtype='O')
  class_names = np.unique(np.concatenate([l1, l2, l3]))
  e = np.array([value for value in list(d) if not value in list(class_names)] + [value for value in list(class_names) if not value in list(d)], dtype='O')
  e_num = le.transform(e)

  for dd in e_num:
    class_num = confusion_matrix(y_true, y_pred).T
    class_num = np.insert(class_num, dd, np.array([0 for i in range(11)]), axis=1)
    class_num = np.insert(class_num, dd, np.array([0 for i in range(12)]), axis=0)
  c_index = le.transform(d)
  colors = ['tan', 'magenta', 'lawngreen', '#000000', 'gold', 'lightblue', 'lightseagreen', 'cyan', 'darkorange', '#228B22' , 'grey', '#FF4500']
  figure(figsize=(10, 10))
  # Plot stacked bar chart
  for i in range(12):
    plt.bar(d, class_num[i], color=colors[i], label=d[i])
  # Define labels
  plt.xlabel("actual class")
  plt.ylabel("Percent of predicted class")
  #plt.yticks(np.arange(0, 61, 5))
  # Add legend
  plt.legend(loc=1)
  # Display
  plt.show()