
"""Additional functions

This script holds the 
1- matrix_score 
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
    matrix_path = '/content/gdrive/My Drive/Colab Notebooks/penalty_matrix.npy'
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
    c3 = np.unique(np.concatenate([l1, l2, l3]))
    class_names = le.classes_
    fig, ax = plot_confusion_matrix(conf_mat=cm,
                                    colorbar=False,
                                    show_absolute=False,
                                    show_normed=True,
                                    class_names=c3,
                                    figsize=(8, 8))
    plt.show()
