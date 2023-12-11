from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelBinarizer
import pdb
from sklearn.metrics import RocCurveDisplay, auc, roc_curve
from itertools import cycle

def to_array(file):
  return pd.read_csv(file).to_numpy()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("xTrain",
                        help="filename of the updated training data")
  parser.add_argument("yTrain",
                        help="filename of the updated training data")
  parser.add_argument("xTest",
                        help="filename of the updated training data")
  parser.add_argument("yTest",
                        help="filename of the updated training data")
  args = parser.parse_args()
  xTrain = to_array(args.xTrain)
  yTrain = to_array(args.yTrain).flatten()
  xTest = to_array(args.xTest)
  yTest= to_array(args.yTest).flatten()

  '''model'''
  knn = KNeighborsClassifier(metric="manhattan", n_neighbors=9)
  knn.fit(xTrain,yTrain)

  dt = DecisionTreeClassifier(criterion='gini', max_depth=10, min_samples_leaf=2)
  dt.fit(xTrain,yTrain)

  rf = RandomForestClassifier(max_depth=10,max_features='sqrt',min_samples_leaf=4,n_estimators=20)
  rf.fit(xTrain, yTrain)

  '''predictions'''

  knn_pred = knn.predict_proba(xTest)
  dt_pred = dt.predict_proba(xTest)
  rf_pred = rf.predict_proba(xTest)
  labels = ["0-20000", "20000-50000", "50000-100000", "100000-200000", 
            "200000-500000", "500000-1000000", "1000000-2000000","2000000-5000000",
            "5000000-10000000","10000000-20000000","20000000-50000000","50000000-100000000"]
  '''binarizing for One vs rest'''
  labbin = LabelBinarizer().fit(yTrain)
  test_oh = labbin.transform(yTest)
  macro_fpr, macro_tpr, macro_auc = dict(), dict(), dict()
  #knn
  '''macro average'''
  fpr, tpr, roc_auc = dict(), dict(), dict()
  for i in range(12):
    fpr[i], tpr[i], _ =roc_curve(test_oh[:,i],knn_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  fpr_grid = np.linspace(0.0,1.0,1000)

  mean_tpr = np.zeros_like(fpr_grid)
  for i in range(12):
    mean_tpr += np.interp(fpr_grid,fpr[i], tpr[i])

  mean_tpr /= 12

  fpr["macro"] = fpr_grid
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
  macro_fpr["knn"] = fpr["macro"]
  macro_tpr["knn"] = tpr["macro"]
  macro_auc["knn"] = roc_auc["macro"]
  '''plotting'''
  
  fig, ax = plt.subplots(figsize=(6,6))

  plt.plot(
    fpr["macro"],
    tpr["macro"],
    label = f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color = "deeppink",
    linestyle=":",
    linewidth = 4
  )
  colors = cycle(["salmon", "darkorange", "gold", "yellowgreen", "turquoise", "deepskyblue", "royalblue", "mediumpurple", "violet", "pink", "olive", "slategray"])
  for class_id, color in zip(range(12), colors):
    RocCurveDisplay.from_predictions(
    test_oh[:, class_id],
    knn_pred[:,class_id],
    name=f"{labels[class_id]} vs rest",
    color = color,
    ax = ax,
    plot_chance_level=(class_id == 2),
    )
  
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("One-vs-Rest ROC Curves")
  plt.legend()
  plt.show()
  
  #Decision Tree
  '''macro average'''
  fpr, tpr, roc_auc = dict(), dict(), dict()
  for i in range(12):
    fpr[i], tpr[i], _ =roc_curve(test_oh[:,i],dt_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  fpr_grid = np.linspace(0.0,1.0,1000)

  mean_tpr = np.zeros_like(fpr_grid)
  for i in range(12):
    mean_tpr += np.interp(fpr_grid,fpr[i], tpr[i])

  mean_tpr /= 12

  fpr["macro"] = fpr_grid
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  macro_fpr["dt"] = fpr["macro"]
  macro_tpr["dt"] = tpr["macro"]
  macro_auc["dt"] = roc_auc["macro"]
  
  '''plotting'''
  
  fig, ax = plt.subplots(figsize=(6,6))

  plt.plot(
    fpr["macro"],
    tpr["macro"],
    label = f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color = "deeppink",
    linestyle=":",
    linewidth = 4
  )
  colors = cycle(["salmon", "darkorange", "gold", "yellowgreen", "turquoise", "deepskyblue", "royalblue", "mediumpurple", "violet", "pink", "olive", "slategray"])
  for class_id, color in zip(range(12), colors):
    RocCurveDisplay.from_predictions(
    test_oh[:, class_id],
    dt_pred[:,class_id],
    name=f"{labels[class_id]} vs rest",
    color = color,
    ax = ax,
    plot_chance_level=(class_id == 2),
    )
  
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("One-vs-Rest ROC Curves")
  plt.legend()
  plt.show()
  
  # Random Forest
  '''macro average'''
  fpr, tpr, roc_auc = dict(), dict(), dict()
  for i in range(12):
    fpr[i], tpr[i], _ =roc_curve(test_oh[:,i],rf_pred[:,i])
    roc_auc[i] = auc(fpr[i], tpr[i])
  fpr_grid = np.linspace(0.0,1.0,1000)

  mean_tpr = np.zeros_like(fpr_grid)
  for i in range(12):
    mean_tpr += np.interp(fpr_grid,fpr[i], tpr[i])

  mean_tpr /= 12

  fpr["macro"] = fpr_grid
  tpr["macro"] = mean_tpr
  roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

  macro_fpr["rf"] = fpr["macro"]
  macro_tpr["rf"] = tpr["macro"]
  macro_auc["rf"] = roc_auc["macro"]
  
  '''plotting'''
  
  fig, ax = plt.subplots(figsize=(6,6))

  plt.plot(
    fpr["macro"],
    tpr["macro"],
    label = f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color = "deeppink",
    linestyle=":",
    linewidth = 4
  )
  colors = cycle(["salmon", "darkorange", "gold", "yellowgreen", "turquoise", "deepskyblue", "royalblue", "mediumpurple", "violet", "pink", "olive", "slategray"])
  for class_id, color in zip(range(12), colors):
    RocCurveDisplay.from_predictions(
    test_oh[:, class_id],
    rf_pred[:,class_id],
    name=f"{labels[class_id]} vs rest",
    color = color,
    ax = ax,
    plot_chance_level=(class_id == 2),
    )
  
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("One-vs-Rest ROC Curves")
  plt.legend()
  plt.show()

  '''plotting for macros only'''
  fig, ax = plt.subplots(figsize=(6,6))

  plt.plot(
    macro_fpr["knn"],
    macro_tpr["knn"],
    label = f"KNN macro-average ROC curve (AUC = {macro_auc['knn']:.5f})",
    color = "deeppink",
    linewidth = 2
  )

  plt.plot(
    macro_fpr["dt"],
    macro_tpr["dt"],
    label = f"DT macro-average ROC curve (AUC = {macro_auc['dt']:.5f})",
    color = "turquoise",
    linewidth = 2
  )

  plt.plot(
    macro_fpr["rf"],
    macro_tpr["rf"],
    label = f"RF macro-average ROC curve (AUC = {macro_auc['rf']:.5f})",
    color = "violet",
    linewidth = 2
  )
  plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Chance Level (AUC = 0.5)')
  plt.axis("square")
  plt.xlabel("False Positive Rate")
  plt.ylabel("True Positive Rate")
  plt.title("Macro Average ROC Curves")
  plt.legend()
  plt.show()
if __name__ == "__main__":
  main()
